"""
Cross-Dataset Evaluation for Bengali Scene Text Recognition Paper
==================================================================
Evaluates the best Florence-2 model on external benchmark datasets.

Supported datasets:
  1. IndicSTR12 Bengali test split (ICDAR 2023, IIIT Hyderabad)
  2. BSTD Bengali test split (IIT Jodhpur, 2025)

This script provides download helpers and standardized evaluation.

Usage:
  python cross_dataset_eval.py --dataset indicstr12 --model_dir checkpoints_florence2_grapheme/best_model
  python cross_dataset_eval.py --dataset bstd --model_dir checkpoints_florence2_grapheme/best_model
  python cross_dataset_eval.py --all --model_dir checkpoints_florence2_grapheme/best_model
"""
import os
import sys
import json
import argparse
import time
from typing import List, Dict

sys.stdout.reconfigure(encoding='utf-8', errors='replace')
BASE_DIR = os.path.abspath(os.path.dirname(__file__))
sys.path.insert(0, BASE_DIR)

from metrics import evaluate_corpus, format_results_table, FPSTimer


# ── Dataset Downloaders ───────────────────────────────────────────
INDICSTR12_INFO = {
    'name': 'IndicSTR12',
    'url': 'https://github.com/AiswaryaNambiar/IndicSTR12',
    'paper': 'Nambiar et al., ICDAR 2023',
    'dir': os.path.join(BASE_DIR, 'external_datasets', 'indicstr12_bengali'),
}

BSTD_INFO = {
    'name': 'BSTD (Bharat Scene Text)',
    'url': 'https://github.com/suryadevsingh/BSTD',
    'paper': 'Singh et al., 2025',
    'dir': os.path.join(BASE_DIR, 'external_datasets', 'bstd_bengali'),
}


def load_external_dataset(dataset_dir: str, dataset_name: str) -> List[Dict]:
    """Load an external dataset. Expects:
      - images/ folder with .jpg/.png files
      - labels.json or GT/ folder with .txt files
    """
    pairs = []

    # Try labels.json format first
    labels_json = os.path.join(dataset_dir, 'labels.json')
    if os.path.exists(labels_json):
        with open(labels_json, 'r', encoding='utf-8') as f:
            labels = json.load(f)
        for item in labels:
            img_path = os.path.join(dataset_dir, item.get('image', item.get('img', '')))
            gt = item.get('label', item.get('text', item.get('gt', '')))
            if os.path.exists(img_path) and gt and gt != '###':
                pairs.append({'image': img_path, 'gt': gt})

    # Try GT/ folder format
    elif os.path.exists(os.path.join(dataset_dir, 'GT')):
        img_dir = os.path.join(dataset_dir, 'images')
        gt_dir = os.path.join(dataset_dir, 'GT')

        if not os.path.exists(img_dir):
            img_dir = dataset_dir

        for gt_file in sorted(os.listdir(gt_dir)):
            if not gt_file.endswith('.txt'):
                continue
            name = gt_file.replace('.txt', '')
            gt_text = open(os.path.join(gt_dir, gt_file), 'r', encoding='utf-8').read().strip()
            if not gt_text or gt_text == '###':
                continue

            # Find matching image
            for ext in ['.jpg', '.png', '.jpeg', '.bmp']:
                img_path = os.path.join(img_dir, name + ext)
                if os.path.exists(img_path):
                    pairs.append({'image': img_path, 'gt': gt_text})
                    break

    # Try CSV format
    elif os.path.exists(os.path.join(dataset_dir, 'labels.csv')):
        import csv
        with open(os.path.join(dataset_dir, 'labels.csv'), 'r', encoding='utf-8') as f:
            reader = csv.reader(f)
            header = next(reader, None)
            for row in reader:
                if len(row) >= 2:
                    img_path = os.path.join(dataset_dir, row[0])
                    gt = row[1].strip()
                    if os.path.exists(img_path) and gt and gt != '###':
                        pairs.append({'image': img_path, 'gt': gt})

    if not pairs:
        # Last attempt: look for images + same-name txt files
        for f in sorted(os.listdir(dataset_dir)):
            if f.lower().endswith(('.jpg', '.png', '.jpeg')):
                name = os.path.splitext(f)[0]
                txt_path = os.path.join(dataset_dir, name + '.txt')
                if os.path.exists(txt_path):
                    gt = open(txt_path, 'r', encoding='utf-8').read().strip()
                    if gt and gt != '###':
                        pairs.append({
                            'image': os.path.join(dataset_dir, f),
                            'gt': gt,
                        })

    print(f"  Loaded {len(pairs)} samples from {dataset_name}")
    return pairs


# ── Florence-2 Evaluation ─────────────────────────────────────────
def evaluate_florence2_on_dataset(pairs: List[Dict], model_dir: str, dataset_name: str) -> Dict:
    """Run Florence-2 inference on external dataset."""
    import torch
    from PIL import Image

    try:
        from transformers import AutoModelForCausalLM, AutoProcessor
        from peft import PeftModel
    except ImportError:
        print("ERROR: Install transformers, peft")
        return None

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"  Device: {device}")

    # Load model
    print(f"  Loading model from {model_dir}...")

    # Check if this is a LoRA checkpoint
    if os.path.exists(os.path.join(model_dir, 'adapter_config.json')):
        # Load base model + LoRA adapter
        with open(os.path.join(model_dir, 'adapter_config.json'), 'r') as f:
            adapter_cfg = json.load(f)
        base_model_id = adapter_cfg.get('base_model_name_or_path', 'microsoft/Florence-2-base')

        processor = AutoProcessor.from_pretrained(model_dir, trust_remote_code=True)
        base_model = AutoModelForCausalLM.from_pretrained(
            base_model_id, trust_remote_code=True, torch_dtype=torch.float32
        )

        # Check if token embeddings were resized
        if len(processor.tokenizer) != base_model.get_input_embeddings().weight.shape[0]:
            base_model.resize_token_embeddings(len(processor.tokenizer))

        model = PeftModel.from_pretrained(base_model, model_dir)
    else:
        processor = AutoProcessor.from_pretrained(model_dir, trust_remote_code=True)
        model = AutoModelForCausalLM.from_pretrained(
            model_dir, trust_remote_code=True, torch_dtype=torch.float32
        )

    model.to(device)
    model.eval()

    # Run inference
    ground_truths = []
    predictions = []

    timer = FPSTimer()
    timer.__enter__()

    for i, pair in enumerate(pairs):
        try:
            image = Image.open(pair['image']).convert('RGB')
        except:
            predictions.append('')
            ground_truths.append(pair['gt'])
            continue

        inputs = processor(images=image, text="<OCR>", return_tensors="pt").to(device)

        with torch.no_grad():
            gen_ids = model.generate(**inputs, max_new_tokens=128, num_beams=3)

        pred = processor.batch_decode(gen_ids, skip_special_tokens=True)[0]
        predictions.append(pred.strip())
        ground_truths.append(pair['gt'])
        timer.num_images += 1

        if (i + 1) % 100 == 0:
            print(f"  [{i+1}/{len(pairs)}] processed")

    timer.__exit__(None, None, None)

    results = evaluate_corpus(ground_truths, predictions)
    results['fps'] = timer.fps
    results['dataset'] = dataset_name

    print(format_results_table(results, f"Florence-2 on {dataset_name}"))
    return results


# ── Cross-Dataset Table ───────────────────────────────────────────
def generate_cross_dataset_table(all_results: List[Dict]):
    """Generate cross-dataset comparison table."""
    print(f"\n{'='*80}")
    print(f"  CROSS-DATASET EVALUATION TABLE (for paper)")
    print(f"{'='*80}")

    header = f"{'Dataset':<30} {'WRR%':>7} {'CER%':>7} {'1-NED%':>7} {'Samples':>8}"
    print(header)
    print('-' * 65)

    rows = []
    for r in all_results:
        if r is None:
            continue
        row = {
            'dataset': r.get('dataset', '?'),
            'WRR': r.get('WRR', 0),
            'CER': r.get('CER', 100),
            '1-NED': r.get('1-NED', 0),
            'num_samples': r.get('num_samples', 0),
        }
        print(f"{row['dataset']:<30} {row['WRR']:>7.2f} {row['CER']:>7.2f} "
              f"{row['1-NED']:>7.2f} {row['num_samples']:>8}")
        rows.append(row)

    print('-' * 65)

    # Save
    table_path = os.path.join(BASE_DIR, 'cross_dataset_table.json')
    with open(table_path, 'w', encoding='utf-8') as f:
        json.dump(rows, f, indent=2)
    print(f"\nSaved: {table_path}")

    # LaTeX
    latex_path = os.path.join(BASE_DIR, 'cross_dataset_table.tex')
    with open(latex_path, 'w', encoding='utf-8') as f:
        f.write("\\begin{table}[h]\n\\centering\n")
        f.write("\\caption{Cross-dataset evaluation of our method on Bengali scene text benchmarks.}\n")
        f.write("\\label{tab:crossdataset}\n")
        f.write("\\begin{tabular}{lcccc}\n\\toprule\n")
        f.write("Dataset & WRR(\\%) & CER(\\%) & 1-NED(\\%) & Samples \\\\\n\\midrule\n")
        for row in rows:
            f.write(f"{row['dataset']} & {row['WRR']:.2f} & {row['CER']:.2f} & "
                    f"{row['1-NED']:.2f} & {row['num_samples']} \\\\\n")
        f.write("\\bottomrule\n\\end{tabular}\n\\end{table}\n")
    print(f"LaTeX: {latex_path}")


# ── Main ───────────────────────────────────────────────────────────
def main():
    parser = argparse.ArgumentParser(description="Cross-dataset evaluation")
    parser.add_argument('--dataset', choices=['indicstr12', 'bstd', 'all'], default='all')
    parser.add_argument('--model_dir', required=True, help='Path to Florence-2 checkpoint')
    parser.add_argument('--dataset_dir', default=None,
                        help='Override dataset directory')
    args = parser.parse_args()

    print("=" * 60)
    print("  Cross-Dataset Evaluation")
    print("=" * 60)

    datasets = []
    if args.dataset in ('indicstr12', 'all'):
        d = args.dataset_dir or INDICSTR12_INFO['dir']
        if os.path.exists(d):
            datasets.append(('IndicSTR12 Bengali', d))
        else:
            print(f"\n  IndicSTR12 not found at {d}")
            print(f"  Download from: {INDICSTR12_INFO['url']}")
            print(f"  Place Bengali test images + labels in: {d}")

    if args.dataset in ('bstd', 'all'):
        d = args.dataset_dir or BSTD_INFO['dir']
        if os.path.exists(d):
            datasets.append(('BSTD Bengali', d))
        else:
            print(f"\n  BSTD not found at {d}")
            print(f"  Download from: {BSTD_INFO['url']}")
            print(f"  Place Bengali test images + labels in: {d}")

    # Always include our dataset
    splits_path = os.path.join(BASE_DIR, 'florence2_splits.json')
    if os.path.exists(splits_path):
        with open(splits_path, 'r', encoding='utf-8') as f:
            splits = json.load(f)
        our_pairs = splits['test']
        datasets.insert(0, ('Our Dataset', None))  # Special case

    if not datasets:
        print("\nNo datasets found. Please download external datasets first.")
        return

    all_results = []
    for name, d in datasets:
        if d is None:  # Our dataset
            pairs = our_pairs
            print(f"\n  Dataset: {name} ({len(pairs)} samples)")
        else:
            pairs = load_external_dataset(d, name)

        if pairs:
            result = evaluate_florence2_on_dataset(pairs, args.model_dir, name)
            all_results.append(result)

    if all_results:
        generate_cross_dataset_table(all_results)

    print(f"\n{'='*60}")
    print(f"  Cross-dataset evaluation complete!")
    print(f"{'='*60}")


if __name__ == '__main__':
    main()
