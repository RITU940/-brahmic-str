"""
Baselines Evaluation for Bengali Scene Text Recognition Paper
==============================================================
Evaluates all baseline models on the SAME test split for fair comparison.

Baselines:
  1. Tesseract 5 (ben) — off-the-shelf OCR
  2. CRNN (best_model_v3.pth) — existing trained model
  3. TrOCR-small — transformer OCR baseline
  4. Florence-2 standard tokenizer — VLM baseline
  5. Florence-2 + grapheme tokenizer — OUR METHOD

This script handles #1 and #2 (Tesseract + CRNN).
Florence-2 baselines are run via train_florence2.py --evaluate.

Usage:
  python run_baselines.py                    # Run all available baselines
  python run_baselines.py --tesseract_only   # Just Tesseract
  python run_baselines.py --crnn_only        # Just CRNN
"""
import os
import sys
import json
import time
import argparse
import subprocess
from typing import List, Dict

sys.stdout.reconfigure(encoding='utf-8', errors='replace')

BASE_DIR = os.path.abspath(os.path.dirname(__file__))
sys.path.insert(0, BASE_DIR)

from metrics import evaluate_corpus, format_results_table, FPSTimer, normalize_bengali


# ── Load Test Split ────────────────────────────────────────────────
def load_test_split() -> List[Dict]:
    """Load test split from florence2_splits.json."""
    splits_path = os.path.join(BASE_DIR, 'florence2_splits.json')
    if not os.path.exists(splits_path):
        print("ERROR: florence2_splits.json not found. Run prepare_florence2_data.py first!")
        sys.exit(1)
    
    with open(splits_path, 'r', encoding='utf-8') as f:
        data = json.load(f)
    
    return data['test']


# ── Tesseract Baseline ─────────────────────────────────────────────
def run_tesseract_baseline(test_pairs: List[Dict], lang='ben', psm=7) -> Dict:
    """Evaluate Tesseract OCR on the test split."""
    print(f"\n{'='*60}")
    print(f"  Baseline: Tesseract (lang={lang})")
    print(f"{'='*60}")
    
    tessdata_dir = os.path.join(BASE_DIR, 'tessdata')
    
    # Check if tesseract is available
    try:
        result = subprocess.run(['tesseract', '--version'], capture_output=True, text=True, timeout=10)
        tess_version = result.stdout.split('\n')[0] if result.stdout else 'unknown'
        print(f"  Tesseract version: {tess_version}")
    except (FileNotFoundError, subprocess.TimeoutExpired):
        print("  WARNING: Tesseract not found. Skipping.")
        return None
    
    # Check if language data exists
    traineddata = os.path.join(tessdata_dir, f'{lang}.traineddata')
    if not os.path.exists(traineddata):
        # Try system tessdata
        tessdata_dir = ''
        print(f"  Using system tessdata (no custom {lang}.traineddata found)")
    
    ground_truths = []
    predictions = []
    
    timer = FPSTimer()
    timer.__enter__()
    
    for i, pair in enumerate(test_pairs):
        gt = pair['gt']
        img_path = pair['image']
        ground_truths.append(gt)
        
        try:
            cmd = ['tesseract', img_path, 'stdout', '-l', lang, '--psm', str(psm)]
            if tessdata_dir:
                cmd.extend(['--tessdata-dir', tessdata_dir])
            
            result = subprocess.run(
                cmd, capture_output=True, text=True, timeout=30,
                encoding='utf-8', errors='replace'
            )
            pred = result.stdout.strip()
        except (subprocess.TimeoutExpired, Exception):
            pred = ''
        
        predictions.append(pred)
        timer.num_images += 1
        
        if (i + 1) % 200 == 0:
            print(f"  [{i+1}/{len(test_pairs)}] processed")
    
    timer.__exit__(None, None, None)
    
    results = evaluate_corpus(ground_truths, predictions)
    results['fps'] = timer.fps
    results['model_name'] = f'Tesseract ({lang})'
    results['model_params'] = 'N/A'
    
    print(format_results_table(results, f'Tesseract ({lang})'))
    print(f"  Speed: {timer.fps:.1f} FPS")
    
    # Save predictions for analysis
    save_path = os.path.join(BASE_DIR, f'baseline_tesseract_{lang}_results.json')
    save_data = {k: v for k, v in results.items() if k != 'per_sample'}
    save_data['predictions'] = [
        {'gt': gt, 'pred': pred}
        for gt, pred in zip(ground_truths, predictions)
    ]
    with open(save_path, 'w', encoding='utf-8') as f:
        json.dump(save_data, f, ensure_ascii=False, indent=2)
    print(f"  Saved: {save_path}")
    
    return results


# ── CRNN Baseline ──────────────────────────────────────────────────
def run_crnn_baseline(test_pairs: List[Dict]) -> Dict:
    """Evaluate the existing CRNN model (best_model_v3.pth)."""
    print(f"\n{'='*60}")
    print(f"  Baseline: CRNN (best_model_v3.pth)")
    print(f"{'='*60}")
    
    model_path = os.path.join(BASE_DIR, 'best_model_v3.pth')
    if not os.path.exists(model_path):
        print("  WARNING: best_model_v3.pth not found. Skipping CRNN baseline.")
        return None
    
    # Check if required modules are available
    try:
        import torch
        import numpy as np
        from PIL import Image
        # The checkpoint was trained with CRNN_V3 (ResNet-style) from colab_train_v3.py
        from colab_train_v3 import CRNN_V3
    except ImportError as e:
        print(f"  ERROR: Missing dependency: {e}")
        return None
    
    # Load checkpoint to get config
    print("  Loading checkpoint...")
    checkpoint = torch.load(model_path, map_location='cpu', weights_only=False)
    
    # Extract model config from checkpoint
    cfg = checkpoint.get('config', {})
    num_classes = checkpoint.get('num_classes', cfg.get('num_classes', 200))
    img_height = cfg.get('img_height', 64)
    img_width = cfg.get('img_width', 256)
    hidden_size = cfg.get('hidden_size', 512)
    
    # Load character mapping from checkpoint
    char2idx = checkpoint.get('char2idx', {})
    if 'idx2char' in checkpoint:
        idx2char = {int(k): v for k, v in checkpoint['idx2char'].items()}
    else:
        idx2char = {v: k for k, v in char2idx.items()}
    
    print(f"  Config: img={img_height}x{img_width}, hidden={hidden_size}, classes={num_classes}")
    print(f"  Checkpoint epoch: {checkpoint.get('epoch', '?')}")
    print(f"  Checkpoint CER: {checkpoint.get('val_cer', '?')}")
    print(f"  Checkpoint Char Acc: {checkpoint.get('val_char_acc', '?')}")
    
    # Build model (CRNN_V3 = ResNet-style architecture)
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model = CRNN_V3(num_classes=num_classes, img_height=img_height, hidden_size=hidden_size)
    
    state_dict = checkpoint.get('model_state_dict', checkpoint)
    model.load_state_dict(state_dict, strict=True)
    print("  Model loaded successfully (strict=True)")
    
    model.to(device)
    model.eval()
    
    total_params = sum(p.numel() for p in model.parameters())
    print(f"  Model params: {total_params:,}")
    print(f"  Device: {device}")
    
    # Inference
    ground_truths = []
    predictions = []
    
    timer = FPSTimer()
    timer.__enter__()
    
    for i, pair in enumerate(test_pairs):
        gt = pair['gt']
        img_path = pair['image']
        ground_truths.append(gt)
        
        try:
            img = Image.open(img_path).convert('L')
            
            # Preprocess (same as dataset.py)
            w, h = img.size
            scale = img_height / h
            target_w = min(int(w * scale), img_width)
            img = img.resize((target_w, img_height), Image.LANCZOS)
            
            padded = Image.new('L', (img_width, img_height), 255)
            padded.paste(img, (0, 0))
            
            img_array = np.array(padded, dtype=np.float32) / 255.0
            img_array = 1.0 - img_array
            img_tensor = torch.FloatTensor(img_array).unsqueeze(0).unsqueeze(0).to(device)
            
            with torch.no_grad():
                output = model(img_tensor)  # (seq_len, 1, num_classes)
                output = output.squeeze(1)   # (seq_len, num_classes)
                _, preds_idx = output.max(1)  # (seq_len,)
            
            # CTC decode (greedy)
            pred_indices = preds_idx.cpu().numpy().tolist()
            decoded = []
            prev_idx = -1
            for idx in pred_indices:
                if idx != 0 and idx != prev_idx:  # 0 = CTC blank
                    if idx in idx2char:
                        decoded.append(idx2char[idx])
                prev_idx = idx
            
            pred = ''.join(decoded)
        except Exception as e:
            pred = ''
        
        predictions.append(pred)
        timer.num_images += 1
        
        if (i + 1) % 200 == 0:
            print(f"  [{i+1}/{len(test_pairs)}] processed")
    
    timer.__exit__(None, None, None)
    
    results = evaluate_corpus(ground_truths, predictions)
    results['fps'] = timer.fps
    results['model_name'] = 'CRNN (BiLSTM-CTC)'
    results['model_params'] = f'{total_params:,}'
    
    print(format_results_table(results, 'CRNN (BiLSTM-CTC)'))
    print(f"  Speed: {timer.fps:.1f} FPS")
    
    # Save
    save_path = os.path.join(BASE_DIR, 'baseline_crnn_results.json')
    save_data = {k: v for k, v in results.items() if k != 'per_sample'}
    save_data['predictions'] = [
        {'gt': gt, 'pred': pred}
        for gt, pred in zip(ground_truths, predictions)
    ]
    with open(save_path, 'w', encoding='utf-8') as f:
        json.dump(save_data, f, ensure_ascii=False, indent=2)
    print(f"  Saved: {save_path}")
    
    return results


# ── Comparison Table ───────────────────────────────────────────────
def generate_comparison_table(all_results: List[Dict]):
    """Generate a comparison table for the paper."""
    print(f"\n{'='*80}")
    print(f"  COMPARISON TABLE (for paper)")
    print(f"{'='*80}")
    
    header = f"{'Model':<30} {'WRR%':>7} {'CER%':>7} {'1-NED%':>7} {'FPS':>7} {'Params':>12}"
    print(header)
    print('-' * 80)
    
    rows = []
    for r in all_results:
        if r is None:
            continue
        name = r.get('model_name', 'Unknown')
        wrr = r.get('WRR', 0)
        cer = r.get('CER', 100)
        one_ned = r.get('1-NED', 0)
        fps = r.get('fps', 0)
        params = r.get('model_params', 'N/A')
        
        row = f"{name:<30} {wrr:>7.2f} {cer:>7.2f} {one_ned:>7.2f} {fps:>7.1f} {str(params):>12}"
        print(row)
        rows.append({
            'model': name, 'WRR': wrr, 'CER': cer,
            '1-NED': one_ned, 'FPS': fps, 'params': params
        })
    
    print('-' * 80)
    
    # Save as JSON for paper table generation
    table_path = os.path.join(BASE_DIR, 'paper_comparison_table.json')
    with open(table_path, 'w', encoding='utf-8') as f:
        json.dump(rows, f, indent=2)
    print(f"\nTable saved: {table_path}")
    
    # Also generate LaTeX table
    latex_path = os.path.join(BASE_DIR, 'paper_table.tex')
    with open(latex_path, 'w', encoding='utf-8') as f:
        f.write("\\begin{table}[h]\n")
        f.write("\\centering\n")
        f.write("\\caption{Comparison of Bengali scene text recognition methods on our dataset.}\n")
        f.write("\\label{tab:comparison}\n")
        f.write("\\begin{tabular}{lccccc}\n")
        f.write("\\toprule\n")
        f.write("Model & WRR(\\%) & CER(\\%) & 1-NED(\\%) & FPS & Params \\\\\n")
        f.write("\\midrule\n")
        for row in rows:
            f.write(f"{row['model']} & {row['WRR']:.2f} & {row['CER']:.2f} & "
                    f"{row['1-NED']:.2f} & {row['FPS']:.1f} & {row['params']} \\\\\n")
        f.write("\\bottomrule\n")
        f.write("\\end{tabular}\n")
        f.write("\\end{table}\n")
    print(f"LaTeX table saved: {latex_path}")


# ── Main ───────────────────────────────────────────────────────────
def main():
    parser = argparse.ArgumentParser(description="Run baselines for Bengali OCR paper")
    parser.add_argument('--tesseract_only', action='store_true')
    parser.add_argument('--crnn_only', action='store_true')
    parser.add_argument('--tesseract_lang', default='ben',
                        help='Tesseract language code (default: ben)')
    args = parser.parse_args()
    
    print("=" * 60)
    print("  Bengali Scene Text — Baselines Evaluation")
    print("=" * 60)
    
    test_pairs = load_test_split()
    print(f"  Test samples: {len(test_pairs)}")
    
    all_results = []
    
    if not args.crnn_only:
        # Tesseract baseline
        tess_result = run_tesseract_baseline(test_pairs, lang=args.tesseract_lang)
        all_results.append(tess_result)
    
    if not args.tesseract_only:
        # CRNN baseline
        crnn_result = run_crnn_baseline(test_pairs)
        all_results.append(crnn_result)
    
    # Load any Florence-2 results if they exist
    for mode in ['standard', 'grapheme']:
        result_path = os.path.join(BASE_DIR, f'florence2_results_{mode}.json')
        if os.path.exists(result_path):
            with open(result_path, 'r', encoding='utf-8') as f:
                f2_result = json.load(f)
            f2_result['model_name'] = f'Florence-2 ({mode})'
            all_results.append(f2_result)
    
    # Generate comparison table
    if all_results:
        generate_comparison_table(all_results)
    
    print(f"\n{'='*60}")
    print(f"  Baselines complete!")
    print(f"  Next: Run Florence-2 training with train_florence2.py")
    print(f"{'='*60}")


if __name__ == '__main__':
    main()
