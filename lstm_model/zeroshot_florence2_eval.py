"""
Zero-shot Florence-2 evaluation on the Bengali OCR test set.

Mirrors run_evaluation() in train_florence2.py but does NOT load any
fine-tuned adapter — runs the raw `microsoft/Florence-2-base` model
with the `<OCR>` task prompt. Same 307 test images, same beam search
(num_beams=3), same metrics, so results are directly comparable to
the STANDARD and GRAPHEME runs in the paper.

Outputs: florence2_results_zeroshot.json (same schema as the
fine-tuned eval outputs).
"""
import os
import sys
import json
import argparse
from typing import Optional, List

sys.stdout.reconfigure(encoding='utf-8', errors='replace')

# Reuse the flash_attn patch + image resolution from train_florence2.py
# so behaviour matches the fine-tuned eval exactly.
BASE_DIR = os.path.abspath(os.path.dirname(__file__))
sys.path.insert(0, BASE_DIR)

from train_florence2 import (  # noqa: E402
    _patch_flash_attn,
    resolve_image_path,
    validate_and_resolve_pairs,
    SPLITS_FILE,
    DEFAULT_MODEL_ID,
    TASK_PROMPT,
    DEFAULT_CONFIG,
    set_seed,
    DEFAULT_SEED,
)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--model_id', default=DEFAULT_MODEL_ID)
    parser.add_argument('--splits_file', default=SPLITS_FILE)
    parser.add_argument('--seed', type=int, default=DEFAULT_SEED)
    args = parser.parse_args()

    _patch_flash_attn()
    set_seed(args.seed)

    print("=" * 60)
    print("  Florence-2 Bengali OCR — ZERO-SHOT baseline")
    print(f"  Model: {args.model_id} (no fine-tuning)")
    print("=" * 60)

    import torch
    from PIL import Image
    from transformers import AutoModelForCausalLM, AutoProcessor, AutoConfig

    print("\n[1/4] Loading test split...")
    with open(args.splits_file, 'r', encoding='utf-8') as f:
        splits = json.load(f)
    raw_test = splits['test']
    test_pairs, missing = validate_and_resolve_pairs(
        'test', raw_test, filter_missing=False
    )
    assert missing == 0, f"{missing} test images missing"

    print(f"\n[2/4] Loading model {args.model_id} (CPU init, then GPU)...")
    processor = AutoProcessor.from_pretrained(args.model_id, trust_remote_code=True)
    config = AutoConfig.from_pretrained(args.model_id, trust_remote_code=True)
    model = AutoModelForCausalLM.from_pretrained(
        args.model_id,
        config=config,
        trust_remote_code=True,
        torch_dtype=torch.float32,
        attn_implementation="eager",
    )
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model.to(device)
    model.eval()
    print(f"  Device: {device}")

    print(f"\n[3/4] Running inference on {len(test_pairs)} test images (beam=3)...")
    from metrics import evaluate_corpus, format_results_table, FPSTimer

    predictions: List[str] = []
    ground_truths: List[str] = []

    timer = FPSTimer()
    timer.__enter__()
    for i, pair in enumerate(test_pairs):
        try:
            image = Image.open(pair['image']).convert('RGB')
        except Exception:
            predictions.append('')
            ground_truths.append(pair['gt'])
            continue
        inputs = processor(
            images=image, text=TASK_PROMPT, return_tensors="pt"
        ).to(device)
        with torch.no_grad():
            generated_ids = model.generate(
                **inputs,
                max_new_tokens=DEFAULT_CONFIG['max_length'],
                num_beams=3,
            )
        pred = processor.batch_decode(generated_ids, skip_special_tokens=True)[0]
        predictions.append(pred.strip())
        ground_truths.append(pair['gt'])
        timer.num_images += 1
        if (i + 1) % 50 == 0:
            print(f"    [{i+1}/{len(test_pairs)}] processed")
    timer.__exit__(None, None, None)

    print(f"\n[4/4] Computing metrics...")
    results = evaluate_corpus(ground_truths, predictions, verbose=True)
    results['fps'] = timer.fps
    results['mode'] = 'ZEROSHOT'
    print(format_results_table(results, "Florence-2 (ZERO-SHOT)"))
    print(f"  Inference Speed: {timer.fps:.2f} FPS")

    save = {k: v for k, v in results.items() if k != 'per_sample'}
    save['predictions'] = [
        {'gt': g, 'pred': p} for g, p in zip(ground_truths, predictions)
    ]
    out = os.path.join(BASE_DIR, 'florence2_results_zeroshot.json')
    with open(out, 'w', encoding='utf-8') as f:
        json.dump(save, f, ensure_ascii=False, indent=2)
    print(f"\n  Results saved: {out}")


if __name__ == '__main__':
    main()
