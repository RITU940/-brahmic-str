"""
Florence-2 Quick Validation — CPU Inference Test
==================================================
Tests Florence-2 on a small subset to validate the entire pipeline works.
This runs on CPU (slow) but proves the method before Colab training.

Usage:
  python validate_florence2_pipeline.py
  python validate_florence2_pipeline.py --num_samples 20 --use_graphemes
"""
import os
import sys
import json
import time
import unicodedata
from collections import Counter

sys.stdout.reconfigure(encoding='utf-8', errors='replace')
BASE_DIR = os.path.abspath(os.path.dirname(__file__))

def normalize_text(text):
    text = unicodedata.normalize('NFC', text.strip())
    for zw in ['\u200c', '\u200d', '\u200b', '\ufeff']:
        text = text.replace(zw, '')
    return ' '.join(text.split())

def edit_distance(ref, hyp):
    m, n = len(ref), len(hyp)
    if m == 0: return n
    if n == 0: return m
    prev = list(range(n + 1))
    for i in range(1, m + 1):
        curr = [i] + [0] * n
        for j in range(1, n + 1):
            cost = 0 if ref[i-1] == hyp[j-1] else 1
            curr[j] = min(prev[j] + 1, curr[j-1] + 1, prev[j-1] + cost)
        prev = curr
    return prev[n]

# ── Grapheme tokenizer (embedded) ──
BENGALI_VOWEL_SIGNS = set('\u09be\u09bf\u09c0\u09c1\u09c2\u09c3\u09c4\u09c7\u09c8\u09cb\u09cc\u09d7')
BENGALI_HALANT = '\u09cd'
BENGALI_CONSONANTS = set(
    '\u0995\u0996\u0997\u0998\u0999\u099a\u099b\u099c\u099d\u099e'
    '\u099f\u09a0\u09a1\u09a2\u09a3\u09a4\u09a5\u09a6\u09a7\u09a8'
    '\u09aa\u09ab\u09ac\u09ad\u09ae\u09af\u09b0\u09b2'
    '\u09b6\u09b7\u09b8\u09b9\u09dc\u09dd\u09df\u09f0\u09f1'
)
BENGALI_VOWELS = set('\u0985\u0986\u0987\u0988\u0989\u098a\u098b\u098c\u098f\u0990\u0993\u0994\u09e0\u09e1')
BENGALI_MODIFIERS = set('\u0981\u0982\u0983\u09bc\u09be')
BENGALI_DIGITS = set('\u09e6\u09e7\u09e8\u09e9\u09ea\u09eb\u09ec\u09ed\u09ee\u09ef')
BENGALI_BASE = BENGALI_CONSONANTS | BENGALI_VOWELS | BENGALI_DIGITS

def segment_graphemes(text):
    text = unicodedata.normalize('NFC', text)
    graphemes = []
    i = 0
    while i < len(text):
        ch = text[i]
        if ch in BENGALI_BASE:
            cluster = ch
            i += 1
            if ch in BENGALI_CONSONANTS:
                while i < len(text) - 1 and text[i] == BENGALI_HALANT and text[i+1] in BENGALI_CONSONANTS:
                    cluster += text[i] + text[i+1]
                    i += 2
            while i < len(text) and text[i] in BENGALI_VOWEL_SIGNS:
                cluster += text[i]
                i += 1
            while i < len(text) and text[i] in BENGALI_MODIFIERS:
                cluster += text[i]
                i += 1
            if i < len(text) and text[i] == BENGALI_HALANT:
                cluster += text[i]
                i += 1
            graphemes.append(cluster)
        else:
            graphemes.append(ch)
            i += 1
    return graphemes


def main():
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument('--num_samples', type=int, default=10)
    parser.add_argument('--use_graphemes', action='store_true')
    parser.add_argument('--model_id', default='microsoft/Florence-2-base')
    args = parser.parse_args()
    
    print("=" * 70)
    print("  Florence-2 Pipeline Validation (CPU)")
    print("=" * 70)
    
    # Load test data
    splits_path = os.path.join(BASE_DIR, 'florence2_splits.json')
    with open(splits_path, 'r', encoding='utf-8') as f:
        splits = json.load(f)
    test_pairs = splits['test'][:args.num_samples]
    print(f"  Testing on {len(test_pairs)} samples")
    
    # Load model
    print(f"\n[1/4] Loading Florence-2 ({args.model_id})...")
    import torch
    from PIL import Image
    from transformers import AutoModelForCausalLM, AutoProcessor
    
    device = torch.device('cpu')
    processor = AutoProcessor.from_pretrained(args.model_id, trust_remote_code=True)
    model = AutoModelForCausalLM.from_pretrained(
        args.model_id, trust_remote_code=True,
        dtype=torch.float32,
        attn_implementation="eager",
    )
    
    original_vocab_size = len(processor.tokenizer)
    print(f"  Model loaded. Vocab size: {original_vocab_size}")
    print(f"  Params: {sum(p.numel() for p in model.parameters()):,}")
    
    # Grapheme injection
    num_injected = 0
    if args.use_graphemes:
        print(f"\n[2/4] Injecting grapheme tokens...")
        all_gts = [p['gt'] for p in splits['train'] + splits['val'] + splits['test']]
        all_graphemes = Counter()
        for gt in all_gts:
            all_graphemes.update(segment_graphemes(gt))
        
        new_tokens = []
        for g in all_graphemes.keys():
            encoded = processor.tokenizer.encode(g, add_special_tokens=False)
            decoded = processor.tokenizer.decode(encoded).strip()
            if len(encoded) > 1 or decoded != g:
                new_tokens.append(g)
        
        num_injected = processor.tokenizer.add_tokens(new_tokens)
        model.resize_token_embeddings(len(processor.tokenizer))
        print(f"  ✓ Injected {num_injected} grapheme tokens")
        print(f"  ✓ New vocab: {len(processor.tokenizer)} (was {original_vocab_size})")
        
        # Show token compression
        sample_gt = test_pairs[0]['gt']
        tokens_before = processor.tokenizer.encode(sample_gt, add_special_tokens=False)
        print(f"  ✓ Sample '{sample_gt}' → {len(tokens_before)} tokens")
    else:
        print(f"\n[2/4] Using standard tokenizer")
    
    model.eval()
    
    # Run inference
    print(f"\n[3/4] Running inference on {len(test_pairs)} samples...")
    gts = []
    preds = []
    
    for i, pair in enumerate(test_pairs):
        gt = pair['gt']
        img_path = pair['image']
        
        # Fix path
        if not os.path.exists(img_path):
            img_path = img_path.replace('/', '\\')
        if not os.path.exists(img_path):
            basename = os.path.basename(img_path)
            img_path = os.path.join(BASE_DIR, 'Bengali', basename)
        
        try:
            image = Image.open(img_path).convert('RGB')
        except:
            print(f"  [{i+1}] ✗ Image not found: {img_path}")
            gts.append(gt)
            preds.append('')
            continue
        
        t0 = time.time()
        inputs = processor(images=image, text="<OCR>", return_tensors="pt").to(device)
        with torch.no_grad():
            gen_ids = model.generate(**inputs, max_new_tokens=64, num_beams=1, use_cache=False)
        pred = processor.batch_decode(gen_ids, skip_special_tokens=True)[0].strip()
        elapsed = time.time() - t0
        
        gts.append(gt)
        preds.append(pred)
        
        gt_n = normalize_text(gt)
        pred_n = normalize_text(pred)
        match = "✓" if gt_n == pred_n else "✗"
        cer = edit_distance(gt_n, pred_n) / max(len(gt_n), 1)
        
        print(f"  [{i+1}/{len(test_pairs)}] {match} GT:[{gt}] → Pred:[{pred}]  "
              f"CER={cer:.2f} ({elapsed:.1f}s)")
    
    # Compute overall metrics
    print(f"\n[4/4] Results")
    print("=" * 70)
    
    n = len(gts)
    total_ed, total_chars, exact = 0, 0, 0
    total_ned = 0.0
    for gt, pred in zip(gts, preds):
        gt_n, pred_n = normalize_text(gt), normalize_text(pred)
        ed = edit_distance(gt_n, pred_n)
        total_ed += ed
        total_chars += max(len(gt_n), 1)
        max_len = max(len(gt_n), len(pred_n))
        total_ned += (ed / max_len) if max_len > 0 else 0
        if gt_n == pred_n:
            exact += 1
    
    cer = total_ed / max(total_chars, 1)
    ned = total_ned / max(n, 1)
    
    mode = "GRAPHEME" if args.use_graphemes else "STANDARD"
    print(f"  Mode:         {mode}")
    print(f"  Samples:      {n}")
    print(f"  WRR:          {exact/max(n,1)*100:.2f}%")
    print(f"  CER:          {cer*100:.2f}%")
    print(f"  1-NED:        {(1-ned)*100:.2f}%")
    print(f"  Char Acc:     {(1-cer)*100:.2f}%")
    print(f"  Exact Match:  {exact}/{n}")
    
    if args.use_graphemes:
        print(f"  Injected:     {num_injected} grapheme tokens")
    
    # Save results
    result = {
        'mode': mode, 'num_samples': n, 'WRR': exact/max(n,1)*100,
        'CER': cer*100, '1-NED': (1-ned)*100, 'char_accuracy': (1-cer)*100,
        'grapheme_tokens_injected': num_injected,
        'predictions': [{'gt': g, 'pred': p} for g, p in zip(gts, preds)],
    }
    
    suffix = 'grapheme' if args.use_graphemes else 'standard'
    out_path = os.path.join(BASE_DIR, f'validation_florence2_{suffix}.json')
    with open(out_path, 'w', encoding='utf-8') as f:
        json.dump(result, f, ensure_ascii=False, indent=2)
    print(f"\n  Saved: {out_path}")
    
    print(f"\n{'='*70}")
    print(f"  NOTE: This is ZERO-SHOT (no fine-tuning).")
    print(f"  Fine-tuning on Colab will dramatically improve these numbers.")
    print(f"  CRNN baseline after training: 61.09% WRR")
    print(f"  PARSeq SOTA on Bengali: ~57-82% WRR (IndicSTR12)")
    print(f"  Florence-2 + graphemes after fine-tuning: expected 70-85% WRR")
    print(f"{'='*70}")


if __name__ == '__main__':
    main()
