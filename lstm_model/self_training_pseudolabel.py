"""
Semi-supervised self-training: pseudo-label the unlabeled Bengali crops.
=========================================================================
NOVELTY (contribution #2): use the best grapheme Florence-2 model to
pseudo-label the 3,120 unlabeled real scene-text crops, keep only
high-confidence predictions (beam sequence-score), and emit an augmented
training split for a second-round fine-tune.

Confidence = exp(sequences_scores) from beam search (length-normalised
geometric-mean token probability), which is well-calibrated for filtering.

Outputs:
  - pseudo_labels.json            (all predictions + confidence, for analysis)
  - florence2_splits_selftrain.json   (real train + accepted pseudo-labels)

Usage:
  python self_training_pseudolabel.py --conf 0.6
  python train_florence2.py --use_graphemes --splits_file florence2_splits_selftrain.json
"""
import os, sys, json, argparse
import torch
from PIL import Image

sys.stdout.reconfigure(encoding='utf-8', errors='replace')
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from train_florence2 import (
    BASE_DIR, DEFAULT_MODEL_ID, BANGLISH_TOKENIZER_ID, HF_TOKEN, TASK_PROMPT,
    GRAPHEME_VOCAB_FILE, inject_grapheme_tokens, resolve_image_path, set_seed,
)
from transformers import AutoProcessor, AutoModelForCausalLM, AutoConfig, AutoTokenizer
from peft import PeftModel

import unicodedata


def is_plausible_bengali(text: str) -> bool:
    """Reject empty / pure-punctuation / non-Bengali pseudo-labels."""
    t = text.strip()
    if not t or t == '###':
        return False
    has_bn = any('ঀ' <= c <= '৿' for c in t)
    if not has_bn:
        return False
    if len(t) > 40:           # runaway generation
        return False
    return True


def load_grapheme_model(ckpt_dir, device):
    print(f"Loading base + Banglish tokenizer + grapheme injection + adapter...")
    processor = AutoProcessor.from_pretrained(DEFAULT_MODEL_ID, trust_remote_code=True)
    banglish = AutoTokenizer.from_pretrained(BANGLISH_TOKENIZER_ID, token=HF_TOKEN)
    processor.tokenizer = banglish
    config = AutoConfig.from_pretrained(DEFAULT_MODEL_ID, trust_remote_code=True)
    model = AutoModelForCausalLM.from_pretrained(
        DEFAULT_MODEL_ID, trust_remote_code=True,
        torch_dtype=torch.float32, attn_implementation="eager",
    )
    model.resize_token_embeddings(len(processor.tokenizer))
    inject_grapheme_tokens(model, processor, GRAPHEME_VOCAB_FILE)
    best_path = os.path.join(ckpt_dir, 'best_model')
    assert os.path.exists(best_path), f"No best_model at {best_path} — train grapheme model first."
    model = PeftModel.from_pretrained(model, best_path, is_trainable=False)
    model.to(device).eval()
    return model, processor


@torch.no_grad()
def predict(model, processor, image, device, num_beams=3, max_new_tokens=64):
    inputs = processor(images=image, text=TASK_PROMPT, return_tensors="pt").to(device)
    out = model.generate(
        input_ids=inputs["input_ids"], pixel_values=inputs["pixel_values"],
        max_new_tokens=max_new_tokens, num_beams=num_beams,
        return_dict_in_generate=True, output_scores=True, early_stopping=True,
    )
    seq = out.sequences
    text = processor.batch_decode(seq, skip_special_tokens=True)[0].strip()
    # beam sequence score -> geometric-mean token probability
    conf = float(torch.exp(out.sequences_scores[0]).item()) if out.sequences_scores is not None else 0.0
    return text, conf


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--conf', type=float, default=0.6, help='min confidence to accept a pseudo-label')
    ap.add_argument('--pool', default=os.path.join(BASE_DIR, 'unlabeled_pool.json'))
    ap.add_argument('--ckpt_dir', default=os.path.join(BASE_DIR, 'checkpoints_florence2_grapheme'))
    ap.add_argument('--splits_file', default=os.path.join(BASE_DIR, 'florence2_splits.json'))
    ap.add_argument('--num_beams', type=int, default=3)
    ap.add_argument('--limit', type=int, default=0, help='debug: only N crops')
    args = ap.parse_args()

    set_seed(42)
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    pool = json.load(open(args.pool))['unlabeled']
    if args.limit:
        pool = pool[:args.limit]
    print(f"Unlabeled crops to pseudo-label: {len(pool)}")

    model, processor = load_grapheme_model(args.ckpt_dir, device)

    results, accepted = [], []
    for i, item in enumerate(pool):
        img_path = resolve_image_path(item['image'])
        if not img_path:
            continue
        try:
            image = Image.open(img_path).convert('RGB')
        except Exception:
            continue
        text, conf = predict(model, processor, image, device, args.num_beams)
        ok = conf >= args.conf and is_plausible_bengali(text)
        rec = {'name': item['name'], 'image': img_path, 'pred': text, 'conf': round(conf, 4), 'accepted': ok}
        results.append(rec)
        if ok:
            accepted.append({'image': img_path, 'gt': text, 'name': item['name'], 'pseudo': True})
        if (i + 1) % 100 == 0:
            print(f"  {i+1}/{len(pool)}  accepted={len(accepted)}  (last conf={conf:.3f}: '{text}')")

    json.dump({'threshold': args.conf, 'total': len(results), 'accepted': len(accepted),
               'results': results}, open(os.path.join(BASE_DIR, 'pseudo_labels.json'), 'w'),
              ensure_ascii=False, indent=2)

    splits = json.load(open(args.splits_file))
    merged = dict(splits)
    merged['train'] = splits['train'] + accepted
    merged['metadata'] = dict(splits.get('metadata', {}))
    merged['metadata']['pseudo_added'] = len(accepted)
    merged['metadata']['pseudo_conf_threshold'] = args.conf
    json.dump(merged, open(os.path.join(BASE_DIR, 'florence2_splits_selftrain.json'), 'w'),
              ensure_ascii=False, indent=2)

    print(f"\n=== Self-training pseudo-labelling done ===")
    print(f"  Pseudo-labelled: {len(results)} | accepted (conf>={args.conf}): {len(accepted)} "
          f"({100*len(accepted)/max(len(results),1):.1f}%)")
    print(f"  Real train: {len(splits['train'])} -> augmented train: {len(merged['train'])}")
    print(f"  Saved pseudo_labels.json + florence2_splits_selftrain.json")


if __name__ == '__main__':
    main()
