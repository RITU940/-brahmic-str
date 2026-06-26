"""
Run a trained Florence-2 model on a split's TEST set, saving prediction + beam
confidence per sample -> conf_<tag>.json. Used for dual-tokenization fusion.
"""
import os, sys, json, argparse
import torch
from PIL import Image
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from train_florence2 import (
    BASE_DIR, DEFAULT_MODEL_ID, BANGLISH_TOKENIZER_ID, HF_TOKEN, TASK_PROMPT,
    GRAPHEME_VOCAB_FILE, inject_grapheme_tokens, resolve_image_path, set_seed,
)
from transformers import AutoProcessor, AutoModelForCausalLM, AutoTokenizer
from peft import PeftModel


def load_model(ckpt_dir, use_graphemes, device, grapheme_vocab=None):
    processor = AutoProcessor.from_pretrained(DEFAULT_MODEL_ID, trust_remote_code=True)
    processor.tokenizer = AutoTokenizer.from_pretrained(BANGLISH_TOKENIZER_ID, token=HF_TOKEN)
    model = AutoModelForCausalLM.from_pretrained(
        DEFAULT_MODEL_ID, trust_remote_code=True, torch_dtype=torch.float32,
        attn_implementation="eager")
    model.resize_token_embeddings(len(processor.tokenizer))
    if use_graphemes:
        inject_grapheme_tokens(model, processor, grapheme_vocab or GRAPHEME_VOCAB_FILE)
    model = PeftModel.from_pretrained(model, os.path.join(ckpt_dir, 'best_model'), is_trainable=False)
    return model.to(device).eval(), processor


@torch.no_grad()
def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--tag', required=True)
    ap.add_argument('--ckpt_dir', required=True)
    ap.add_argument('--splits_file', default=os.path.join(BASE_DIR, 'florence2_splits.json'))
    ap.add_argument('--use_graphemes', action='store_true')
    ap.add_argument('--grapheme_vocab', default=None)
    ap.add_argument('--num_beams', type=int, default=3)
    args = ap.parse_args()
    set_seed(42)
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    test = json.load(open(args.splits_file))['test']
    model, processor = load_model(args.ckpt_dir, args.use_graphemes, device, args.grapheme_vocab)
    out = []
    for i, p in enumerate(test):
        ip = resolve_image_path(p['image'])
        if not ip:
            out.append({'gt': p['gt'], 'pred': '', 'conf': 0.0}); continue
        img = Image.open(ip).convert('RGB')
        inp = processor(images=img, text=TASK_PROMPT, return_tensors="pt").to(device)
        g = model.generate(input_ids=inp["input_ids"], pixel_values=inp["pixel_values"],
                           max_new_tokens=64, num_beams=args.num_beams,
                           return_dict_in_generate=True, output_scores=True, early_stopping=True)
        pred = processor.batch_decode(g.sequences, skip_special_tokens=True)[0].strip()
        conf = float(torch.exp(g.sequences_scores[0]).item()) if g.sequences_scores is not None else 0.0
        out.append({'gt': p['gt'], 'pred': pred, 'conf': round(conf, 4)})
        if (i + 1) % 100 == 0:
            print(f"  {args.tag}: {i+1}/{len(test)}")
    json.dump(out, open(os.path.join(BASE_DIR, f'conf_{args.tag}.json'), 'w'), ensure_ascii=False, indent=2)
    print(f"saved conf_{args.tag}.json ({len(out)} samples)")


if __name__ == '__main__':
    main()
