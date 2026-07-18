"""
eval_vlm_baseline.py — PREREGISTRATION Amendment 4c: frontier-VLM reference.
===========================================================================
Qwen2.5-VL-7B-Instruct prompted zero-shot on the NINE BSTD test sets (the same
real images every LOSO rung was evaluated on), with the declared protocol:
fixed simple prompt ("Read the text in the image."), greedy decoding,
and the SAME normalization/WRR metric (metrics.evaluate_corpus) as all our
results. Ground truth is the NATIVE-script text (orig_gt) — the VLM reads the
actual script, not our pivot space.

Scoring, declared: PRIMARY = exact-match WRR under the shared normalization
(the preregistered protocol). SECONDARY (disclosed as post-hoc, reported
alongside, biased IN THE VLM'S FAVOR so it cannot be called a strawman):
lenient WRR = credit if the normalized gt appears as a whitespace-token or
substring of the normalized prediction (chat models wrap answers in prose).

Per tag: conf_vlm_qwen25_<tag>.json (name/gt/pred) +
         result_vlm_qwen25_<tag>.json (N, WRR, WRR_lenient, CharAcc, CER).
Resumable per tag (skips tags whose result JSON exists).

Runs in the .tools/vlm_venv (transformers 4.51.3); needs the FULL GPU (~17 GB
bf16) — only between training runs. Model must be pre-downloaded (HF cache).
"""
import os, sys, json, unicodedata

BASE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, BASE)
from metrics import evaluate_corpus, normalize_bengali

import torch
from transformers import Qwen2_5_VLForConditionalGeneration, AutoProcessor
from qwen_vl_utils import process_vision_info

MODEL_ID = 'Qwen/Qwen2.5-VL-7B-Instruct'
PROMPT = 'Read the text in the image.'
TAGS = ['tamil', 'telugu', 'kannada', 'malayalam', 'oriya', 'gujarati',
        'bengali', 'devanagari', 'gurmukhi']


def lenient_hit(gt, pred):
    g, p = normalize_bengali(gt), normalize_bengali(pred)
    if not g:
        return False
    return g in p.split() or g in p


def main():
    print(f'loading {MODEL_ID} (bf16)...', flush=True)
    model = Qwen2_5_VLForConditionalGeneration.from_pretrained(
        MODEL_ID, torch_dtype=torch.bfloat16, device_map='cuda')
    processor = AutoProcessor.from_pretrained(MODEL_ID)
    model.eval()

    for tag in TAGS:
        res_path = os.path.join(BASE, f'result_vlm_qwen25_{tag}.json')
        if os.path.exists(res_path):
            print(f'[{tag}] result exists, skip', flush=True)
            continue
        test = json.load(open(os.path.join(
            BASE, f'splits_zeroshot_loso_rungA_{tag}.json')))['test']
        out = []
        for i, r in enumerate(test):
            msgs = [{'role': 'user', 'content': [
                {'type': 'image', 'image': r['image']},
                {'type': 'text', 'text': PROMPT}]}]
            text = processor.apply_chat_template(msgs, tokenize=False,
                                                 add_generation_prompt=True)
            imgs, vids = process_vision_info(msgs)
            inputs = processor(text=[text], images=imgs, videos=vids,
                               padding=True, return_tensors='pt').to('cuda')
            with torch.inference_mode():
                gen = model.generate(**inputs, max_new_tokens=64,
                                     do_sample=False, num_beams=1)
            gen = gen[:, inputs.input_ids.shape[1]:]
            pred = processor.batch_decode(gen, skip_special_tokens=True,
                                          clean_up_tokenization_spaces=False)[0].strip()
            out.append({'name': r.get('name', ''), 'gt': r['orig_gt'], 'pred': pred})
            if (i + 1) % 200 == 0:
                print(f'  [{tag}] {i+1}/{len(test)}', flush=True)
        json.dump(out, open(os.path.join(BASE, f'conf_vlm_qwen25_{tag}.json'), 'w'),
                  ensure_ascii=False, indent=1)
        gts = [r['gt'] for r in out]
        preds = [r['pred'] for r in out]
        m = evaluate_corpus(gts, preds)
        len_wrr = 100.0 * sum(lenient_hit(g, p) for g, p in zip(gts, preds)) / len(out)
        rec = {'model': MODEL_ID, 'script': tag, 'N': len(out),
               'WRR': round(m['WRR'], 2), 'WRR_lenient': round(len_wrr, 2),
               'CharAcc': round(m['char_accuracy'], 2), 'CER': round(m['CER'], 2),
               'prompt': PROMPT, 'decoding': 'greedy'}
        json.dump(rec, open(res_path, 'w'))
        print(f"[RESULT VLM qwen25 {tag}] N={rec['N']} WRR={rec['WRR']} "
              f"WRR_lenient={rec['WRR_lenient']} CharAcc={rec['CharAcc']}", flush=True)


if __name__ == '__main__':
    main()
