"""
Cross-tokenizer agreement rescoring (GPU phase).
=================================================
Idea: each branch (BPE / grapheme / ...) generates its top-k beam candidates;
every candidate string is then scored by EVERY branch via teacher-forced
length-normalized log-likelihood. A candidate that all tokenization views
agree on (high likelihood under each) wins — a product-of-experts across
tokenizations. Word-level max-conf fusion is the special case that only
considers each branch's top-1 and ignores cross-scores.

This script only HARVESTS raw scores; combination rules are explored offline
by cross_rescore_eval.py (CPU). Models are loaded one at a time to keep GPU
memory low.

Output: cross_scores_<out_tag>.json
  [{gt, candidates: [str...],
    gen_conf: {tag: {cand: beam_prob}},          # only for cands from that beam
    xs: {tag: {cand: avg_logprob_per_token}}},  ...]
"""
import os, sys, json, argparse, gc
import torch
from PIL import Image
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from train_florence2 import BASE_DIR, TASK_PROMPT, resolve_image_path, set_seed
from predict_with_conf import load_model

K = 5  # beams kept per model


@torch.no_grad()
def harvest_candidates(model, processor, test, device, tag):
    """Generate top-K beam candidates + beam probs for each test image."""
    res = []
    for i, p in enumerate(test):
        ip = resolve_image_path(p['image'])
        if not ip:
            res.append({}); continue
        img = Image.open(ip).convert('RGB')
        inp = processor(images=img, text=TASK_PROMPT, return_tensors="pt").to(device)
        g = model.generate(input_ids=inp["input_ids"], pixel_values=inp["pixel_values"],
                           max_new_tokens=64, num_beams=K, num_return_sequences=K,
                           return_dict_in_generate=True, output_scores=True,
                           early_stopping=True)
        cands = {}
        for j in range(g.sequences.shape[0]):
            s = processor.batch_decode(g.sequences[j:j + 1], skip_special_tokens=True)[0].strip()
            pr = float(torch.exp(g.sequences_scores[j]).item())
            if s and (s not in cands or pr > cands[s]):
                cands[s] = round(pr, 5)
        res.append(cands)
        if (i + 1) % 50 == 0:
            print(f"  [{tag}] gen {i+1}/{len(test)}", flush=True)
    return res


@torch.no_grad()
def cross_score(model, processor, test, device, all_cands, tag):
    """Teacher-forced length-normalized log P(candidate | image) per candidate."""
    res = []
    for i, p in enumerate(test):
        ip = resolve_image_path(p['image'])
        cands = sorted(all_cands[i])
        if not ip or not cands:
            res.append({}); continue
        img = Image.open(ip).convert('RGB')
        inp = processor(images=[img] * len(cands), text=[TASK_PROMPT] * len(cands),
                        return_tensors="pt", padding=True).to(device)
        lab = processor.tokenizer(cands, return_tensors="pt", padding=True,
                                  truncation=True, max_length=128)
        ids = lab['input_ids']
        ids_masked = ids.clone()
        ids_masked[ids == processor.tokenizer.pad_token_id] = -100
        outp = model(input_ids=inp['input_ids'], pixel_values=inp['pixel_values'],
                     labels=ids_masked.to(device))
        # per-candidate avg logprob (the model returns mean loss over the whole
        # batch, so recompute per-sample CE from logits)
        logits = outp.logits  # [B, T, V]
        ll = {}
        lsm = torch.log_softmax(logits.float(), dim=-1)
        tgt = ids_masked.to(device)
        for b, c in enumerate(cands):
            mask = tgt[b] != -100
            n_tok = int(mask.sum())
            if n_tok == 0:
                ll[c] = -99.0; continue
            tok_ll = lsm[b, :tgt.shape[1]][mask, tgt[b][mask]]
            ll[c] = round(float(tok_ll.mean()), 5)
        res.append(ll)
        if (i + 1) % 50 == 0:
            print(f"  [{tag}] score {i+1}/{len(test)}", flush=True)
    return res


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--models', required=True,
                    help='semicolon list: tag,ckpt_dir,use_graphemes(0/1)[,vocab]')
    ap.add_argument('--splits_file', default=os.path.join(BASE_DIR, 'florence2_splits.json'))
    ap.add_argument('--out_tag', required=True)
    args = ap.parse_args()
    set_seed(42)
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    test = json.load(open(args.splits_file))['test']

    specs = []
    for m in args.models.split(';'):
        parts = m.split(',')
        specs.append({'tag': parts[0], 'ckpt': parts[1], 'grph': parts[2] == '1',
                      'vocab': parts[3] if len(parts) > 3 else None})

    # phase A: candidate generation (one model in memory at a time)
    gen = {}
    for s in specs:
        print(f"== generating with {s['tag']}", flush=True)
        model, processor = load_model(s['ckpt'], s['grph'], device, s['vocab'])
        gen[s['tag']] = harvest_candidates(model, processor, test, device, s['tag'])
        del model, processor; gc.collect(); torch.cuda.empty_cache()

    union = []
    for i in range(len(test)):
        u = set()
        for s in specs:
            u |= set(gen[s['tag']][i])
        union.append(sorted(u))

    # phase B: cross-scoring
    xs = {}
    for s in specs:
        print(f"== cross-scoring with {s['tag']}", flush=True)
        model, processor = load_model(s['ckpt'], s['grph'], device, s['vocab'])
        xs[s['tag']] = cross_score(model, processor, test, device, union, s['tag'])
        del model, processor; gc.collect(); torch.cuda.empty_cache()

    out = []
    for i, p in enumerate(test):
        out.append({'gt': p['gt'], 'candidates': union[i],
                    'gen_conf': {s['tag']: gen[s['tag']][i] for s in specs},
                    'xs': {s['tag']: xs[s['tag']][i] for s in specs}})
    path = os.path.join(BASE_DIR, f'cross_scores_{args.out_tag}.json')
    json.dump(out, open(path, 'w'), ensure_ascii=False, indent=1)
    print(f"saved {path}")


if __name__ == '__main__':
    main()
