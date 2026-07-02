#!/usr/bin/env python
"""Visual-similarity descriptor for the H3 horse-race (PREREGISTRATION.md §8,
Amendment 2, filed 2026-07-02).

For each of the 9 LOSO scripts, render its top-K most frequent grapheme
clusters (same segmentation as measure_script_descriptors.py, same
fontconfig-verified fonts as prepare_zeroshot_loso.py), embed the glyph
images with the FROZEN Florence-2 vision encoder (no training), and compute

    visual_similarity(S) = mean over the 8 other scripts T of the mean
                           pairwise cosine similarity between S's and T's
                           glyph embeddings.

Result-blind by construction: uses only BSTD train labels + fonts; no WRR.
Runs on CPU on purpose — must never compete with the LOSO GPU queue.

Output: visual_similarity_descriptors.json  (+ table on stdout)
Usage:  python compute_visual_similarity.py [--topk 50] [--max_fonts 3]
"""
import os, sys, json, argparse
from collections import Counter

BASE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, BASE)
os.environ.setdefault('HF_HOME', '/c/ujjwalb/.cache/huggingface')
os.environ['HF_HUB_OFFLINE'] = '1'
os.environ['TRANSFORMERS_OFFLINE'] = '1'
os.environ.setdefault('CUDA_VISIBLE_DEVICES', '')   # CPU only — GPU belongs to LOSO

import numpy as np
import torch
from PIL import Image, ImageDraw, ImageFont

from prepare_zeroshot_loso import (SCRIPT2LANGS, SCRIPT2LANGCODE, load_records,
                                   resolve_fonts)
from build_grapheme_vocab_lang import segment_graphemes_indic

MODEL_ID = 'microsoft/Florence-2-base'
SEED = 42


def top_clusters(script, k):
    """Top-k native grapheme clusters by frequency over the script's languages'
    BSTD b1800 TRAIN labels (identical segmentation to the descriptor pipeline)."""
    freq = Counter()
    for lang in SCRIPT2LANGS[script]:
        for r in load_records(lang, 'train'):
            g = (r.get('gt') or '').strip()
            if g:
                freq.update(segment_graphemes_indic(g))
    # keep clusters with at least one Indic codepoint (drop digits/Latin/punct)
    indic = [(c, n) for c, n in freq.most_common()
             if any('ऀ' <= ch <= 'ൿ' for ch in c)]
    return [c for c, _ in indic[:k]]


def render_cluster(text, font_path, size=48, pad=12):
    """Deterministic glyph render: black on white, no augmentation.
    Returns a PIL image or None if blank/tofu."""
    try:
        font = ImageFont.truetype(font_path, size)
    except Exception:
        return None
    d0 = ImageDraw.Draw(Image.new('RGB', (8, 8)))
    bb = d0.textbbox((0, 0), text, font=font)
    w, h = bb[2] - bb[0], bb[3] - bb[1]
    if w <= 0 or h <= 0:
        return None
    img = Image.new('RGB', (w + 2 * pad, h + 2 * pad), 'white')
    ImageDraw.Draw(img).text((pad - bb[0], pad - bb[1]), text, font=font, fill='black')
    arr = np.asarray(img)
    nz = int((arr != 255).any(axis=2).sum())
    if nz < 0.01 * img.width * img.height:       # blank / tofu
        return None
    return img


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--topk', type=int, default=50)
    ap.add_argument('--max_fonts', type=int, default=3)
    ap.add_argument('--batch', type=int, default=8)
    ap.add_argument('--out', default='visual_similarity_descriptors.json')
    args = ap.parse_args()

    scripts = sorted(SCRIPT2LANGCODE)            # 9 Brahmic scripts
    print(f'[vissim] scripts: {scripts}')

    # ── 1) render glyph images per script ────────────────────────────────────
    imgs, meta = {}, {}
    for s in scripts:
        clusters = top_clusters(s, args.topk)
        assert clusters, f'no clusters for {s}'
        fonts = resolve_fonts(s, sample=''.join(clusters[:3]))[:args.max_fonts]
        rendered = []
        for c in clusters:
            for f in fonts:
                im = render_cluster(c, f)
                if im is not None:
                    rendered.append(im)
        assert rendered, f'no non-blank renders for {s}'
        imgs[s] = rendered
        meta[s] = {'n_clusters': len(clusters), 'fonts': fonts,
                   'n_glyph_images': len(rendered)}
        print(f'[vissim] {s:12s} clusters={len(clusters)} fonts={len(fonts)} '
              f'images={len(rendered)}')

    # ── 2) embed with the FROZEN Florence-2 vision encoder (CPU) ─────────────
    from transformers import AutoModelForCausalLM, AutoProcessor
    torch.manual_seed(SEED)
    proc = AutoProcessor.from_pretrained(MODEL_ID, trust_remote_code=True)
    model = AutoModelForCausalLM.from_pretrained(
        MODEL_ID, trust_remote_code=True, torch_dtype=torch.float32)
    model.eval()
    for p in model.parameters():
        p.requires_grad_(False)
    img_proc = getattr(proc, 'image_processor', proc)

    def embed(pil_list):
        vecs = []
        for i in range(0, len(pil_list), args.batch):
            px = img_proc(images=pil_list[i:i + args.batch],
                          return_tensors='pt')['pixel_values']
            with torch.no_grad():
                if hasattr(model, '_encode_image'):
                    feats = model._encode_image(px)      # (B, T, D) proj. visual tokens
                else:
                    feats = model.vision_tower(px)
                if isinstance(feats, (tuple, list)):
                    feats = feats[0]
            v = feats.mean(dim=1)                        # mean-pool visual tokens
            v = torch.nn.functional.normalize(v, dim=-1)
            vecs.append(v)
        return torch.cat(vecs).numpy()

    embs = {}
    for s in scripts:
        embs[s] = embed(imgs[s])
        print(f'[vissim] embedded {s}: {embs[s].shape}')

    # ── 3) mean pairwise cosine between scripts ──────────────────────────────
    # vectors are L2-normalized, so mean pairwise cos(S,T) = mean(S) . mean(T)
    centroid = {s: embs[s].mean(axis=0) for s in scripts}
    M = {s: {t: float(np.dot(centroid[s], centroid[t])) for t in scripts}
         for s in scripts}
    vissim = {s: float(np.mean([M[s][t] for t in scripts if t != s]))
              for s in scripts}

    out = {'metadata': {'model': MODEL_ID, 'pooling': 'mean visual tokens, L2-norm',
                        'topk_clusters': args.topk, 'max_fonts': args.max_fonts,
                        'definition': 'mean over 8 other scripts of mean pairwise '
                                      'cosine between glyph embeddings',
                        'preregistration': 'PREREGISTRATION.md §8 Amendment 2 '
                                           '(2026-07-02); result-blind',
                        'per_script': meta},
           'pairwise_mean_cosine': M,
           'visual_similarity': vissim}
    json.dump(out, open(os.path.join(BASE, args.out), 'w'),
              ensure_ascii=False, indent=2)

    print('\n=== visual_similarity (higher = more visually similar to sources) ===')
    for s in sorted(vissim, key=vissim.get, reverse=True):
        print(f'  {s:12s} {vissim[s]:.4f}   (LOSO tag: {s.lower()})')
    print(f'\n[vissim] wrote {args.out}')


if __name__ == '__main__':
    main()
