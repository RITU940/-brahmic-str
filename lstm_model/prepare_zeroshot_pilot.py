"""
prepare_zeroshot_pilot.py  --  Part ② feasibility pilot data builder.
=====================================================================
Builds the data for a leave-one-script-out zero-shot pilot in the SHARED ABUGIDA
SPACE (see build_shared_grapheme_space.py). Default: train on Telugu+Kannada+
Malayalam (Dravidian neighbours), read held-out TAMIL with ZERO real Tamil images.

Produces, for the held-out target:
  splits_zeroshot_rungA_<target>.json   train/val = SOURCE images (pivot text),
                                        test = real TARGET test images (pivot text).
  splits_zeroshot_rungB_<target>.json   = Rung A train + SYNTHETIC target words
                                        (font-rendered, NO real target images).
  grapheme_vocab_zeroshot_rungA_<target>.json   built from SOURCES ONLY (no target peek).
  grapheme_vocab_zeroshot_rungB_<target>.json   built from sources + synthetic target.
  synth_zeroshot_<target>/              rendered synthetic word images.
  zeroshot_pilot_meta_<target>.json     provenance + the pre-GPU coverage signal.

Eval is done IN PIVOT SPACE; because the mapping is verified round-trip-lossless
(build_shared_grapheme_space --verify), pivot-space WRR == target-script WRR.

NO GPU. Usage:
  PY=/home/ujjwal/miniconda3/envs/ritu_scenetext/bin/python
  $PY prepare_zeroshot_pilot.py --sources telugu kannada malayalam --target tamil
"""
import os, sys, json, argparse, random, subprocess
from collections import Counter
from PIL import Image, ImageDraw, ImageFont, ImageFilter

BASE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, BASE)
from build_shared_grapheme_space import to_pivot, pivot_graphemes, LANG2SCRIPT

SEED = 42


def split_path(lang):
    return os.path.join(BASE, f'florence2_splits_bstd_{lang}_b1800.json')


def load_records(lang, part):
    d = json.load(open(split_path(lang), encoding='utf-8'))
    return d.get(part, [])


def pivotize(records):
    """Keep the real image, replace gt with pivot text; stash original for vocab/debug."""
    out = []
    for r in records:
        g = r['gt'].strip()
        if not g:
            continue
        out.append({'image': r['image'], 'gt': to_pivot(g), 'name': r.get('name', ''),
                    'orig_gt': g})
    return out


def build_vocab(records, lang_tag, out_path):
    freq = Counter()
    for r in records:
        freq.update(pivot_graphemes(r['orig_gt']))   # segment in original, map to pivot
    graphemes = sorted(g for g in freq if g.strip())
    g2i = {'<blank>': 0, '<unk>': 1}
    for g in graphemes:
        g2i[g] = len(g2i)
    json.dump({'grapheme2idx': g2i,
               'metadata': {'tag': lang_tag, 'num_graphemes': len(graphemes),
                            'space': 'shared-abugida-pivot'}},
              open(out_path, 'w'), ensure_ascii=False, indent=2)
    return set(graphemes)


def resolve_font(script):
    p = f'/usr/share/fonts/truetype/noto/NotoSans{script}-Regular.ttf'
    if os.path.exists(p):
        return p
    try:
        return subprocess.check_output(['fc-match', '-f', '%{file}', f'Noto Sans {script}']).decode().strip()
    except Exception:
        return None


def render_word(text, font_path, out_path, rng):
    fsize = rng.randint(34, 58)
    font = ImageFont.truetype(font_path, fsize)
    d0 = ImageDraw.Draw(Image.new('RGB', (8, 8)))
    bb = d0.textbbox((0, 0), text, font=font)
    w, h = bb[2] - bb[0], bb[3] - bb[1]
    pad = rng.randint(8, 22)
    W, H = max(w + 2 * pad, 16), max(h + 2 * pad, 16)
    bg = tuple(rng.randint(170, 255) for _ in range(3))
    img = Image.new('RGB', (W, H), bg)
    d = ImageDraw.Draw(img)
    fg = tuple(rng.randint(0, 95) for _ in range(3))
    d.text((pad - bb[0], pad - bb[1]), text, font=font, fill=fg)
    if rng.random() < 0.30:
        img = img.filter(ImageFilter.GaussianBlur(rng.uniform(0.4, 0.9)))
    img.save(out_path, quality=92)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--sources', nargs='+', default=['telugu', 'kannada', 'malayalam'])
    ap.add_argument('--target', default='tamil')
    ap.add_argument('--synth_per_word', type=int, default=2)
    a = ap.parse_args()
    rng = random.Random(SEED)
    tgt = a.target

    # ── source (training) data in pivot space ────────────────────────────────
    src_train, src_val = [], []
    for s in a.sources:
        src_train += pivotize(load_records(s, 'train'))
        src_val += pivotize(load_records(s, 'val'))
    rng.shuffle(src_train)

    # ── held-out target test in pivot space (real images, never trained on) ───
    tgt_test = pivotize(load_records(tgt, 'test'))

    # ── Rung A: source-only ───────────────────────────────────────────────────
    json.dump({'train': src_train, 'val': src_val, 'test': tgt_test,
               'metadata': {'rung': 'A', 'sources': a.sources, 'target': tgt}},
              open(os.path.join(BASE, f'splits_zeroshot_rungA_{tgt}.json'), 'w'),
              ensure_ascii=False)
    src_vocab = build_vocab(src_train, f'zs_rungA_{tgt}',
                            os.path.join(BASE, f'grapheme_vocab_zeroshot_rungA_{tgt}.json'))

    # ── Rung B: + synthetic target words (font-rendered; NO real target images) ─
    synth_dir = os.path.join(BASE, f'synth_zeroshot_{tgt}')
    os.makedirs(synth_dir, exist_ok=True)
    font = resolve_font(LANG2SCRIPT[tgt])
    assert font, f"no Noto font for {LANG2SCRIPT[tgt]}"
    tgt_words = [r['gt'].strip() for r in load_records(tgt, 'train') if r['gt'].strip()]
    synth = []
    k = 0
    for w in tgt_words:
        for _ in range(a.synth_per_word):
            fp = os.path.join(synth_dir, f'synth_{tgt}_{k:06d}.jpg')
            try:
                render_word(w, font, fp, rng)
            except Exception:
                continue
            synth.append({'image': fp, 'gt': to_pivot(w), 'name': f'synth{k}', 'orig_gt': w})
            k += 1
    rungB_train = src_train + synth
    rng.shuffle(rungB_train)
    json.dump({'train': rungB_train, 'val': src_val, 'test': tgt_test,
               'metadata': {'rung': 'B', 'sources': a.sources, 'target': tgt,
                            'n_synth': len(synth)}},
              open(os.path.join(BASE, f'splits_zeroshot_rungB_{tgt}.json'), 'w'),
              ensure_ascii=False)
    build_vocab(rungB_train, f'zs_rungB_{tgt}',
                os.path.join(BASE, f'grapheme_vocab_zeroshot_rungB_{tgt}.json'))

    # ── PRE-GPU FEASIBILITY SIGNAL: does the source-only pivot vocab already ──
    #    cover the held-out target's grapheme units? (zero target training.) ────
    tgt_tokens = Counter()
    for r in tgt_test:
        tgt_tokens.update(pivot_graphemes(r['orig_gt']))
    tot = sum(tgt_tokens.values())
    covered = sum(c for g, c in tgt_tokens.items() if g in src_vocab)
    types_cov = sum(1 for g in tgt_tokens if g in src_vocab)
    cov_tok = 100 * covered / max(tot, 1)
    cov_typ = 100 * types_cov / max(len(tgt_tokens), 1)

    meta = {'sources': a.sources, 'target': tgt, 'seed': SEED,
            'n_src_train': len(src_train), 'n_src_val': len(src_val),
            'n_target_test': len(tgt_test), 'n_synth': len(synth),
            'src_vocab_size': len(src_vocab),
            'target_token_coverage_by_source_vocab_pct': round(cov_tok, 2),
            'target_type_coverage_by_source_vocab_pct': round(cov_typ, 2)}
    json.dump(meta, open(os.path.join(BASE, f'zeroshot_pilot_meta_{tgt}.json'), 'w'),
              ensure_ascii=False, indent=2)

    print(json.dumps(meta, indent=2, ensure_ascii=False))
    print(f"\n>>> PRE-GPU SIGNAL: source-only pivot vocab already covers "
          f"{cov_tok:.1f}% of held-out {tgt} grapheme-token occurrences "
          f"({cov_typ:.1f}% of distinct units) with ZERO {tgt} training.")
    print(">>> High coverage => structural zero-shot transfer is plausible; "
          "low => Rung A will be near-zero and Rung B (synthetic) is essential.")


if __name__ == '__main__':
    main()
