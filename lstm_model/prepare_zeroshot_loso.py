"""
prepare_zeroshot_loso.py  --  Part ② FULL leave-one-SCRIPT-out data builder.
============================================================================
Generalises the (validated) Tamil pilot (prepare_zeroshot_pilot.py) to the
COMPLETE leave-one-script-out protocol over all 9 Brahmic scripts. For each
held-out SCRIPT S it builds the data to read S with ZERO real S images, training
only on the OTHER scripts (+ synthetic S in Rung B), all in the shared abugida
pivot space (build_shared_grapheme_space.py). This produces one zero-shot
transfer number per script -> the H3 unifying result (does fertility predict
which unseen scripts transfer best?).

WHAT THIS FIXES vs the pilot (so results are correct, not silently broken):
  1. FONT BUG. The pilot's resolve_font used family 'Noto Sans <Script>', which on
     this machine (no Noto Indic fonts) silently falls back to DejaVu -> tofu boxes.
     Here we resolve via `fc-match :lang=<code>` (guaranteed glyph coverage), use
     MULTIPLE fonts per script for visual diversity, and VERIFY every rendered image
     is non-blank (abort loudly if a script can't render).
  2. STALE PATHS. The b1800 split files store /home/ujjwal/ritu1/... paths that do
     not resolve after the move to server3. Every image path is rewritten to the
     local repo (anchored at 'benchmarks/...') and existence-checked.
  3. SCRIPT-level hold-out. When a script has >1 language (Bengali: bengali+assamese;
     Devanagari: hindi+marathi) ALL its languages are held out together and ALL
     same-script languages are excluded from the sources -> a genuinely UNSEEN script.
  4. BALANCED sources. Equal per-source-language budget (seed 42) so transfer is not
     dominated by whichever source script has the most data.

Eval is in pivot space; the mapping is verified round-trip-lossless
(build_shared_grapheme_space --verify) so pivot-space WRR == target-script WRR.

NO GPU. Usage:
  PY=/c/ujjwalb/.conda/envs/ritu_scenetext/bin/python
  $PY prepare_zeroshot_loso.py --script Tamil          # one held-out script
  $PY prepare_zeroshot_loso.py --all                   # all 9
"""
import os, sys, json, argparse, random, subprocess
from collections import Counter
import numpy as np
from PIL import Image, ImageDraw, ImageFont, ImageFilter

BASE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, BASE)
from build_shared_grapheme_space import (to_pivot, pivot_graphemes, LANG2SCRIPT,
                                         SCRIPT_BASES)

SEED = 42

# Brahmic languages only (Latin/english excluded from the zero-shot core).
BRAHMIC_LANGS = [l for l, s in LANG2SCRIPT.items() if s in SCRIPT_BASES]
# script -> its languages
SCRIPT2LANGS = {}
for l in BRAHMIC_LANGS:
    SCRIPT2LANGS.setdefault(LANG2SCRIPT[l], []).append(l)
# script -> a fontconfig language code that guarantees glyph coverage
SCRIPT2LANGCODE = {
    'Devanagari': 'hi', 'Bengali': 'bn', 'Gurmukhi': 'pa', 'Gujarati': 'gu',
    'Oriya': 'or', 'Tamil': 'ta', 'Telugu': 'te', 'Kannada': 'kn', 'Malayalam': 'ml',
}


# ── path handling ────────────────────────────────────────────────────────────
def localize(path):
    """Rewrite a stored image path to this repo. Splits store stale
    /home/ujjwal/ritu1/... prefixes; the images live under BASE/benchmarks/..."""
    if 'benchmarks/' in path:
        return os.path.join(BASE, path[path.index('benchmarks/'):])
    return path


def split_path(lang):
    return os.path.join(BASE, f'florence2_splits_bstd_{lang}_b1800.json')


def load_records(lang, part):
    d = json.load(open(split_path(lang), encoding='utf-8'))
    return d.get(part, [])


def pivotize(records, require_exist=True):
    """Keep the real image (path localized), replace gt with pivot text;
    stash original gt for vocab/debug. Drop records whose image is missing."""
    out, missing = [], 0
    for r in records:
        g = r['gt'].strip()
        if not g:
            continue
        img = localize(r['image'])
        if require_exist and not os.path.exists(img):
            missing += 1
            continue
        out.append({'image': img, 'gt': to_pivot(g), 'name': r.get('name', ''),
                    'orig_gt': g})
    return out, missing


def build_vocab(records, tag, out_path):
    freq = Counter()
    for r in records:
        freq.update(pivot_graphemes(r['orig_gt']))
    graphemes = sorted(g for g in freq if g.strip())
    g2i = {'<blank>': 0, '<unk>': 1}
    for g in graphemes:
        g2i[g] = len(g2i)
    json.dump({'grapheme2idx': g2i,
               'metadata': {'tag': tag, 'num_graphemes': len(graphemes),
                            'space': 'shared-abugida-pivot'}},
              open(out_path, 'w'), ensure_ascii=False, indent=2)
    return set(graphemes)


# ── fonts (correct, verified) ────────────────────────────────────────────────
def _renders_nonblank(font_path, sample):
    try:
        font = ImageFont.truetype(font_path, 48)
        img = Image.new('RGB', (480, 90), 'white')
        d = ImageDraw.Draw(img)
        d.text((6, 6), sample, font=font, fill='black')
        # fraction of non-white pixels; tofu/blank -> near 0
        arr = np.asarray(img)
        nz = int((arr != 255).any(axis=2).sum())
        return nz / (480 * 90) > 0.003
    except Exception:
        return False


def resolve_fonts(script, sample, max_fonts=6):
    """Return a list of font files that ACTUALLY render `sample` (a target word)
    with real glyphs, via fontconfig language matching. Aborts if none qualify."""
    code = SCRIPT2LANGCODE[script]
    try:
        files = subprocess.check_output(
            ['fc-list', f':lang={code}', '--format=%{file}\n']).decode().splitlines()
    except Exception:
        files = []
    files = [f.strip() for f in files if f.strip()]
    good = []
    seen = set()
    for f in files:
        if f in seen or not os.path.exists(f):
            continue
        seen.add(f)
        if _renders_nonblank(f, sample):
            good.append(f)
        if len(good) >= max_fonts:
            break
    assert good, f"NO font renders {script} ({code}) on this machine — aborting."
    return good


def render_word(text, fonts, out_path, rng):
    """Render `text` with a randomly chosen verified font. Returns True if the
    output image is non-blank (real glyphs), else False (caller skips)."""
    font_path = rng.choice(fonts)
    fsize = rng.randint(34, 58)
    font = ImageFont.truetype(font_path, fsize)
    d0 = ImageDraw.Draw(Image.new('RGB', (8, 8)))
    bb = d0.textbbox((0, 0), text, font=font)
    w, h = bb[2] - bb[0], bb[3] - bb[1]
    if w <= 0 or h <= 0:
        return False
    pad = rng.randint(8, 22)
    W, H = max(w + 2 * pad, 16), max(h + 2 * pad, 16)
    bg = tuple(rng.randint(170, 255) for _ in range(3))
    img = Image.new('RGB', (W, H), bg)
    d = ImageDraw.Draw(img)
    fg = tuple(rng.randint(0, 95) for _ in range(3))
    d.text((pad - bb[0], pad - bb[1]), text, font=font, fill=fg)
    # non-blank check on the text region (vs background)
    arr = np.asarray(img)
    nz = int((arr != np.array(bg)).any(axis=2).sum())
    if nz < 0.01 * W * H:
        return False
    if rng.random() < 0.30:
        img = img.filter(ImageFilter.GaussianBlur(rng.uniform(0.4, 0.9)))
    img.save(out_path, quality=92)
    return True


# ── one held-out script ──────────────────────────────────────────────────────
def build_for_script(script, per_source_train=700, per_source_val=90,
                     synth_per_word=2):
    assert script in SCRIPT2LANGS, f"{script} not a Brahmic script with data"
    rng = random.Random(SEED)
    targets = SCRIPT2LANGS[script]
    sources = [l for l in BRAHMIC_LANGS if LANG2SCRIPT[l] != script]
    tag = script.lower()
    print(f"\n===== held-out SCRIPT={script}  targets={targets}  "
          f"sources={sources} =====")

    # ── balanced source (training) data in pivot space ───────────────────────
    src_train, src_val, miss_total = [], [], 0
    for s in sources:
        tr, m1 = pivotize(load_records(s, 'train'));  miss_total += m1
        va, m2 = pivotize(load_records(s, 'val'));    miss_total += m2
        rng.shuffle(tr); rng.shuffle(va)
        src_train += tr[:per_source_train]
        src_val += va[:per_source_val]
    rng.shuffle(src_train)
    if miss_total:
        print(f"  [warn] {miss_total} source records had missing images (skipped)")

    # ── held-out target test (real images, never trained on) ─────────────────
    tgt_test, miss_t = [], 0
    tgt_train_words = []
    for t in targets:
        te, m = pivotize(load_records(t, 'test'));  miss_t += m
        tgt_test += te
        tgt_train_words += [r['gt'].strip() for r in load_records(t, 'train')
                            if r['gt'].strip()]
    assert tgt_test, f"no target test images for {script} — check paths!"
    if miss_t:
        print(f"  [warn] {miss_t} target TEST records missing images (skipped)")

    # ── Rung A: sources only ─────────────────────────────────────────────────
    json.dump({'train': src_train, 'val': src_val, 'test': tgt_test,
               'metadata': {'rung': 'A', 'script': script, 'targets': targets,
                            'sources': sources}},
              open(os.path.join(BASE, f'splits_zeroshot_loso_rungA_{tag}.json'), 'w'),
              ensure_ascii=False)
    src_vocab = build_vocab(src_train, f'zs_loso_rungA_{tag}',
                            os.path.join(BASE, f'grapheme_vocab_zeroshot_loso_rungA_{tag}.json'))

    # ── Rung B: + synthetic target words (verified-correct rendering) ─────────
    sample = next(w for w in tgt_train_words if w)
    fonts = resolve_fonts(script, sample)
    print(f"  fonts ({len(fonts)}): " + ", ".join(os.path.basename(f) for f in fonts))
    synth_dir = os.path.join(BASE, f'synth_zeroshot_loso_{tag}')
    os.makedirs(synth_dir, exist_ok=True)
    synth, k, fail = [], 0, 0
    for w in tgt_train_words:
        for _ in range(synth_per_word):
            fp = os.path.join(synth_dir, f'synth_{tag}_{k:06d}.jpg')
            if render_word(w, fonts, fp, rng):
                synth.append({'image': fp, 'gt': to_pivot(w),
                              'name': f'synth{k}', 'orig_gt': w})
                k += 1
            else:
                fail += 1
    blank_rate = 100 * fail / max(k + fail, 1)
    assert blank_rate < 20, (f"{blank_rate:.1f}% of {script} synth renders blank — "
                             "font/shaping problem, aborting.")
    print(f"  synth rendered: {len(synth)} (blank/failed {fail}, {blank_rate:.1f}%)")
    rungB_train = src_train + synth
    rng.shuffle(rungB_train)
    json.dump({'train': rungB_train, 'val': src_val, 'test': tgt_test,
               'metadata': {'rung': 'B', 'script': script, 'targets': targets,
                            'sources': sources, 'n_synth': len(synth)}},
              open(os.path.join(BASE, f'splits_zeroshot_loso_rungB_{tag}.json'), 'w'),
              ensure_ascii=False)
    build_vocab(rungB_train, f'zs_loso_rungB_{tag}',
                os.path.join(BASE, f'grapheme_vocab_zeroshot_loso_rungB_{tag}.json'))

    # ── pre-GPU coverage signal ──────────────────────────────────────────────
    tgt_tokens = Counter()
    for r in tgt_test:
        tgt_tokens.update(pivot_graphemes(r['orig_gt']))
    tot = sum(tgt_tokens.values())
    cov_tok = 100 * sum(c for g, c in tgt_tokens.items() if g in src_vocab) / max(tot, 1)
    cov_typ = 100 * sum(1 for g in tgt_tokens if g in src_vocab) / max(len(tgt_tokens), 1)

    meta = {'script': script, 'targets': targets, 'sources': sources, 'seed': SEED,
            'per_source_train': per_source_train, 'per_source_val': per_source_val,
            'n_src_train': len(src_train), 'n_src_val': len(src_val),
            'n_target_test': len(tgt_test), 'n_synth': len(synth),
            'src_vocab_size': len(src_vocab), 'fonts': [os.path.basename(f) for f in fonts],
            'target_token_coverage_by_source_vocab_pct': round(cov_tok, 2),
            'target_type_coverage_by_source_vocab_pct': round(cov_typ, 2)}
    json.dump(meta, open(os.path.join(BASE, f'zeroshot_loso_meta_{tag}.json'), 'w'),
              ensure_ascii=False, indent=2)
    print(f"  PRE-GPU SIGNAL: source vocab covers {cov_tok:.1f}% of {script} test "
          f"token occurrences ({cov_typ:.1f}% distinct) with ZERO {script} training.")
    return meta


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--script', help='one Brahmic script, e.g. Tamil')
    ap.add_argument('--all', action='store_true', help='build all 9 scripts')
    ap.add_argument('--per_source_train', type=int, default=700)
    ap.add_argument('--per_source_val', type=int, default=90)
    ap.add_argument('--synth_per_word', type=int, default=2)
    a = ap.parse_args()
    scripts = sorted(SCRIPT2LANGS) if a.all else [a.script]
    assert scripts and scripts[0], "give --script <Script> or --all"
    summary = []
    for s in scripts:
        summary.append(build_for_script(s, a.per_source_train, a.per_source_val,
                                        a.synth_per_word))
    print("\n===== COVERAGE SUMMARY (held-out script : token-coverage%) =====")
    for m in summary:
        print(f"  {m['script']:<12} tok={m['target_token_coverage_by_source_vocab_pct']:>6}"
              f"  typ={m['target_type_coverage_by_source_vocab_pct']:>6}"
              f"  test={m['n_target_test']}  synth={m['n_synth']}")


if __name__ == '__main__':
    main()
