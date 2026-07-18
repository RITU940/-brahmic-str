"""
prepare_scaling_sweep.py — Amendment-4a/5 synthetic-exposure scaling sweep data builder.
=======================================================================================
Implements PREREGISTRATION Amendment 5 (2026-07-18) EXACTLY, for the three declared
scripts (malayalam, kannada, telugu) x budgets {810, 1620, 3240*, 6480, 12960}:

  810/1620  = prefix of ONE seed-42 shuffle of the existing 3240-image synth list
              (deterministic base order = sorted by image filename) -> nested subsets.
  6480      = existing 3240 + 3240 NEW renders: previously-unused BSTD train records
              FIRST (x2 renders each — "additional words"), then top-up cycling the
              full train pool. New-render rng = fresh random.Random(42).
  12960     = the 6480 set + 6480 further cycled renders (nested; no new words exist).

Word source is ONLY the target language's BSTD train-split TEXT (never test text —
rendering test words would leak test vocabulary). Rendering/verification reuses the
audited functions from prepare_zeroshot_loso.py (resolve_fonts, render_word,
build_vocab); fonts are ASSERTED identical to the per-script list recorded in
zeroshot_loso_meta_<tag>.json at the original build. Blank-rate assert <20% unchanged.

Outputs per (tag, budget):
  splits_zs_scale{B}_{tag}.json            train = 7000 src + {B} synth, val/test = Rung-B's
  grapheme_vocab_zs_scale{B}_{tag}.json    built from that budget's train set
  scaling_sweep_meta_{tag}.json            full provenance (counts, lexicon sizes, fonts)
New images land in synth_scale_{tag}/ (the original synth dir is never touched).

NO GPU. Usage:
  /c/ujjwalb/.conda/envs/ritu_scenetext/bin/python prepare_scaling_sweep.py
"""
import os, sys, json, random
from collections import Counter

BASE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, BASE)
from prepare_zeroshot_loso import resolve_fonts, render_word, build_vocab
from build_shared_grapheme_space import to_pivot

SEED = 42
LANGS = {'malayalam': 'Malayalam', 'kannada': 'Kannada', 'telugu': 'Telugu'}
BUDGETS = [810, 1620, 6480, 12960]   # 3240 = existing Rung-B result, reused
N_NEW = 9720                          # new renders per language (3240 for 6480 + 6480 more)
REC = os.path.join(BASE, 'benchmarks', 'bstd', 'Recognition')


def full_train_pool(lang):
    """(name, text) for every usable BSTD TRAIN record of `lang` (test never read)."""
    raw = json.load(open(os.path.join(REC, 'train_recognition_data.json')))
    out = []
    for fn, v in raw.items():
        if v.get('language') != lang:
            continue
        txt = (v.get('text') or '').strip()
        if not txt or txt == '###':
            continue
        out.append((os.path.splitext(fn)[0], txt))
    out.sort()                        # deterministic base order
    return out


def build_lang(tag):
    script = LANGS[tag]
    print(f"\n===== SCALING SWEEP build: {tag} ({script}) =====")
    rungB = json.load(open(os.path.join(BASE, f'splits_zeroshot_loso_rungB_{tag}.json')))
    synth_prefix = os.path.join(BASE, f'synth_zeroshot_loso_{tag}')
    synth = [r for r in rungB['train'] if r['image'].startswith(synth_prefix)]
    src   = [r for r in rungB['train'] if not r['image'].startswith(synth_prefix)]
    assert len(synth) == 3240, f"{tag}: expected 3240 existing synth, got {len(synth)}"
    assert len(src) == 7000,  f"{tag}: expected 7000 src train, got {len(src)}"
    val, test = rungB['val'], rungB['test']

    # fonts must match the original build exactly (environment-drift guard)
    meta0 = json.load(open(os.path.join(BASE, f'zeroshot_loso_meta_{tag}.json')))
    sample = synth[0]['orig_gt']
    fonts = resolve_fonts(script, sample)
    got = [os.path.basename(f) for f in fonts]
    assert got == meta0['fonts'], (f"{tag}: font set drifted! now {got}, "
                                   f"original {meta0['fonts']} — aborting per Amendment 5")

    # ── nested 810/1620 subsets of the existing 3240 ─────────────────────────
    base = sorted(synth, key=lambda r: r['image'])
    random.Random(SEED).shuffle(base)
    subsets = {810: base[:810], 1620: base[:1620]}

    # ── word supply for the 9720 new renders ─────────────────────────────────
    pool = full_train_pool(tag)
    used_names = {r['name'] for r in json.load(open(
        os.path.join(BASE, f'florence2_splits_bstd_{tag}_b1800.json')))['train']}
    unused = [(n, w) for (n, w) in pool if n not in used_names]
    rng_pick = random.Random(SEED)
    rng_pick.shuffle(unused)
    cycled = pool[:]                  # full pool, reshuffled per cycle
    word_slots = [w for (_, w) in unused for _ in range(2)]   # additional words first, x2
    while len(word_slots) < N_NEW + 2000:                     # headroom for render failures
        rng_pick.shuffle(cycled)
        word_slots += [w for (_, w) in cycled]
    print(f"  pool={len(pool)} records, unused={len(unused)} "
          f"(additional-words slots={2*len(unused)}), cycled top-up follows")

    # ── render the 9720 new images ───────────────────────────────────────────
    out_dir = os.path.join(BASE, f'synth_scale_{tag}')
    os.makedirs(out_dir, exist_ok=True)
    rng_render = random.Random(SEED)
    new_recs, k, fail = [], 0, 0
    for w in word_slots:
        if k >= N_NEW:
            break
        fp = os.path.join(out_dir, f'synthsc_{tag}_{k:06d}.jpg')
        if render_word(w, fonts, fp, rng_render):
            new_recs.append({'image': fp, 'gt': to_pivot(w),
                             'name': f'synthsc{k}', 'orig_gt': w})
            k += 1
        else:
            fail += 1
    blank_rate = 100 * fail / max(k + fail, 1)
    assert k == N_NEW, f"{tag}: only rendered {k}/{N_NEW}"
    assert blank_rate < 20, f"{tag}: blank rate {blank_rate:.1f}% — font/shaping problem"
    n_addword_imgs = min(2 * len(unused), N_NEW)
    print(f"  rendered {k} new images (fail {fail}, blank {blank_rate:.2f}%); "
          f"first {n_addword_imgs} are additional-word images")

    subsets[6480] = synth + new_recs[:3240]
    subsets[12960] = synth + new_recs

    # ── write splits + vocab per budget ──────────────────────────────────────
    lex = {}
    for b in BUDGETS:
        train = src + subsets[b]
        random.Random(SEED).shuffle(train)
        lex[b] = len({r['orig_gt'] for r in subsets[b]})
        json.dump({'train': train, 'val': val, 'test': test,
                   'metadata': {'rung': f'scale{b}', 'script': script, 'tag': tag,
                                'n_synth': len(subsets[b]), 'n_src': len(src),
                                'lexicon_unique_words': lex[b],
                                'amendment': 'PREREGISTRATION Amendment 5, 2026-07-18'}},
                  open(os.path.join(BASE, f'splits_zs_scale{b}_{tag}.json'), 'w'),
                  ensure_ascii=False)
        build_vocab(train, f'zs_scale{b}_{tag}',
                    os.path.join(BASE, f'grapheme_vocab_zs_scale{b}_{tag}.json'))
        print(f"  budget {b:>5}: train={len(train)}  synth={len(subsets[b])}  "
              f"unique-words={lex[b]}")

    meta = {'tag': tag, 'script': script, 'seed': SEED,
            'pool_records': len(pool), 'unused_records': len(unused),
            'additional_word_images': n_addword_imgs,
            'new_images': k, 'render_failures': fail,
            'blank_rate_pct': round(blank_rate, 2),
            'fonts': got, 'budgets': BUDGETS + [3240],
            'lexicon_unique_words_per_budget': {**{str(b): lex[b] for b in BUDGETS},
                                                '3240': len({r['orig_gt'] for r in synth})},
            'nested': '810⊂1620⊂3240⊂6480⊂12960'}
    json.dump(meta, open(os.path.join(BASE, f'scaling_sweep_meta_{tag}.json'), 'w'),
              ensure_ascii=False, indent=2)
    return meta


if __name__ == '__main__':
    for tag in LANGS:
        build_lang(tag)
    print("\nAll three languages built. Next: run_scaling_sweep.sh (waits for LOSO 27/27).")
