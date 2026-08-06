#!/usr/bin/env python3
"""Round-trip audit of the shared grapheme pivot, over the nine held-out TEST
splits that the paper actually scores.

The paper claims pivot-WRR equals target-script WRR because pi is a per-script
bijective offset shift. That is an argument, not a measurement. This measures it:
for every scored test record, map the original-script label into pivot space and
back, and compare with the original.

Residual failures are then classified. A label is CROSS-SCRIPT if it contains a
Brahmic character from a block other than the one its language is annotated with
(a Bengali word filed under punjabi, Devanagari digits in an odia label). Those
break the round trip because the inverse targets a single script by design -- the
loss is in the benchmark's annotation, not in the mapping.

Two further things are measured here rather than argued. First, each cross-script
label is classified: WHOLLY foreign (every Brahmic character comes from another
block, i.e. the word was typed in the wrong script) versus MIXED (target-block and
foreign-block characters in one word, i.e. a homoglyph substituted inside an
otherwise correct word). Second, every system the paper reports -- our Rung B, the
supervised PARSeq ceiling and the Tesseract floor -- is scored on exactly those
labels, to check the claim that they are unwinnable under exact match rather than
merely hard.

Read-only. No GPU. Writes pivot_roundtrip_audit.json.

  /c/ujjwalb/.conda/envs/ritu_scenetext/bin/python audit_pivot_roundtrip.py
"""
import json, os, sys, unicodedata
from collections import Counter

BASE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, BASE)
from build_shared_grapheme_space import (
    to_pivot, from_pivot, SCRIPT_BASES, BLOCK_WIDTH, SHARED_IN_BRAHMIC,
)

SCRIPTS = ['tamil', 'telugu', 'kannada', 'malayalam', 'oriya',
           'gujarati', 'bengali', 'devanagari', 'gurmukhi']


def brahmic_blocks(text):
    """Set of Brahmic block names the text draws characters from (dandas, which
    are shared verbatim by every script, do not count as a block)."""
    hit = set()
    for ch in text:
        cp = ord(ch)
        if cp in SHARED_IN_BRAHMIC:
            continue
        for name, base in SCRIPT_BASES.items():
            if base <= cp < base + BLOCK_WIDTH:
                hit.add(name)
    return hit


def score_cross(script, target, cross_names, cross_idx, test):
    """How many of this script's cross-script labels does each reported system get
    exactly right? Returns {system: n_correct}. Systems with no per-sample file for
    this script are omitted rather than counted as zero."""
    got = {}
    for sysname in ('parseq', 'tesseract'):
        path = os.path.join(BASE, f'conf_{sysname}_{script}.json')
        if not os.path.exists(path):
            continue
        by = {r['name']: r for r in json.load(open(path, encoding='utf-8'))}
        got[sysname] = sum(1 for n in cross_names
                           if n in by and by[n].get('pred_native') == by[n].get('orig_gt'))
    # ours: conf_zs_loso_rungB_* is positional (no name field), so verify alignment
    path = os.path.join(BASE, f'conf_zs_loso_rungB_{script}.json')
    if os.path.exists(path):
        cf = json.load(open(path, encoding='utf-8'))
        if len(cf) == len(test) and all(a['gt'] == b['gt'] for a, b in zip(test, cf)):
            got['ours'] = sum(1 for i in cross_idx
                              if from_pivot(cf[i]['pred'], target) == test[i]['orig_gt'])
    return got


def main():
    rows, totals = [], Counter()
    offenders = Counter()
    sys_correct = Counter()
    kinds = Counter()

    for script in SCRIPTS:
        path = os.path.join(BASE, f'splits_zeroshot_loso_rungB_{script}.json')
        split = json.load(open(path, encoding='utf-8'))
        target = split['metadata']['script']
        recs = split['test']

        nw = nw_ok = nch = nch_ok = 0
        n_cross = n_cross_ok = 0
        cross_names, cross_idx = [], []
        for i, r in enumerate(recs):
            orig = r['orig_gt']
            rt = from_pivot(to_pivot(orig), target)
            nw += 1
            good = (rt == orig)
            nw_ok += good
            for a, b in zip(orig, rt):
                nch += 1
                nch_ok += (a == b)
            nch += abs(len(orig) - len(rt))

            present = brahmic_blocks(orig)
            foreign = present - {target}
            if foreign:
                n_cross += 1
                n_cross_ok += good
                cross_names.append(r['name'])
                cross_idx.append(i)
                # wholly foreign = word typed in the wrong script; mixed = homoglyph
                kinds['wholly_foreign' if target not in present else 'mixed'] += 1
            elif not good:
                for a, b in zip(orig, rt):
                    if a != b:
                        offenders[a] += 1

        for k, v in score_cross(script, target, cross_names, cross_idx, recs).items():
            sys_correct[k] += v

        char_rt = 100.0 * nch_ok / max(nch, 1)
        word_rt = 100.0 * nw_ok / max(nw, 1)
        # round trip restricted to labels written wholly in their annotated script
        pure = nw - n_cross
        pure_ok = nw_ok - n_cross_ok
        pure_rt = 100.0 * pure_ok / max(pure, 1)

        rows.append(dict(script=script, target=target, n=nw, char_rt=char_rt,
                         word_rt=word_rt, n_cross=n_cross, pure_rt=pure_rt))
        totals['n'] += nw
        totals['nw_ok'] += nw_ok
        totals['nch'] += nch
        totals['nch_ok'] += nch_ok
        totals['n_cross'] += n_cross

    print(f"{'script':<11}{'N':>6}{'char RT%':>10}{'word RT%':>10}"
          f"{'cross-script':>14}{'word RT% (pure)':>17}")
    print('-' * 68)
    for r in rows:
        print(f"{r['script']:<11}{r['n']:>6}{r['char_rt']:>9.3f}%{r['word_rt']:>9.3f}%"
              f"{r['n_cross']:>14}{r['pure_rt']:>16.3f}%")
    print('-' * 68)

    char_rt = 100.0 * totals['nch_ok'] / totals['nch']
    word_rt = 100.0 * totals['nw_ok'] / totals['n']
    cross_pct = 100.0 * totals['n_cross'] / totals['n']
    worst_char = min(r['char_rt'] for r in rows)
    worst_word = min(r['word_rt'] for r in rows)
    worst_pure = min(r['pure_rt'] for r in rows)

    print(f"pooled over {totals['n']} scored test labels: "
          f"char {char_rt:.3f}%  word {word_rt:.3f}%")
    print(f"worst script: char {worst_char:.3f}%  word {worst_word:.3f}%")
    print(f"cross-script labels: {totals['n_cross']} ({cross_pct:.2f}% of the test pool)")
    print(f"word RT over labels written wholly in their annotated script: "
          f"worst {worst_pure:.3f}%")

    if offenders:
        print("\nresidual failures NOT explained by cross-script labels:")
        for ch, k in offenders.most_common(20):
            print(f"  U+{ord(ch):04X} {unicodedata.name(ch, '?'):<40} x{k}")
    else:
        print("\nevery residual failure is a cross-script label; the mapping "
              "itself loses nothing on single-script text.")

    print(f"\nkind: {kinds['wholly_foreign']} wholly foreign (word typed in the wrong "
          f"script), {kinds['mixed']} mixed (homoglyph inside a correct word)")
    print("are they winnable? exact-match score on those same labels:")
    for k in ('ours', 'tesseract', 'parseq'):
        if k in sys_correct:
            print(f"  {k:<10} {sys_correct[k]} / {totals['n_cross']}")

    out = dict(per_script=rows, pooled=dict(
        n=totals['n'], char_rt=char_rt, word_rt=word_rt,
        n_cross=totals['n_cross'], cross_pct=cross_pct,
        worst_char=worst_char, worst_word=worst_word, worst_pure=worst_pure,
        cross_wholly_foreign=kinds['wholly_foreign'], cross_mixed=kinds['mixed'],
        cross_solved=dict(sys_correct)))
    with open(os.path.join(BASE, 'pivot_roundtrip_audit.json'), 'w') as f:
        json.dump(out, f, indent=2)
    print("\nwrote pivot_roundtrip_audit.json")


if __name__ == '__main__':
    main()
