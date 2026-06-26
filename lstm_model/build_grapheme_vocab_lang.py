"""
Script-agnostic Indic grapheme-cluster vocab builder (cross-script extension).
==============================================================================
Generalizes the Bengali grapheme rule (base + (virama+consonant)* + matras +
modifiers) to ANY Indic script using Unicode categories:
  - combining marks (Mn/Mc) attach to the preceding base
  - a virama followed by a letter joins the next consonant (conjunct/aksara)
  - ZWJ/ZWNJ are consumed into the current cluster

Builds grapheme_vocab_<lang>.json (same format as grapheme_vocab.json) from a
splits file's train+val text.

Usage:
  python build_grapheme_vocab_lang.py --splits florence2_splits_bstd_hindi.json --lang hindi
"""
import os, sys, json, argparse, unicodedata
from collections import Counter

BASE = os.path.dirname(os.path.abspath(__file__))

# Viramas across Indic scripts (Devanagari, Bengali, Gurmukhi, Gujarati, Odia,
# Tamil, Telugu, Kannada, Malayalam)
VIRAMAS = {'्', '্', '੍', '્', '୍',
           '்', '్', '್', '്'}
ZWJ_ZWNJ = {'‌', '‍'}


def segment_graphemes_indic(text):
    clusters, i, n = [], 0, len(text)
    while i < n:
        c = text[i]
        if c in ZWJ_ZWNJ:           # stray joiner: skip
            i += 1
            continue
        cl = c
        i += 1
        while i < n:
            ch = text[i]
            if ch in ZWJ_ZWNJ:
                cl += ch
                i += 1
                continue
            cat = unicodedata.category(ch)
            if cat in ('Mn', 'Mc'):
                cl += ch
                i += 1
                # virama joins the following consonant into the cluster
                if ch in VIRAMAS and i < n and unicodedata.category(text[i]).startswith('L'):
                    cl += text[i]
                    i += 1
            else:
                break
        clusters.append(cl)
    return clusters


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--splits', required=True)
    ap.add_argument('--lang', required=True)
    ap.add_argument('--min_freq', type=int, default=1)
    ap.add_argument('--out_suffix', default='',
                    help='append to output filename, e.g. _b1800 (non-destructive)')
    a = ap.parse_args()

    sp = json.load(open(os.path.join(BASE, a.splits), encoding='utf-8'))
    freq = Counter()
    for part in ('train', 'val'):
        for r in sp.get(part, []):
            freq.update(segment_graphemes_indic(r['gt'].strip()))

    graphemes = sorted(g for g, c in freq.items() if c >= a.min_freq and g.strip())
    grapheme2idx = {'<blank>': 0, '<unk>': 1}
    for g in graphemes:
        grapheme2idx[g] = len(grapheme2idx)

    out = {'grapheme2idx': grapheme2idx,
           'metadata': {'lang': a.lang, 'source_splits': a.splits,
                        'num_graphemes': len(graphemes),
                        'segmenter': 'indic-generic (Mn/Mc + virama-join)'}}
    out_path = os.path.join(BASE, f'grapheme_vocab_{a.lang}{a.out_suffix}.json')
    json.dump(out, open(out_path, 'w'), ensure_ascii=False, indent=2)
    multi = sum(1 for g in graphemes if len(g) > 1)
    print(f"{a.lang}: {len(graphemes)} graphemes ({multi} multi-char/conjunct-bearing)")
    print(f"top conjunct clusters: {[g for g, _ in freq.most_common(200) if any(v in g for v in VIRAMAS)][:10]}")
    print(f"saved {out_path}")


if __name__ == '__main__':
    main()
