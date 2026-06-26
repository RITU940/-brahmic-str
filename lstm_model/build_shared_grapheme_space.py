"""
build_shared_grapheme_space.py  --  Part ② foundation (zero-shot cross-script).
==============================================================================
A DETERMINISTIC, REVERSIBLE mapping of every Brahmic script into one shared
"abugida primitive space", exploiting the ISCII-derived alignment of the Indic
Unicode blocks: corresponding letters sit at the SAME offset within each block
(e.g. KA = U+0915 Devanagari = U+0995 Bengali = U+0B95 Tamil = U+0C95 Telugu...).

We pivot through Devanagari (base U+0900). For any character:
  - if it lies in a Brahmic block  -> offset = cp - block_base ; pivot = 0x0900 + offset
  - otherwise (Latin, digits ASCII, punctuation, space, ZWJ/ZWNJ) -> pass through
The inverse maps a pivot codepoint back to a target script by the same offset.

WHY THIS MATTERS: a model trained to output PIVOT text learns one shared output
vocabulary; a held-out script's grapheme units land on pivot codes the model has
ALREADY seen from other scripts -> structural zero-shot transfer. The mapping is
reversible BY CONSTRUCTION (a per-script bijective offset shift), so reading in
pivot space == reading the target script. `--verify` proves that empirically.

NO GPU. Pure Unicode arithmetic + the project's grapheme segmenter.

Usage:
  PY=/home/ujjwal/miniconda3/envs/ritu_scenetext/bin/python
  $PY build_shared_grapheme_space.py --verify      # round-trip fidelity over BSTD
"""
import os, sys, json, argparse, unicodedata
from collections import Counter, defaultdict

BASE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, BASE)
from build_grapheme_vocab_lang import segment_graphemes_indic  # reuse exact segmenter

# Brahmic block bases (each block is 0x80 wide). Pivot = Devanagari.
PIVOT_BASE = 0x0900
SCRIPT_BASES = {
    'Devanagari': 0x0900, 'Bengali': 0x0980, 'Gurmukhi': 0x0A00, 'Gujarati': 0x0A80,
    'Oriya': 0x0B00, 'Tamil': 0x0B80, 'Telugu': 0x0C00, 'Kannada': 0x0C80,
    'Malayalam': 0x0D00,
}
BLOCK_WIDTH = 0x80

# Script-neutral (Unicode script=Common) characters encoded in the Devanagari
# block and SHARED verbatim by all Indic scripts -> must NOT be offset-shifted.
# ONLY the two dandas qualify safely: their offsets (0x64/0x65) are UNASSIGNED in
# every other Brahmic block, so passing them through creates no collision. (OM 0x50
# / ABBREVIATION 0x70 are deliberately excluded: those offsets hold real letters in
# other scripts — e.g. Assamese RA U+09F0 and Gurmukhi TIPPI U+0A70 both sit at 0x70.)
SHARED_IN_BRAHMIC = {
    0x0964,  # DANDA ।
    0x0965,  # DOUBLE DANDA ॥
}

# language -> writing system (same map as measure_script_descriptors.py)
LANG2SCRIPT = {
    'bengali': 'Bengali', 'assamese': 'Bengali',
    'hindi': 'Devanagari', 'marathi': 'Devanagari',
    'gujarati': 'Gujarati', 'punjabi': 'Gurmukhi',
    'kannada': 'Kannada', 'malayalam': 'Malayalam',
    'odia': 'Oriya', 'tamil': 'Tamil', 'telugu': 'Telugu',
    'english': 'Latin',
}


def _script_of_cp(cp):
    """Return the Brahmic script name whose block contains cp, else None."""
    for name, base in SCRIPT_BASES.items():
        if base <= cp < base + BLOCK_WIDTH:
            return name
    return None


def to_pivot(text):
    """Map any Brahmic text -> shared pivot (Devanagari-offset) space.
    Non-Brahmic characters pass through unchanged."""
    out = []
    for ch in text:
        cp = ord(ch)
        sc = _script_of_cp(cp)
        if sc is None or cp in SHARED_IN_BRAHMIC:
            out.append(ch)                      # Latin / ASCII / punct / ZW* / shared danda
        else:
            out.append(chr(PIVOT_BASE + (cp - SCRIPT_BASES[sc])))
    return ''.join(out)


def from_pivot(text, target_script):
    """Inverse: map pivot text back to a concrete target Brahmic script."""
    base = SCRIPT_BASES[target_script]
    out = []
    for ch in text:
        cp = ord(ch)
        if PIVOT_BASE <= cp < PIVOT_BASE + BLOCK_WIDTH and cp not in SHARED_IN_BRAHMIC:
            out.append(chr(base + (cp - PIVOT_BASE)))
        else:
            out.append(ch)                      # passthrough char (came through unchanged)
    return ''.join(out)


def pivot_graphemes(text):
    """Grapheme-cluster the text IN ITS ORIGINAL SCRIPT (categories are correct
    there), then map each cluster into pivot space. Returns list of pivot clusters."""
    return [to_pivot(g) for g in segment_graphemes_indic(text)]


# ── verification ────────────────────────────────────────────────────────────
def verify():
    train = json.load(open(os.path.join(BASE,
        'benchmarks/bstd/Recognition/train_recognition_data.json'), encoding='utf-8'))
    by_lang = defaultdict(list)
    for v in train.values():
        t = (v.get('text') or '').strip()
        if t:
            by_lang[v.get('language', '?')].append(t)

    print(f"{'lang':<10}{'script':<12}{'words':>7}{'char_RT%':>10}{'word_RT%':>10}"
          f"{'bad_chars(sample)':>22}")
    print('-' * 75)
    overall_bad = Counter()
    for lang in sorted(by_lang):
        script = LANG2SCRIPT.get(lang)
        if script in (None, 'Latin'):
            continue
        texts = by_lang[lang]
        nch = nch_ok = nw = nw_ok = 0
        bad = Counter()
        for t in texts:
            rt = from_pivot(to_pivot(t), script)
            nw += 1
            if rt == t:
                nw_ok += 1
            for a, b in zip(t, rt):
                nch += 1
                if a == b:
                    nch_ok += 1
                else:
                    bad[a] += 1
            # length mismatch (shouldn't happen: mapping is char-wise 1:1)
            nch += abs(len(t) - len(rt))
        sample = ''.join(list(bad)[:6])
        print(f"{lang:<10}{script:<12}{nw:>7}{100*nch_ok/max(nch,1):>9.3f}%"
              f"{100*nw_ok/max(nw,1):>9.2f}%{sample:>22}")
        overall_bad.update(bad)
    if overall_bad:
        print("\nNon-round-trip characters (codepoint : count) — investigate if any:")
        for ch, c in overall_bad.most_common(20):
            print(f"  U+{ord(ch):04X} {unicodedata.name(ch,'?'):<35} x{c}")
    else:
        print("\n✅ PERFECT round-trip for ALL Brahmic scripts (char & word level).")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--verify', action='store_true', help='round-trip fidelity over BSTD train')
    ap.add_argument('--demo', action='store_true')
    a = ap.parse_args()
    if a.demo:
        for s in ['தமிழ்', 'বাংলা', 'हिंदी', 'తెలుగు']:
            p = to_pivot(s)
            print(f"{s}  -> pivot {p}  -> back(Tamil) {from_pivot(p,'Tamil')}")
    if a.verify or not (a.demo):
        verify()


if __name__ == '__main__':
    main()
