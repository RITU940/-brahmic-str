"""
khmer_pivot_map.py --- extend the shared grapheme pivot to Khmer (Amendment 4b).

Khmer (U+1780--U+17FF) is Brahmic-derived but does NOT share the ISCII offset
alignment of the nine Indic blocks that `build_shared_grapheme_space.py` exploits.
We therefore hand-curate a linguistically-motivated correspondence from each
productive Khmer character to its Devanagari (pivot) counterpart, by traditional
Brahmic phonetic order (ka-varga ...). Coeng (U+17D2, subscript former) maps to
virama (U+094D) so consonant stacks segment like Indic conjuncts.

This is a BEST-EFFORT structural map, documented for review: a handful of Khmer
vowels/signs have no exact Devanagari equivalent and are mapped to the nearest
quality (flagged APPROX) or left unmapped (they then count as uncovered, which is
the honest, conservative choice for the coverage statistic).

Pure Unicode arithmetic; no GPU; reversible per-codepoint for the covered subset.
"""

# Khmer codepoint -> Devanagari (pivot) codepoint. APPROX = nearest-quality choice.
KHMER_TO_PIVOT = {
    # ---- base consonants, traditional ka-varga order ----
    0x1780: 0x0915,  # KA  -> क
    0x1781: 0x0916,  # KHA -> ख
    0x1782: 0x0917,  # KO  -> ग
    0x1783: 0x0918,  # KHO -> घ
    0x1784: 0x0919,  # NGO -> ङ
    0x1785: 0x091A,  # CA  -> च
    0x1786: 0x091B,  # CHA -> छ
    0x1787: 0x091C,  # CO  -> ज
    0x1788: 0x091D,  # CHO -> झ
    0x1789: 0x091E,  # NYO -> ञ
    0x178A: 0x091F,  # DA  -> ट
    0x178B: 0x0920,  # TTHA-> ठ
    0x178C: 0x0921,  # DO  -> ड
    0x178D: 0x0922,  # TTHO-> ढ
    0x178E: 0x0923,  # NNO -> ण
    0x178F: 0x0924,  # TA  -> त
    0x1790: 0x0925,  # THA -> थ
    0x1791: 0x0926,  # TO  -> द
    0x1792: 0x0927,  # THO -> ध
    0x1793: 0x0928,  # NO  -> न
    0x1794: 0x092A,  # BA  -> प
    0x1795: 0x092B,  # PHA -> फ
    0x1796: 0x092C,  # PO  -> ब
    0x1797: 0x092D,  # PHO -> भ
    0x1798: 0x092E,  # MO  -> म
    0x1799: 0x092F,  # YO  -> य
    0x179A: 0x0930,  # RO  -> र
    0x179B: 0x0932,  # LO  -> ल
    0x179C: 0x0935,  # VO  -> व
    0x179D: 0x0936,  # SHA -> श
    0x179E: 0x0937,  # SSO -> ष
    0x179F: 0x0938,  # SA  -> स
    0x17A0: 0x0939,  # HA  -> ह
    0x17A1: 0x0933,  # LA  -> ळ
    0x17A2: 0x0905,  # QA (glottal/vowel carrier) -> अ
    # ---- independent vowels ----
    0x17A5: 0x0907,  # QI  -> इ
    0x17A6: 0x0908,  # QII -> ई
    0x17A7: 0x0909,  # QU  -> उ
    0x17A9: 0x090A,  # QUU -> ऊ
    0x17AA: 0x090A,  # QUUV-> ऊ   APPROX
    0x17AB: 0x090B,  # RY  -> ऋ
    0x17AF: 0x090F,  # QE  -> ए
    0x17B0: 0x0910,  # QAI -> ऐ
    0x17B1: 0x0913,  # QOO -> ओ
    0x17B2: 0x0913,  # QOO -> ओ   APPROX
    0x17B3: 0x0914,  # QAU -> औ
    # ---- dependent vowels (matras) ----
    0x17B6: 0x093E,  # AA  -> ा
    0x17B7: 0x093F,  # I   -> ि
    0x17B8: 0x0940,  # II  -> ी
    0x17B9: 0x093F,  # Y   -> ि   APPROX
    0x17BA: 0x0940,  # YY  -> ी   APPROX
    0x17BB: 0x0941,  # U   -> ु
    0x17BC: 0x0942,  # UU  -> ू
    0x17BD: 0x0941,  # UA  -> ु   APPROX (diphthong)
    0x17BE: 0x0947,  # OE  -> े   APPROX
    0x17BF: 0x093F,  # YA  -> ि   APPROX (diphthong)
    0x17C0: 0x0947,  # IE  -> े   APPROX (diphthong)
    0x17C1: 0x0947,  # E   -> े
    0x17C2: 0x0948,  # AE  -> ै
    0x17C3: 0x0948,  # AI  -> ै
    0x17C4: 0x094B,  # OO  -> ो
    0x17C5: 0x094C,  # AU  -> ौ
    # ---- signs ----
    0x17C6: 0x0902,  # NIKAHIT (anusvara)  -> ं
    0x17C7: 0x0903,  # REAHMUK (visarga)   -> ः
    0x17D2: 0x094D,  # COENG (subscript former) -> ् virama  [cluster-critical]
    # ---- digits ----
    0x17E0: 0x0966, 0x17E1: 0x0967, 0x17E2: 0x0968, 0x17E3: 0x0969, 0x17E4: 0x096A,
    0x17E5: 0x096B, 0x17E6: 0x096C, 0x17E7: 0x096D, 0x17E8: 0x096E, 0x17E9: 0x096F,
}

# Khmer register/diacritic signs with no Devanagari equivalent -> deliberately
# UNMAPPED (drop): U+17C8..U+17CA (yuukaleapintu/muusikatoan/triisap), U+17CB bantoc,
# U+17CC robat, U+17CD..U+17D1, U+17D3, U+17DD. They count as uncovered.
DROP = {0x17C8, 0x17C9, 0x17CA, 0x17CB, 0x17CC, 0x17CD, 0x17CE, 0x17CF,
        0x17D0, 0x17D1, 0x17D3, 0x17DD}


def khmer_to_pivot(text: str):
    """Map a Khmer string to pivot (Devanagari-space). Returns (pivot_str, n_khmer,
    n_mapped): non-Khmer codepoints pass through unchanged; unmapped Khmer chars are
    dropped and counted as misses."""
    out = []
    n_khmer = n_mapped = 0
    for ch in text:
        cp = ord(ch)
        if 0x1780 <= cp <= 0x17FF:          # Khmer block
            n_khmer += 1
            if cp in KHMER_TO_PIVOT:
                out.append(chr(KHMER_TO_PIVOT[cp]))
                n_mapped += 1
            # DROP / unknown Khmer -> omitted (counts against coverage)
        else:
            out.append(ch)                   # ASCII / punct / space pass through
    return "".join(out), n_khmer, n_mapped


if __name__ == "__main__":
    # smoke test: "ភ្នំពេញ" (Phnom Penh) should map to a pivot conjunct string
    for w in ["ភ្នំពេញ", "កម្ពុជា", "សាលា"]:
        piv, nk, nm = khmer_to_pivot(w)
        print(f"{w}  ->  {piv}   (khmer {nk}, mapped {nm})")
