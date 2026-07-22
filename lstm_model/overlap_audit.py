"""
Dual-submission overlap audit: WACV Paper-B vs the IJDAR Paper-A.

WACV 2027's declaration prohibits a concurrently submitted paper with
"20 percent or more overlap". This measures it three ways, because no venue
defines the metric precisely and a defensible answer needs all three:

  1. VERBATIM: fraction of WACV word-8-grams that also occur in the IJDAR
     paper (and the reverse direction) — catches copied prose.
  2. SECTION HOTSPOTS: the same measure per WACV section, so any single
     re-used passage is visible rather than diluted by the whole paper.
  3. ASSET SHARING: figures and result tables present in both.

Run:  python overlap_audit.py           (needs pdftotext on PATH)
"""
import os
import re
import subprocess
import sys
from collections import Counter

BASE = os.path.dirname(os.path.abspath(__file__))
A_PDF = os.path.join(BASE, "paper", "main.pdf")            # IJDAR (Paper A)
B_PDF = os.path.join(BASE, "paper_wacv", "main.pdf")       # WACV  (Paper B)
B_SEC = os.path.join(BASE, "paper_wacv", "sec")
N = 8


def pdf_text(path):
    out = subprocess.run(["pdftotext", "-q", path, "-"],
                         capture_output=True, text=True)
    return out.stdout


def words(text):
    text = re.sub(r"\s+", " ", text.lower())
    return re.findall(r"[a-z]+", text)


def shingles(ws, n=N):
    return Counter(tuple(ws[i:i + n]) for i in range(len(ws) - n + 1))


def overlap_pct(src_sh, ref_set):
    tot = sum(src_sh.values())
    if not tot:
        return 0.0, 0, 0
    hit = sum(c for sh, c in src_sh.items() if sh in ref_set)
    return 100.0 * hit / tot, hit, tot


def main():
    for p in (A_PDF, B_PDF):
        if not os.path.exists(p):
            sys.exit(f"missing {p}")

    a_ws, b_ws = words(pdf_text(A_PDF)), words(pdf_text(B_PDF))
    a_sh, b_sh = shingles(a_ws), shingles(b_ws)
    a_set, b_set = set(a_sh), set(b_sh)

    print("=" * 72)
    print("  DUAL-SUBMISSION OVERLAP AUDIT — WACV Paper-B vs IJDAR Paper-A")
    print("=" * 72)
    print(f"  IJDAR : {len(a_ws):,} words, {len(a_sh):,} distinct {N}-grams")
    print(f"  WACV  : {len(b_ws):,} words, {len(b_sh):,} distinct {N}-grams\n")

    pct_b, hit_b, tot_b = overlap_pct(b_sh, a_set)
    pct_a, hit_a, tot_a = overlap_pct(a_sh, b_set)
    print("1. VERBATIM OVERLAP")
    print(f"   WACV text found in IJDAR : {pct_b:6.2f}%  ({hit_b:,}/{tot_b:,} {N}-grams)")
    print(f"   IJDAR text found in WACV : {pct_a:6.2f}%  ({hit_a:,}/{tot_a:,} {N}-grams)")
    print(f"   WACV policy threshold    :  20.00%\n")

    print("2. PER-SECTION HOTSPOTS (WACV sections vs the whole IJDAR paper)")
    for fn in sorted(os.listdir(B_SEC)):
        if not fn.endswith(".tex"):
            continue
        with open(os.path.join(B_SEC, fn), encoding="utf-8") as f:
            tex = f.read()
        tex = re.sub(r"%.*", "", tex)
        tex = re.sub(r"\\[a-zA-Z]+\*?(\[[^\]]*\])?", " ", tex)   # strip macros
        sw = words(tex)
        if len(sw) < N:
            continue
        pct, hit, tot = overlap_pct(shingles(sw), a_set)
        flag = "  <-- inspect" if pct >= 10 else ""
        print(f"   {fn:<22} {pct:6.2f}%  ({hit}/{tot}){flag}")

    print("\n3. SHARED ASSETS")
    b_figs = set()
    for root, _, files in os.walk(os.path.join(BASE, "paper_wacv", "figs")):
        b_figs.update(os.path.splitext(f)[0] for f in files)
    a_figs = set()
    a_dir = os.path.join(BASE, "paper", "figures")
    if os.path.isdir(a_dir):
        for root, _, files in os.walk(a_dir):
            a_figs.update(os.path.splitext(f)[0] for f in files)
    shared = sorted(b_figs & a_figs)
    print(f"   WACV figures : {sorted(b_figs)}")
    print(f"   shared basenames with IJDAR: {shared if shared else 'none'}")

    print("\n" + "=" * 72)
    verdict = "PASS" if max(pct_a, pct_b) < 20 else "FAIL"
    print(f"  VERDICT: {verdict} — max directional verbatim overlap "
          f"{max(pct_a, pct_b):.2f}% vs the 20% threshold.")
    print("  Note: the threshold is about substantive content, not only prose;")
    print("  read section 2 above for any passage that must be rewritten, and")
    print("  keep the shared-claim inventory in the audit doc up to date.")
    print("=" * 72)


if __name__ == "__main__":
    main()
