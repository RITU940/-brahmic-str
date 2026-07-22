"""
rescore_vlm_extracted.py — DISCLOSED post-hoc re-scoring of the Qwen2.5-VL-7B
baseline (does NOT alter the preregistered Amendment-4c primary).

Motivation: the preregistered protocol used a minimal prompt ("Read the text in
the image.") and scored exact-match WRR on the model's raw generation. The chat
model wraps its answer in prose ("The text in the image is \"X\" which means..."),
so the raw exact WRR is 0.0 and the raw CER is a meaningless 1000%+ (the prose is
10-30x longer than the target word). Those raw numbers stay on record as the
declared primary; this script adds a THIRD, most-VLM-favourable column:

  extract the transcription (first double-quoted span; else strip a fixed set of
  lead-in phrases), then apply the SAME normalization + WRR/CER as every other
  result via metrics.evaluate_corpus.

Extraction is fixed BEFORE looking at scores and applied identically to all nine
scripts. Where the model reads the wrong script (e.g. Gurmukhi images returned in
Devanagari) extraction cannot help and the low score is a real model limitation.

Writes result_vlm_qwen25_extracted_<tag>.json per tag and prints a combined table
(preregistered exact | preregistered lenient | extracted-exact | extracted-CER).
"""
import os, re, sys, json

BASE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, BASE)
from metrics import evaluate_corpus, normalize_bengali  # shared normalization

TAGS = ['tamil', 'telugu', 'kannada', 'malayalam', 'oriya', 'gujarati',
        'bengali', 'devanagari', 'gurmukhi']

# double + smart double quotes only (single-quote class would eat contractions)
_QUOTED = re.compile(r'["“”„«「『]([^"“”„»」』]{1,80})["”“»」』]')
_LEADIN = re.compile(
    r'^\s*(the (text|word)( in the image)?( that appears| shown)?'
    r'\s*(is|reads|says|appears to (be|read))'
    r'|the image (shows|contains|reads|displays)'
    r'|it (reads|says)|reads|transcription)\s*[:\-]?\s*',
    re.IGNORECASE)


def extract(pred: str) -> str:
    """Pull the transcription out of a chat-wrapped answer. Deterministic."""
    m = _QUOTED.search(pred)
    if m:
        return m.group(1).strip()
    p = _LEADIN.sub('', pred)
    p = re.split(r'[.\n]', p, 1)[0]        # first clause only
    return p.strip()


def main():
    rows = []
    for tag in TAGS:
        conf = json.load(open(os.path.join(BASE, f'conf_vlm_qwen25_{tag}.json')))
        pre = json.load(open(os.path.join(BASE, f'result_vlm_qwen25_{tag}.json')))
        gts = [r['gt'] for r in conf]
        ext = [extract(r['pred']) for r in conf]
        quoted = sum(bool(_QUOTED.search(r['pred'])) for r in conf)

        m = evaluate_corpus(gts, ext)
        # also exact-after-normalization hit-rate cross-check (== WRR here)
        rec = {
            'model': pre['model'], 'script': tag, 'N': len(conf),
            'WRR_prereg_exact': pre['WRR'],          # declared primary (raw)
            'WRR_prereg_lenient': pre['WRR_lenient'], # declared secondary (substring)
            'WRR_extracted': round(m['WRR'], 2),      # NEW: exact WRR on extracted span
            'CER_extracted': round(m['CER'], 2),      # NEW: sane CER (raw CER was prose-length noise)
            'CharAcc_extracted': round(m['char_accuracy'], 2),
            'pct_quoted_span': round(100.0 * quoted / len(conf), 1),
            'extraction': 'first double-quoted span; else lead-in strip + first clause',
        }
        json.dump(rec, open(os.path.join(BASE, f'result_vlm_qwen25_extracted_{tag}.json'), 'w'),
                  ensure_ascii=False, indent=1)
        rows.append(rec)

    print("=" * 92)
    print("Qwen2.5-VL-7B zero-shot — preregistered primary vs disclosed extraction re-score")
    print("=" * 92)
    print(f"{'script':<12}{'N':>6}{'exact(raw)':>12}{'lenient':>10}"
          f"{'EXTRACTED':>11}{'CER_ext':>9}{'%quoted':>9}")
    for r in rows:
        print(f"{r['script']:<12}{r['N']:>6}{r['WRR_prereg_exact']:>12.2f}"
              f"{r['WRR_prereg_lenient']:>10.2f}{r['WRR_extracted']:>11.2f}"
              f"{r['CER_extracted']:>9.2f}{r['pct_quoted_span']:>9.1f}")
    n = sum(r['N'] for r in rows)
    macro = sum(r['WRR_extracted'] for r in rows) / len(rows)
    micro = sum(r['WRR_extracted'] * r['N'] for r in rows) / n
    print("-" * 92)
    print(f"{'MACRO avg':<12}{n:>6}{'':>12}{'':>10}{macro:>11.2f}")
    print(f"{'MICRO avg':<12}{'':>6}{'':>12}{'':>10}{micro:>11.2f}")


if __name__ == '__main__':
    main()
