"""
Side-by-side comparison: STANDARD vs GRAPHEME predictions on the test set.

Produces:
  - prediction_comparison.json   (per-image GT + std-pred + grph-pred + which model won)
  - prediction_comparison.md     (Markdown table with clickable image links for visual inspection)
  - prediction_comparison_summary.txt (aggregate counts: both correct / grph only / std only / both wrong)
"""
import json
import os
import unicodedata

BASE_DIR = os.path.dirname(os.path.abspath(__file__))


def normalize(s: str) -> str:
    s = unicodedata.normalize('NFC', (s or '').strip())
    for zw in ('‌', '‍', '​', '﻿'):
        s = s.replace(zw, '')
    return ' '.join(s.split())


def load_results(path: str):
    with open(path, 'r', encoding='utf-8') as f:
        return json.load(f)


def main():
    std = load_results(os.path.join(BASE_DIR, 'florence2_results_standard.json'))
    grp = load_results(os.path.join(BASE_DIR, 'florence2_results_grapheme.json'))
    splits = load_results(os.path.join(BASE_DIR, 'florence2_splits.json'))

    test_pairs = splits['test']
    std_preds = std['predictions']
    grp_preds = grp['predictions']

    assert len(std_preds) == len(grp_preds) == len(test_pairs), (
        f"length mismatch: std={len(std_preds)} grp={len(grp_preds)} test={len(test_pairs)}"
    )

    rows = []
    for i, (sp, gp, tp) in enumerate(zip(std_preds, grp_preds, test_pairs)):
        gt_n = normalize(sp['gt'])
        std_n = normalize(sp['pred'])
        grp_n = normalize(gp['pred'])
        std_ok = (gt_n == std_n)
        grp_ok = (gt_n == grp_n)
        if std_ok and grp_ok:
            bucket = 'both_correct'
        elif grp_ok and not std_ok:
            bucket = 'grapheme_wins'
        elif std_ok and not grp_ok:
            bucket = 'standard_wins'
        else:
            bucket = 'both_wrong'
        rows.append({
            'idx': i,
            'image': tp['image'],
            'gt': sp['gt'],
            'std_pred': sp['pred'],
            'grph_pred': gp['pred'],
            'std_ok': std_ok,
            'grph_ok': grp_ok,
            'bucket': bucket,
        })

    # Counts
    counts = {'both_correct': 0, 'grapheme_wins': 0, 'standard_wins': 0, 'both_wrong': 0}
    for r in rows:
        counts[r['bucket']] += 1

    total = len(rows)
    summary = (
        "STANDARD vs GRAPHEME — 307 test images\n"
        f"  Both correct        : {counts['both_correct']:4d} ({counts['both_correct']/total*100:5.2f}%)\n"
        f"  GRAPHEME wins only  : {counts['grapheme_wins']:4d} ({counts['grapheme_wins']/total*100:5.2f}%)  <-- paper's gain\n"
        f"  STANDARD wins only  : {counts['standard_wins']:4d} ({counts['standard_wins']/total*100:5.2f}%)\n"
        f"  Both wrong          : {counts['both_wrong']:4d} ({counts['both_wrong']/total*100:5.2f}%)\n"
        f"\n  Net Δ for grapheme  : {counts['grapheme_wins'] - counts['standard_wins']:+d} words\n"
    )
    print(summary)
    with open(os.path.join(BASE_DIR, 'prediction_comparison_summary.txt'), 'w', encoding='utf-8') as f:
        f.write(summary)

    with open(os.path.join(BASE_DIR, 'prediction_comparison.json'), 'w', encoding='utf-8') as f:
        json.dump({'counts': counts, 'rows': rows}, f, ensure_ascii=False, indent=2)

    # Markdown table — grouped by bucket, image paths as clickable links
    md = ['# Florence-2 Bengali OCR — Prediction Comparison\n',
          f'**Test set: 307 images** | STANDARD WRR 55.70% | GRAPHEME WRR 58.96% (+3.26 pp)\n']
    md.append(f'\n```\n{summary}```\n')

    bucket_titles = {
        'grapheme_wins': '## ★ GRAPHEME wins (paper headline examples)',
        'standard_wins': '## STANDARD wins (where grapheme hurt)',
        'both_correct': '## Both correct (sanity-check easy cases — first 30)',
        'both_wrong': '## Both wrong (hard cases — future work)',
    }
    bucket_limits = {'both_correct': 30, 'both_wrong': 30, 'grapheme_wins': None, 'standard_wins': None}

    for bucket in ('grapheme_wins', 'standard_wins', 'both_correct', 'both_wrong'):
        md.append(f'\n{bucket_titles[bucket]}\n')
        md.append('| # | Image | GT | STANDARD | GRAPHEME |')
        md.append('|---|---|---|---|---|')
        items = [r for r in rows if r['bucket'] == bucket]
        limit = bucket_limits[bucket]
        for r in (items if limit is None else items[:limit]):
            rel = os.path.relpath(r['image'], BASE_DIR)
            std_mark = '✓' if r['std_ok'] else '✗'
            grph_mark = '✓' if r['grph_ok'] else '✗'
            md.append(
                f"| {r['idx']} | [{os.path.basename(r['image'])}]({rel}) | `{r['gt']}` "
                f"| {std_mark} `{r['std_pred']}` | {grph_mark} `{r['grph_pred']}` |"
            )
        if limit is not None and len(items) > limit:
            md.append(f'\n*({len(items) - limit} more cases hidden — see prediction_comparison.json)*\n')

    with open(os.path.join(BASE_DIR, 'prediction_comparison.md'), 'w', encoding='utf-8') as f:
        f.write('\n'.join(md))

    print('Outputs:')
    print(f'  - {os.path.join(BASE_DIR, "prediction_comparison.json")}')
    print(f'  - {os.path.join(BASE_DIR, "prediction_comparison.md")}')
    print(f'  - {os.path.join(BASE_DIR, "prediction_comparison_summary.txt")}')


if __name__ == '__main__':
    main()
