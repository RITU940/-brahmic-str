"""
Paper-ready analyses produced from the existing eval JSON files.
NO GPU required — pure post-hoc analysis.

Produces:
  - token_compression.json / token_compression.md  : BPE vs grapheme tokens
    per Bengali test word (the motivation figure for the novelty).
  - wrr_by_word_length.json / wrr_by_word_length.md: per-bucket WRR for
    STANDARD vs GRAPHEME (shows where each tokenization helps).
  - master_results_table.md                        : the headline table.
  - figures/training_curves.png                    : per-epoch curves
    (generated only if matplotlib is available).
"""
import json
import os
import unicodedata
from collections import Counter, defaultdict

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
FIG_DIR = os.path.join(BASE_DIR, 'figures')
os.makedirs(FIG_DIR, exist_ok=True)


def normalize(s):
    s = unicodedata.normalize('NFC', (s or '').strip())
    for zw in ('‌', '‍', '​', '﻿'):
        s = s.replace(zw, '')
    return ' '.join(s.split())


def load_json(name):
    with open(os.path.join(BASE_DIR, name), 'r', encoding='utf-8') as f:
        return json.load(f)


# ── 1. Token compression: BPE vs grapheme tokens per Bengali word ───
def token_compression():
    from transformers import AutoTokenizer

    splits = load_json('florence2_splits.json')
    test_pairs = splits['test']

    print("[1/4] Loading Banglish tokenizer (BPE base) ...")
    HF_TOKEN = "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"
    bpe = AutoTokenizer.from_pretrained(
        "RocketFuel810/florence2-banglish-tokenizer", token=HF_TOKEN
    )

    # Build the grapheme tokenizer the same way train_florence2 does
    grapheme_vocab = load_json('grapheme_vocab.json')
    grapheme_tokens = [g for g in grapheme_vocab['grapheme2idx'].keys()
                       if g not in ('<blank>', '<unk>')]
    bpe_with_g = AutoTokenizer.from_pretrained(
        "RocketFuel810/florence2-banglish-tokenizer", token=HF_TOKEN
    )
    added = bpe_with_g.add_tokens(grapheme_tokens)
    print(f"  Grapheme tokens added: {added}")

    rows = []
    bpe_counts, gph_counts = [], []
    for p in test_pairs:
        text = p['gt']
        bpe_n = len(bpe.encode(text, add_special_tokens=False))
        gph_n = len(bpe_with_g.encode(text, add_special_tokens=False))
        rows.append({
            'gt': text, 'chars': len(text), 'bpe': bpe_n, 'grph': gph_n,
            'saved': bpe_n - gph_n,
        })
        bpe_counts.append(bpe_n)
        gph_counts.append(gph_n)

    total_chars = sum(r['chars'] for r in rows)
    avg_bpe = sum(bpe_counts) / len(rows)
    avg_gph = sum(gph_counts) / len(rows)
    avg_chars = total_chars / len(rows)
    compression = (avg_bpe - avg_gph) / max(avg_bpe, 1) * 100

    summary = {
        'num_test_words': len(rows),
        'avg_chars_per_word': round(avg_chars, 2),
        'avg_bpe_tokens_per_word': round(avg_bpe, 3),
        'avg_grapheme_tokens_per_word': round(avg_gph, 3),
        'bpe_total_tokens': sum(bpe_counts),
        'grapheme_total_tokens': sum(gph_counts),
        'compression_ratio_pct': round(compression, 2),
        'examples_max_savings': sorted(rows, key=lambda r: -r['saved'])[:10],
    }

    with open(os.path.join(BASE_DIR, 'token_compression.json'), 'w', encoding='utf-8') as f:
        json.dump({'summary': summary, 'per_word': rows}, f, ensure_ascii=False, indent=2)

    md = [
        '# Token compression — BPE vs grapheme on 307 Bengali test words\n',
        f'**Average Bengali chars / word:** {avg_chars:.2f}',
        f'**Average BPE tokens / word:** {avg_bpe:.2f}',
        f'**Average grapheme tokens / word:** {avg_gph:.2f}',
        f'**Token compression:** {compression:.2f}%  (lower = grapheme uses fewer tokens)\n',
        '## Top 10 words where grapheme tokenization saves the most tokens\n',
        '| Word | Chars | BPE | Grapheme | Tokens saved |',
        '|---|---|---|---|---|',
    ]
    for r in summary['examples_max_savings']:
        md.append(f"| `{r['gt']}` | {r['chars']} | {r['bpe']} | {r['grph']} | {r['saved']} |")
    with open(os.path.join(BASE_DIR, 'token_compression.md'), 'w', encoding='utf-8') as f:
        f.write('\n'.join(md))

    print(f"  Avg BPE  tokens/word: {avg_bpe:.3f}")
    print(f"  Avg grph tokens/word: {avg_gph:.3f}")
    print(f"  Compression: {compression:.2f}%")
    return summary


# ── 2. WRR stratified by word character length ──────────────────────
def wrr_by_word_length():
    std = load_json('florence2_results_standard.json')
    grp = load_json('florence2_results_grapheme.json')

    buckets = defaultdict(lambda: {'n': 0, 'std_ok': 0, 'grph_ok': 0})
    for sp, gp in zip(std['predictions'], grp['predictions']):
        gt_n = normalize(sp['gt'])
        n_chars = len(gt_n)
        if n_chars <= 3:
            bk = '1-3'
        elif n_chars <= 6:
            bk = '4-6'
        elif n_chars <= 9:
            bk = '7-9'
        elif n_chars <= 12:
            bk = '10-12'
        else:
            bk = '13+'
        b = buckets[bk]
        b['n'] += 1
        if normalize(sp['pred']) == gt_n:
            b['std_ok'] += 1
        if normalize(gp['pred']) == gt_n:
            b['grph_ok'] += 1

    rows = []
    for bk in ('1-3', '4-6', '7-9', '10-12', '13+'):
        b = buckets[bk]
        if b['n'] == 0:
            continue
        rows.append({
            'bucket': bk,
            'n': b['n'],
            'std_wrr': round(b['std_ok'] / b['n'] * 100, 2),
            'grph_wrr': round(b['grph_ok'] / b['n'] * 100, 2),
            'delta': round((b['grph_ok'] - b['std_ok']) / b['n'] * 100, 2),
        })

    with open(os.path.join(BASE_DIR, 'wrr_by_word_length.json'), 'w', encoding='utf-8') as f:
        json.dump(rows, f, indent=2)

    md = [
        '# WRR stratified by Bengali word length\n',
        '| Length (chars) | N | STD WRR (%) | GRPH WRR (%) | Δ (pp) |',
        '|---|---|---|---|---|',
    ]
    for r in rows:
        md.append(f"| {r['bucket']} | {r['n']} | {r['std_wrr']} | {r['grph_wrr']} | {r['delta']:+.2f} |")
    with open(os.path.join(BASE_DIR, 'wrr_by_word_length.md'), 'w', encoding='utf-8') as f:
        f.write('\n'.join(md))

    for r in rows:
        print(f"  {r['bucket']:>5} chars (n={r['n']:>3}): "
              f"STD {r['std_wrr']:>5.2f}%  GRPH {r['grph_wrr']:>5.2f}%  "
              f"Δ {r['delta']:+5.2f} pp")
    return rows


# ── 3. Master results table ─────────────────────────────────────────
def master_table():
    rows = []
    for label, fname in [
        ('Florence-2 zero-shot', 'florence2_results_zeroshot.json'),
        ('Florence-2 + STANDARD fine-tune', 'florence2_results_standard.json'),
        ('Florence-2 + GRAPHEME fine-tune (ours)', 'florence2_results_grapheme.json'),
    ]:
        if not os.path.exists(os.path.join(BASE_DIR, fname)):
            continue
        r = load_json(fname)
        rows.append({
            'method': label,
            'WRR': r.get('WRR'),
            'CER': r.get('CER'),
            'CharAcc': r.get('char_accuracy'),
            '1-NED': r.get('1-NED'),
            'FPS': r.get('fps'),
            'exact_matches': r.get('exact_matches'),
            'n': r.get('num_samples'),
        })

    md = [
        '# Master results — Bengali Scene Text (307 test images)\n',
        '| Method | WRR (%) | CER (%) | Char Acc (%) | 1-NED (%) | FPS | Exact / N |',
        '|---|---|---|---|---|---|---|',
    ]
    for r in rows:
        md.append(
            f"| {r['method']} | {r['WRR']:.2f} | {r['CER']:.2f} | "
            f"{r['CharAcc']:.2f} | {r['1-NED']:.2f} | {r['FPS']:.2f} | "
            f"{r['exact_matches']}/{r['n']} |"
        )
    md.append('\n_Note: A separate CRNN baseline trained on the same Florence-2 splits will be added once GPU is back (see `train_crnn_on_florence.py`)._\n')

    with open(os.path.join(BASE_DIR, 'master_results_table.md'), 'w', encoding='utf-8') as f:
        f.write('\n'.join(md))
    for r in rows:
        print(f"  {r['method']:42s} WRR={r['WRR']:5.2f}%  CER={r['CER']:5.2f}%")
    return rows


# ── 4. Training curves (only if matplotlib is real) ─────────────────
def training_curves():
    try:
        import matplotlib.pyplot as plt  # noqa
    except Exception:
        print("  matplotlib not installed — skipping curves figure.")
        return None
    std_log = load_json('training_log_florence2_standard.json')
    grp_log = load_json('training_log_florence2_grapheme.json')
    se = std_log['epochs']
    ge = grp_log['epochs']

    eps_s = [e['epoch'] for e in se]
    eps_g = [e['epoch'] for e in ge]

    fig, axes = plt.subplots(1, 2, figsize=(12, 4.5))
    axes[0].plot(eps_s, [e['val_wrr'] for e in se], 'o-', label='STANDARD')
    axes[0].plot(eps_g, [e['val_wrr'] for e in ge], 's-', label='GRAPHEME (ours)')
    axes[0].set_xlabel('Epoch'); axes[0].set_ylabel('Val WRR (%)')
    axes[0].set_title('Validation Word-Recognition Rate')
    axes[0].legend(); axes[0].grid(True, alpha=0.3)
    axes[1].plot(eps_s, [e['val_loss'] for e in se], 'o-', label='STANDARD')
    axes[1].plot(eps_g, [e['val_loss'] for e in ge], 's-', label='GRAPHEME (ours)')
    axes[1].set_xlabel('Epoch'); axes[1].set_ylabel('Val Cross-Entropy Loss')
    axes[1].set_title('Validation Loss')
    axes[1].legend(); axes[1].grid(True, alpha=0.3)
    plt.tight_layout()
    out = os.path.join(FIG_DIR, 'training_curves.png')
    plt.savefig(out, dpi=200, bbox_inches='tight')
    plt.close()
    print(f"  Wrote {out}")
    return out


if __name__ == '__main__':
    print('=' * 60)
    print('  Token compression')
    print('=' * 60)
    token_compression()
    print()
    print('=' * 60)
    print('  WRR by word length')
    print('=' * 60)
    wrr_by_word_length()
    print()
    print('=' * 60)
    print('  Master results table')
    print('=' * 60)
    master_table()
    print()
    print('=' * 60)
    print('  Training curves')
    print('=' * 60)
    training_curves()
