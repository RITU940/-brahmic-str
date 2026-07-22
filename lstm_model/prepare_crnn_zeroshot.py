"""
Convert a zero-shot LOSO split file into the CRNN_V3 dataset format
(``pairs / vocabulary / splits / stats / metadata``), so the existing
``colab_train_v3.py`` training loop consumes it without any code changes.

Protocol mirrors the Florence-2 rungs exactly (Amendment 6 /
``PROSPECTIVE_PREDICTION_ARCHITECTURE.md``):
  * same ``splits_zeroshot_loso_rung{R}_{tag}.json`` — same images, same
    pivot ``gt``, same train/val/test partition;
  * the character vocabulary is built from **TRAIN records only**, exactly as
    ``prepare_zeroshot_loso.py:build_vocab`` does for the grapheme vocab;
  * the test split is copied verbatim (no length filtering here) so N matches
    the Florence-2 rung.

USAGE:
    python prepare_crnn_zeroshot.py --rung B --tag tamil
"""
import argparse
import json
import os
from collections import Counter

BASE = os.path.dirname(os.path.abspath(__file__))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--rung', required=True, choices=['A', 'B'])
    ap.add_argument('--tag', required=True)
    a = ap.parse_args()

    in_path = os.path.join(BASE, f'splits_zeroshot_loso_rung{a.rung}_{a.tag}.json')
    out_path = os.path.join(BASE, f'splits_crnn_zs_rung{a.rung}_{a.tag}.json')
    with open(in_path, 'r', encoding='utf-8') as f:
        src = json.load(f)

    pairs, splits_idx = [], {'train': [], 'val': [], 'test': []}
    for split in ('train', 'val', 'test'):
        for item in src[split]:
            pairs.append({
                'image': item['image'],
                'gt': item['gt'],
                'name': item.get('name')
                        or os.path.splitext(os.path.basename(item['image']))[0],
            })
            splits_idx[split].append(len(pairs) - 1)

    # Vocabulary from TRAIN only — the target script's test text is never read.
    train_chars = Counter()
    for i in splits_idx['train']:
        train_chars.update(pairs[i]['gt'])
    char2idx = {'<blank>': 0}
    for c in sorted(train_chars):
        char2idx[c] = len(char2idx)

    # Honest coverage bookkeeping: characters the model can never emit.
    test_chars = Counter()
    for i in splits_idx['test']:
        test_chars.update(pairs[i]['gt'])
    tot = sum(test_chars.values())
    oov_tok = sum(c for ch, c in test_chars.items() if ch not in char2idx)
    oov_typ = sum(1 for ch in test_chars if ch not in char2idx)
    reachable = sum(1 for i in splits_idx['test']
                    if all(ch in char2idx for ch in pairs[i]['gt']))

    out = {
        'config': {
            'source': os.path.basename(in_path),
            'note': 'CRNN view of the zero-shot LOSO splits; vocabulary from TRAIN only.',
        },
        'pairs': pairs,
        'vocabulary': {
            'char2idx': char2idx,
            'idx2char': {str(i): c for c, i in char2idx.items()},
        },
        'splits': splits_idx,
        'stats': {
            'num_classes': len(char2idx),
            'n_train': len(splits_idx['train']),
            'n_val': len(splits_idx['val']),
            'n_test': len(splits_idx['test']),
            'test_char_token_oov_pct': round(100 * oov_tok / max(tot, 1), 2),
            'test_char_type_oov_count': oov_typ,
            'test_words_fully_reachable_pct': round(
                100 * reachable / max(len(splits_idx['test']), 1), 2),
        },
        'metadata': dict(src.get('metadata', {}), backbone='CRNN_V3',
                         rung=a.rung, tag=a.tag),
    }
    with open(out_path, 'w', encoding='utf-8') as f:
        json.dump(out, f, ensure_ascii=False)

    s = out['stats']
    print(f"[prep-crnn] rung {a.rung} {a.tag}: train={s['n_train']} val={s['n_val']} "
          f"test={s['n_test']} classes={s['num_classes']}")
    print(f"[prep-crnn]   test char-token OOV {s['test_char_token_oov_pct']}% "
          f"({s['test_char_type_oov_count']} types); "
          f"{s['test_words_fully_reachable_pct']}% of test words fully emittable")
    print(f"[prep-crnn]   -> {os.path.basename(out_path)}")


if __name__ == '__main__':
    main()
