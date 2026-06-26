"""
Convert ``florence2_splits.json`` into the format expected by
``colab_train_v3.py`` (CRNN_V3). This is a one-shot script.

Output: ``florence2_splits_for_crnn.json`` with the keys
``pairs / vocabulary / splits / stats / metadata`` so the existing
CRNN training loop can consume it without any code changes.

Run once before launching ``train_crnn_on_florence.py``.
"""
import json
import os
from collections import Counter

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
IN_PATH = os.path.join(BASE_DIR, 'florence2_splits.json')
OUT_PATH = os.path.join(BASE_DIR, 'florence2_splits_for_crnn.json')


def main():
    with open(IN_PATH, 'r', encoding='utf-8') as f:
        fl = json.load(f)

    # Build flat pairs list + split-index lists
    pairs = []
    splits_idx = {'train': [], 'val': [], 'test': []}
    for split_name in ('train', 'val', 'test'):
        for item in fl[split_name]:
            pairs.append({
                'image': item['image'],
                'gt': item['gt'],
                'name': item.get('name')
                        or os.path.splitext(os.path.basename(item['image']))[0],
            })
            splits_idx[split_name].append(len(pairs) - 1)

    # Build char-level vocabulary (CTC blank at idx 0)
    char_counter = Counter()
    for p in pairs:
        char_counter.update(p['gt'])
    chars = sorted(char_counter.keys())
    char2idx = {'<blank>': 0}
    for c in chars:
        char2idx[c] = len(char2idx)
    idx2char = {i: c for c, i in char2idx.items()}

    out = {
        'config': {
            'source': 'florence2_splits.json',
            'note': 'CRNN-formatted view of the Florence-2 splits — same splits, '
                    'identical to florence2_splits.json after stem mapping.',
        },
        'pairs': pairs,
        'vocabulary': {
            'char2idx': char2idx,
            'idx2char': {str(i): c for i, c in idx2char.items()},
        },
        'splits': splits_idx,
        'stats': {
            'num_classes': len(char2idx),
            'num_pairs': len(pairs),
            'num_train': len(splits_idx['train']),
            'num_val': len(splits_idx['val']),
            'num_test': len(splits_idx['test']),
            'unique_chars': len(chars),
        },
        'metadata': fl.get('metadata', {}),
    }

    with open(OUT_PATH, 'w', encoding='utf-8') as f:
        json.dump(out, f, ensure_ascii=False, indent=2)

    print(f"Wrote {OUT_PATH}")
    print(f"  Total pairs : {len(pairs)}")
    print(f"  Train/Val/Test: {len(splits_idx['train'])}/"
          f"{len(splits_idx['val'])}/{len(splits_idx['test'])}")
    print(f"  Vocabulary  : {len(char2idx)} classes "
          f"(incl. CTC blank at index 0)")


if __name__ == '__main__':
    main()
