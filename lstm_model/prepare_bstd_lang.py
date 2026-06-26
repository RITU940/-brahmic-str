"""
Convert any BSTD language subset into our split format (cross-script extension
of prepare_bstd_bengali.py). Optionally subsample train to a fixed budget so
languages are compared at the same (low-resource) training scale.

Usage:
  python prepare_bstd_lang.py --lang assamese
  python prepare_bstd_lang.py --lang hindi --train_budget 5000
"""
import os, json, random, argparse

BASE = os.path.abspath(os.path.dirname(__file__))
BST = os.path.join(BASE, 'benchmarks', 'bstd', 'Recognition')


def load(split, lang):
    j = json.load(open(os.path.join(BST, f'{split}_recognition_data.json'), encoding='utf-8'))
    img_dir = os.path.join(BST, split, lang)
    on_disk = {f for f in os.listdir(img_dir)} if os.path.isdir(img_dir) else set()
    pairs = []
    for fname, meta in j.items():
        if meta.get('language') != lang:
            continue
        base = os.path.basename(fname)
        if base not in on_disk:
            continue
        text = (meta.get('text') or '').strip()
        if not text or text == '###':
            continue
        pairs.append({'image': os.path.join(img_dir, base), 'gt': text,
                      'name': os.path.splitext(base)[0]})
    return pairs


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--lang', required=True)
    ap.add_argument('--train_budget', type=int, default=None,
                    help='subsample train to N samples (test always kept full)')
    ap.add_argument('--out_suffix', default='',
                    help='append to output filename, e.g. _b1800 (non-destructive)')
    a = ap.parse_args()
    random.seed(42)

    train = load('train', a.lang)
    test = load('test', a.lang)
    random.shuffle(train)
    if a.train_budget and len(train) > a.train_budget:
        train = train[:a.train_budget]
    n_val = max(1, int(0.1 * len(train)))
    val, train2 = train[:n_val], train[n_val:]

    out = {'train': train2, 'val': val, 'test': test,
           'metadata': {'source': f'BSTD-{a.lang}', 'train': len(train2),
                        'val': len(val), 'test': len(test),
                        'train_budget': a.train_budget}}
    path = os.path.join(BASE, f'florence2_splits_bstd_{a.lang}{a.out_suffix}.json')
    json.dump(out, open(path, 'w'), ensure_ascii=False, indent=2)
    print(f"BSTD {a.lang}: train {len(train2)} | val {len(val)} | test {len(test)} "
          f"| distinct test words {len(set(p['gt'] for p in test))}")
    print(f"saved {path}")


if __name__ == '__main__':
    main()
