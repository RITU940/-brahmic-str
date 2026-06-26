"""
Convert BSTD (Bharat Scene Text) Bengali recognition subset into our split format.
Produces florence2_splits_bstd.json with train/val/test (public benchmark).
GT JSON format: {"filename.jpg": {"path":..., "language":"bengali", "text":"word"}}
"""
import os, json, random
BASE = os.path.abspath(os.path.dirname(__file__))
BST = os.path.join(BASE, 'benchmarks', 'bstd', 'Recognition')
random.seed(42)

def load(split):
    j = json.load(open(os.path.join(BST, f'{split}_recognition_data.json'), encoding='utf-8'))
    img_dir = os.path.join(BST, split, 'bengali')
    on_disk = {f for f in os.listdir(img_dir)} if os.path.isdir(img_dir) else set()
    pairs = []
    for fname, meta in j.items():
        if meta.get('language') != 'bengali':
            continue
        base = os.path.basename(fname)
        if base not in on_disk:
            continue
        text = (meta.get('text') or '').strip()
        if not text or text == '###':
            continue
        pairs.append({'image': os.path.join(img_dir, base), 'gt': text, 'name': os.path.splitext(base)[0]})
    return pairs

train = load('train')
test = load('test')
# carve a val split from train (10%)
random.shuffle(train)
n_val = max(1, int(0.1 * len(train)))
val, train2 = train[:n_val], train[n_val:]

out = {'train': train2, 'val': val, 'test': test,
       'metadata': {'source': 'BSTD-bengali', 'train': len(train2), 'val': len(val), 'test': len(test)}}
json.dump(out, open(os.path.join(BASE, 'florence2_splits_bstd.json'), 'w'), ensure_ascii=False, indent=2)
print(f"BSTD Bengali: train {len(train2)} | val {len(val)} | test {len(test)}")
print(f"distinct test words: {len(set(p['gt'] for p in test))}")
print("Saved florence2_splits_bstd.json")
