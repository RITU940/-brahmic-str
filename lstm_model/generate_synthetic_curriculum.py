"""
Grapheme-rarity-weighted synthetic data (contribution #3).
=========================================================
Bengali OCR fails most on rare conjuncts (the long tail). We synthesise word
images, OVER-SAMPLING words that contain rare graphemes, so the model sees the
hard conjuncts far more often than their natural frequency.

Word source: real labels from our train + BSTD train (realistic lexicon).
Weight(word) = sum_g 1/(freq[g]+1)  -> rare-conjunct words sampled more.
Rendered with the 50 Noto Bengali fonts + varied backgrounds.

Outputs:
  synthetic_curriculum/images/*.jpg , synthetic_curriculum/gt/*.txt
  florence2_splits_synthaug.json  (train = real_train + synthetic, val/test = real)
"""
import os, sys, json, random
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from collections import Counter
from grapheme_tokenizer import segment_graphemes
from generate_synthetic import render_text_image, find_bengali_fonts

BASE = os.path.dirname(os.path.abspath(__file__))
random.seed(42)

def real_words():
    words = []
    for f in ['florence2_splits.json', 'florence2_splits_bstd.json']:
        p = os.path.join(BASE, f)
        if os.path.exists(p):
            for r in json.load(open(p))['train']:
                t = r['gt'].strip()
                if t and t != '###':
                    words.append(t)
    return words

def main():
    import argparse
    ap = argparse.ArgumentParser()
    ap.add_argument('--num', type=int, default=5000)
    args = ap.parse_args()

    words = real_words()
    # grapheme frequency over real training text
    gf = Counter()
    for w in words:
        gf.update(segment_graphemes(w))
    # unique words; weight by rare-grapheme content
    uniq = sorted(set(words))
    weights = []
    for w in uniq:
        gs = segment_graphemes(w)
        weights.append(sum(1.0 / (gf[g] + 1) for g in gs) if gs else 0.0)
    total = sum(weights) or 1.0
    probs = [x / total for x in weights]

    fonts = [f for f in find_bengali_fonts() if 'bengali' in f.lower() or 'beng' in f.lower()]
    if not fonts:
        fonts = find_bengali_fonts()
    print(f"real words: {len(words)} ({len(uniq)} unique) | fonts: {len(fonts)} | generating {args.num} synthetic")

    img_dir = os.path.join(BASE, 'synthetic_curriculum', 'images')
    gt_dir = os.path.join(BASE, 'synthetic_curriculum', 'gt')
    os.makedirs(img_dir, exist_ok=True); os.makedirs(gt_dir, exist_ok=True)

    chosen = random.choices(uniq, weights=probs, k=args.num)
    pairs = []
    for i, w in enumerate(chosen):
        try:
            img = render_text_image(w, font_path=random.choice(fonts))
        except Exception:
            continue
        name = f'synth_{i:06d}'
        ip = os.path.join(img_dir, name + '.jpg')
        img.convert('RGB').save(ip, 'JPEG', quality=90)
        open(os.path.join(gt_dir, name + '.txt'), 'w', encoding='utf-8').write(w)
        pairs.append({'image': ip, 'gt': w, 'name': name, 'synthetic': True})
        if (i + 1) % 1000 == 0:
            print(f"  rendered {i+1}/{args.num}")

    real = json.load(open(os.path.join(BASE, 'florence2_splits.json')))
    out = {'train': real['train'] + pairs, 'val': real['val'], 'test': real['test'],
           'metadata': {**real.get('metadata', {}), 'synthetic_added': len(pairs),
                        'synthetic_weighting': 'inverse-grapheme-frequency'}}
    json.dump(out, open(os.path.join(BASE, 'florence2_splits_synthaug.json'), 'w'), ensure_ascii=False, indent=2)
    # rarity sanity: top synthetic words should contain rare conjuncts
    print(f"generated {len(pairs)} synthetic | augmented train = {len(out['train'])}")
    print("sample synthetic words:", [p['gt'] for p in pairs[:8]])
    print("saved florence2_splits_synthaug.json")

if __name__ == '__main__':
    main()
