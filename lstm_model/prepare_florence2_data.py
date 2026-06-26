"""
Data Preparation for Bengali Scene Text Research Paper
=======================================================
Reads images from Bengali/ and ground truth from Bengali_gt/,
builds grapheme vocabulary, creates train/val/test splits,
and outputs JSON files for all downstream training scripts.

Splits are stratified by document ID to avoid data leakage
(all lines from the same image go to the same split).

Outputs:
  - florence2_splits.json   → {train/val/test pairs + grapheme vocab}
  - grapheme_vocab.json     → grapheme vocabulary for Florence-2 token injection
  - data_statistics.json    → dataset statistics for the paper
"""
import os
import sys
import json
import random
import hashlib
from collections import Counter, defaultdict
from typing import List, Dict, Tuple

sys.stdout.reconfigure(encoding='utf-8', errors='replace')

# Add parent dir for imports
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from grapheme_tokenizer import segment_graphemes, BengaliGraphemeTokenizer, compare_char_vs_grapheme


# ── Configuration ──────────────────────────────────────────────────
BASE_DIR = os.path.abspath(os.path.dirname(__file__))


def _first_existing_dir(*paths: str, suffixes=None) -> str:
    if suffixes:
        for path in paths:
            if not path or not os.path.isdir(path):
                continue
            if any(name.lower().endswith(suffixes) for name in os.listdir(path)):
                return path
    for path in paths:
        if path and os.path.isdir(path):
            return path
    return paths[0]


IMG_DIR = os.path.abspath(os.environ.get(
    "FLORENCE_IMG_DIR",
    os.path.join(BASE_DIR, "Bengali"),
))
GT_DIR = os.path.abspath(os.environ.get(
    "FLORENCE_GT_DIR",
    _first_existing_dir(
        os.path.join(BASE_DIR, "Bengali_gt"),
        os.path.join(BASE_DIR, "Bengali", "Bengali_gt"),
        suffixes=('.txt',),
    ),
))
IMAGE_EXTS = ('.jpg', '.jpeg', '.png')

TRAIN_RATIO = 0.80
VAL_RATIO = 0.10
TEST_RATIO = 0.10
RANDOM_SEED = 42


def load_pairs() -> List[Dict]:
    """Load all image-GT pairs, skipping ### (illegible) and empty entries."""
    if not os.path.isdir(IMG_DIR):
        raise FileNotFoundError(f"Image directory not found: {IMG_DIR}")
    if not os.path.isdir(GT_DIR):
        raise FileNotFoundError(f"Ground-truth directory not found: {GT_DIR}")

    images = {
        os.path.splitext(f)[0]: f
        for f in os.listdir(IMG_DIR)
        if f.lower().endswith(IMAGE_EXTS)
    }
    gts = {
        os.path.splitext(f)[0]: f
        for f in os.listdir(GT_DIR)
        if f.lower().endswith('.txt')
    }

    print(f"Image directory: {IMG_DIR}")
    print(f"GT directory:    {GT_DIR}")

    pairs = []
    skipped_hash = 0
    skipped_empty = 0
    missing_gt = 0
    
    for name in sorted(images.keys()):
        if name not in gts:
            missing_gt += 1
            continue
        
        gt_path = os.path.join(GT_DIR, gts[name])
        gt_text = open(gt_path, 'r', encoding='utf-8').read().strip()
        
        if gt_text == '###':
            skipped_hash += 1
            continue
        if not gt_text:
            skipped_empty += 1
            continue
        
        pairs.append({
            'image': os.path.join(IMG_DIR, images[name]),
            'gt': gt_text,
            'name': name,
        })
    
    print(f"Dataset Loading:")
    print(f"  Total images found:  {len(images)}")
    print(f"  Total GT files:      {len(gts)}")
    print(f"  Missing GT:          {missing_gt}")
    print(f"  Skipped ### :        {skipped_hash}")
    print(f"  Skipped empty:       {skipped_empty}")
    print(f"  Valid pairs:         {len(pairs)}")
    
    return pairs


def extract_doc_id(name: str) -> str:
    """Extract document ID from filename like 'gt_img_6401_line3' -> '6401'.
    
    All lines from the same document go into the same split to avoid
    data leakage (seeing part of a sign during training, rest during test).
    """
    parts = name.split('_')
    # Format: gt_img_XXXX_lineY
    for i, p in enumerate(parts):
        if p == 'img' and i + 1 < len(parts):
            return parts[i + 1]
    return name  # fallback


def split_by_document(pairs: List[Dict], seed: int = RANDOM_SEED) -> Tuple[List, List, List]:
    """Split pairs by document ID to avoid data leakage."""
    # Group by document
    doc_pairs = defaultdict(list)
    for p in pairs:
        doc_id = extract_doc_id(p['name'])
        doc_pairs[doc_id].append(p)
    
    # Shuffle document IDs
    doc_ids = sorted(doc_pairs.keys())
    random.seed(seed)
    random.shuffle(doc_ids)
    
    n = len(doc_ids)
    train_end = int(n * TRAIN_RATIO)
    val_end = int(n * (TRAIN_RATIO + VAL_RATIO))
    
    train_docs = set(doc_ids[:train_end])
    val_docs = set(doc_ids[train_end:val_end])
    test_docs = set(doc_ids[val_end:])
    
    train_pairs = [p for p in pairs if extract_doc_id(p['name']) in train_docs]
    val_pairs = [p for p in pairs if extract_doc_id(p['name']) in val_docs]
    test_pairs = [p for p in pairs if extract_doc_id(p['name']) in test_docs]
    
    print(f"\nSplit by document (seed={seed}):")
    print(f"  Documents: {len(train_docs)} train / {len(val_docs)} val / {len(test_docs)} test")
    print(f"  Samples:   {len(train_pairs)} train / {len(val_pairs)} val / {len(test_pairs)} test")
    
    return train_pairs, val_pairs, test_pairs


def build_grapheme_vocab(pairs: List[Dict]) -> Tuple[BengaliGraphemeTokenizer, Dict]:
    """Build grapheme vocabulary from all texts and return tokenizer + stats."""
    texts = [p['gt'] for p in pairs]
    
    tokenizer = BengaliGraphemeTokenizer()
    tokenizer.build_vocab(texts, min_freq=1)
    
    # Char vs grapheme comparison
    comparison = compare_char_vs_grapheme(texts)
    
    stats = tokenizer.get_vocab_stats()
    stats['comparison'] = comparison
    
    print(f"\nGrapheme Vocabulary:")
    print(f"  Vocab size (incl. special): {stats['vocab_size']}")
    print(f"  Unique graphemes:           {stats['num_graphemes']}")
    print(f"  Unique chars (baseline):    {comparison['unique_chars']}")
    print(f"  Compression ratio:          {comparison['compression_ratio']:.2f}x")
    print(f"  Vocab reduction:            {comparison['vocab_reduction_pct']:.1f}%")
    
    return tokenizer, stats


def compute_dataset_statistics(
    train_pairs: List[Dict],
    val_pairs: List[Dict],
    test_pairs: List[Dict],
    tokenizer: BengaliGraphemeTokenizer
) -> Dict:
    """Compute statistics for the paper's dataset description section."""
    all_pairs = train_pairs + val_pairs + test_pairs
    texts = [p['gt'] for p in all_pairs]
    
    # Text length distributions
    char_lengths = [len(t) for t in texts]
    grapheme_lengths = [len(segment_graphemes(t)) for t in texts]
    word_counts = [len(t.split()) for t in texts]
    
    # Character frequency
    char_freq = Counter()
    for t in texts:
        char_freq.update(t)
    
    # Grapheme frequency
    grapheme_freq = Counter()
    for t in texts:
        grapheme_freq.update(segment_graphemes(t))
    
    stats = {
        'total_samples': len(all_pairs),
        'train_samples': len(train_pairs),
        'val_samples': len(val_pairs),
        'test_samples': len(test_pairs),
        'unique_chars': len(char_freq),
        'unique_graphemes': tokenizer.vocab_size - 2,  # exclude blank/unk
        'total_chars': sum(char_lengths),
        'total_graphemes': sum(grapheme_lengths),
        'avg_char_length': sum(char_lengths) / max(len(char_lengths), 1),
        'avg_grapheme_length': sum(grapheme_lengths) / max(len(grapheme_lengths), 1),
        'avg_words_per_sample': sum(word_counts) / max(len(word_counts), 1),
        'max_char_length': max(char_lengths) if char_lengths else 0,
        'max_grapheme_length': max(grapheme_lengths) if grapheme_lengths else 0,
        'max_words': max(word_counts) if word_counts else 0,
        'top_20_graphemes': grapheme_freq.most_common(20),
    }
    
    return stats


def save_splits(
    train_pairs: List[Dict],
    val_pairs: List[Dict],
    test_pairs: List[Dict],
    tokenizer: BengaliGraphemeTokenizer,
    vocab_stats: Dict,
    dataset_stats: Dict,
):
    """Save all outputs needed by downstream scripts."""
    
    # 1. Main splits file (used by train_florence2.py, evaluate_florence2.py, etc.)
    splits_data = {
        'train': train_pairs,
        'val': val_pairs,
        'test': test_pairs,
        'metadata': {
            'img_dir': IMG_DIR,
            'gt_dir': GT_DIR,
            'train_ratio': TRAIN_RATIO,
            'val_ratio': VAL_RATIO,
            'test_ratio': TEST_RATIO,
            'seed': RANDOM_SEED,
            'total_samples': len(train_pairs) + len(val_pairs) + len(test_pairs),
        }
    }
    
    splits_path = os.path.join(BASE_DIR, 'florence2_splits.json')
    with open(splits_path, 'w', encoding='utf-8') as f:
        json.dump(splits_data, f, ensure_ascii=False, indent=2)
    print(f"\nSaved: {splits_path}")
    
    # 2. Grapheme vocabulary (used for Florence-2 token injection)
    vocab_path = os.path.join(BASE_DIR, 'grapheme_vocab.json')
    tokenizer.save(vocab_path)
    print(f"Saved: {vocab_path}")
    
    # 3. Dataset statistics (for paper)
    stats_path = os.path.join(BASE_DIR, 'data_statistics.json')
    stats_output = {
        'vocab_stats': vocab_stats,
        'dataset_stats': dataset_stats,
    }
    with open(stats_path, 'w', encoding='utf-8') as f:
        json.dump(stats_output, f, ensure_ascii=False, indent=2, default=str)
    print(f"Saved: {stats_path}")
    
    # 4. Also save a simple CSV for quick inspection
    csv_path = os.path.join(BASE_DIR, 'test_samples.csv')
    with open(csv_path, 'w', encoding='utf-8') as f:
        f.write('image_path,ground_truth\n')
        for p in test_pairs:
            gt_escaped = p['gt'].replace('"', '""')
            f.write(f'"{p["image"]}","{gt_escaped}"\n')
    print(f"Saved: {csv_path} ({len(test_pairs)} test samples)")


def main():
    print("=" * 60)
    print("  Bengali Scene Text — Data Preparation")
    print("=" * 60)
    
    # 1. Load all valid pairs
    pairs = load_pairs()
    if not pairs:
        print("ERROR: No valid pairs found!")
        sys.exit(1)
    
    # 2. Build grapheme vocabulary from ALL data
    tokenizer, vocab_stats = build_grapheme_vocab(pairs)
    
    # 3. Split by document ID
    train_pairs, val_pairs, test_pairs = split_by_document(pairs)
    
    # 4. Compute dataset statistics
    dataset_stats = compute_dataset_statistics(train_pairs, val_pairs, test_pairs, tokenizer)
    
    # 5. Save everything
    save_splits(train_pairs, val_pairs, test_pairs, tokenizer, vocab_stats, dataset_stats)
    
    # 6. Print summary for paper
    print(f"\n{'='*60}")
    print(f"  DATASET SUMMARY (for paper)")
    print(f"{'='*60}")
    print(f"  Total samples:      {dataset_stats['total_samples']}")
    print(f"  Train / Val / Test: {dataset_stats['train_samples']} / "
          f"{dataset_stats['val_samples']} / {dataset_stats['test_samples']}")
    print(f"  Unique characters:  {dataset_stats['unique_chars']}")
    print(f"  Unique graphemes:   {dataset_stats['unique_graphemes']}")
    print(f"  Avg chars/sample:   {dataset_stats['avg_char_length']:.1f}")
    print(f"  Avg graphemes/sample: {dataset_stats['avg_grapheme_length']:.1f}")
    print(f"  Max chars/sample:   {dataset_stats['max_char_length']}")
    print(f"  Total characters:   {dataset_stats['total_chars']}")
    print(f"{'='*60}")
    print(f"  Ready for training! Run train_florence2.py next.")


if __name__ == '__main__':
    main()
