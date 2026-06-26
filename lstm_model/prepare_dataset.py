"""
Bengali Scene Text Dataset Preparation
- Loads image-GT pairs from Bengali/ and Bengali_gt/
- Builds character vocabulary
- Splits into Train (70%) / Validation (15%) / Test (15%)
- Saves split info and vocab to JSON
"""
import os
import json
import random
import sys
import unicodedata
from collections import Counter

# --- Configuration ---
BENGALI_IMG_DIR = "Bengali"
BENGALI_GT_DIR = "Bengali_gt"
OUTPUT_FILE = "dataset_splits.json"
SEED = 42


def configure_console():
    """Avoid Windows console crashes when printing Bengali text."""
    try:
        sys.stdout.reconfigure(encoding='utf-8', errors='replace')
        sys.stderr.reconfigure(encoding='utf-8', errors='replace')
    except Exception:
        pass

def normalize_text(text):
    """Normalize Bengali text: NFC normalization, remove zero-width chars."""
    text = unicodedata.normalize('NFC', text.strip())
    text = text.replace('\u200c', '').replace('\u200d', '').replace('\u200b', '').replace('\ufeff', '')
    return ' '.join(text.split())

def load_pairs(img_dir, gt_dir):
    """Load all valid image-GT pairs, excluding ### and empty GT."""
    images = {f.replace('.jpg', ''): f for f in os.listdir(img_dir) if f.endswith('.jpg')}
    gts = {f.replace('.txt', ''): f for f in os.listdir(gt_dir) if f.endswith('.txt')}
    
    pairs = []
    skipped = 0
    
    for name in sorted(images.keys()):
        if name not in gts:
            continue
        gt_path = os.path.join(gt_dir, gts[name])
        with open(gt_path, 'r', encoding='utf-8') as f:
            gt_text = f.read().strip()
        
        gt_norm = normalize_text(gt_text)
        if gt_norm == '###' or not gt_norm:
            skipped += 1
            continue
        
        pairs.append({
            'image': os.path.join(img_dir, images[name]),
            'gt': gt_norm,
            'name': name
        })
    
    print(f"Total images found: {len(images)}")
    print(f"Skipped (### / empty): {skipped}")
    print(f"Valid pairs: {len(pairs)}")
    return pairs

def build_vocabulary(pairs):
    """Build character vocabulary from all ground truth text."""
    char_counter = Counter()
    for p in pairs:
        for ch in p['gt']:
            char_counter[ch] += 1
    
    # Sort by frequency (most common first), then alphabetically for ties
    chars = sorted(char_counter.keys(), key=lambda c: (-char_counter[c], c))
    
    # Create char-to-index mapping (0 = CTC blank, 1+ = characters)
    char2idx = {ch: idx + 1 for idx, ch in enumerate(chars)}
    idx2char = {idx + 1: ch for idx, ch in enumerate(chars)}
    
    print(f"\nVocabulary size: {len(chars)} unique characters")
    print(f"Top 20 characters: {chars[:20]}")
    print(f"Total character instances: {sum(char_counter.values())}")
    
    # Print character distribution
    print("\nCharacter frequency distribution:")
    for ch in chars[:30]:
        print(f"  '{ch}' (U+{ord(ch):04X}): {char_counter[ch]}")
    
    return chars, char2idx, idx2char, char_counter

def split_dataset(pairs, train_ratio=0.70, val_ratio=0.15, test_ratio=0.15, seed=42):
    """Split dataset into train/val/test with random shuffle."""
    random.seed(seed)
    indices = list(range(len(pairs)))
    random.shuffle(indices)
    
    n = len(pairs)
    n_train = int(n * train_ratio)
    n_val = int(n * val_ratio)
    
    train_indices = indices[:n_train]
    val_indices = indices[n_train:n_train + n_val]
    test_indices = indices[n_train + n_val:]
    
    print(f"\nDataset Split (seed={seed}):")
    print(f"  Train:      {len(train_indices)} ({len(train_indices)/n*100:.1f}%)")
    print(f"  Validation: {len(val_indices)} ({len(val_indices)/n*100:.1f}%)")
    print(f"  Test:       {len(test_indices)} ({len(test_indices)/n*100:.1f}%)")
    print(f"  Total:      {n}")
    
    return train_indices, val_indices, test_indices

def compute_stats(pairs, indices, split_name):
    """Compute statistics for a split."""
    subset = [pairs[i] for i in indices]
    gt_lengths = [len(p['gt']) for p in subset]
    word_counts = [len(p['gt'].split()) for p in subset]
    
    print(f"\n  {split_name} Statistics:")
    print(f"    Samples: {len(subset)}")
    print(f"    GT length - min: {min(gt_lengths)}, max: {max(gt_lengths)}, "
          f"avg: {sum(gt_lengths)/len(gt_lengths):.1f}")
    print(f"    Words/sample - min: {min(word_counts)}, max: {max(word_counts)}, "
          f"avg: {sum(word_counts)/len(word_counts):.1f}")
    
    # Count single-word vs multi-word
    single = sum(1 for w in word_counts if w == 1)
    multi = len(word_counts) - single
    print(f"    Single-word: {single} ({single/len(subset)*100:.1f}%)")
    print(f"    Multi-word:  {multi} ({multi/len(subset)*100:.1f}%)")

def main():
    configure_console()
    print("=" * 60)
    print("  Bengali Scene Text Dataset Preparation")
    print("=" * 60)
    
    # Load pairs
    pairs = load_pairs(BENGALI_IMG_DIR, BENGALI_GT_DIR)
    if not pairs:
        print("ERROR: No valid pairs found!")
        return
    
    # Build vocabulary
    chars, char2idx, idx2char, char_freq = build_vocabulary(pairs)
    
    # Split dataset
    train_idx, val_idx, test_idx = split_dataset(pairs)
    
    # Compute stats per split
    compute_stats(pairs, train_idx, "Train")
    compute_stats(pairs, val_idx, "Validation")
    compute_stats(pairs, test_idx, "Test")
    
    # Save everything to JSON
    output = {
        'config': {
            'img_dir': BENGALI_IMG_DIR,
            'gt_dir': BENGALI_GT_DIR,
            'seed': SEED,
            'train_ratio': 0.70,
            'val_ratio': 0.15,
            'test_ratio': 0.15,
        },
        'vocabulary': {
            'chars': chars,
            'char2idx': char2idx,
            'num_classes': len(chars) + 1,  # +1 for CTC blank
        },
        'pairs': [{'image': p['image'], 'gt': p['gt'], 'name': p['name']} for p in pairs],
        'splits': {
            'train': train_idx,
            'val': val_idx,
            'test': test_idx,
        },
        'stats': {
            'total_pairs': len(pairs),
            'vocab_size': len(chars),
            'num_classes': len(chars) + 1,
        }
    }
    
    with open(OUTPUT_FILE, 'w', encoding='utf-8') as f:
        json.dump(output, f, ensure_ascii=False, indent=2)
    
    print(f"\n{'='*60}")
    print(f"  Dataset info saved to: {OUTPUT_FILE}")
    print(f"  Vocabulary: {len(chars)} chars → {len(chars)+1} classes (with CTC blank)")
    print(f"{'='*60}")

if __name__ == '__main__':
    main()
