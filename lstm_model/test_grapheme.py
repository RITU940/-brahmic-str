"""
Test script for Bengali Grapheme Tokenizer
===========================================

Loads dataset_splits.json, builds grapheme vocabulary from training data,
and reports comparison statistics between character-level and grapheme-level
tokenization.
"""

import json
import sys
import os

# Add the project directory to path
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from grapheme_tokenizer import (
    BengaliGraphemeTokenizer,
    segment_graphemes,
    compare_char_vs_grapheme,
)


def main():
    # -----------------------------------------------------------------------
    # 1. Load dataset
    # -----------------------------------------------------------------------
    dataset_path = os.path.join(os.path.dirname(os.path.abspath(__file__)), "dataset_splits.json")
    print(f"Loading dataset from: {dataset_path}")
    
    with open(dataset_path, "r", encoding="utf-8") as f:
        dataset = json.load(f)
    
    pairs = dataset["pairs"]
    splits = dataset["splits"]
    char_vocab = dataset["vocabulary"]
    
    print(f"Total pairs: {len(pairs)}")
    print(f"Train: {len(splits['train'])}, Val: {len(splits['val'])}, Test: {len(splits['test'])}")
    print()
    
    # -----------------------------------------------------------------------
    # 2. Extract training texts
    # -----------------------------------------------------------------------
    train_indices = splits["train"]
    train_texts = [pairs[i]["gt"] for i in train_indices]
    all_texts = [p["gt"] for p in pairs]
    
    print(f"Training texts: {len(train_texts)}")
    print(f"All texts: {len(all_texts)}")
    print()
    
    # -----------------------------------------------------------------------
    # 3. Build grapheme tokenizer from training data
    # -----------------------------------------------------------------------
    tokenizer = BengaliGraphemeTokenizer()
    tokenizer.build_vocab(train_texts)
    
    # -----------------------------------------------------------------------
    # 4. Comparison: unique chars vs unique graphemes
    # -----------------------------------------------------------------------
    print("=" * 70)
    print("COMPARISON: Character-Level vs Grapheme-Level Tokenization")
    print("=" * 70)
    
    # From existing char vocabulary in the dataset
    existing_chars = char_vocab["chars"]
    existing_num_classes = char_vocab["num_classes"]
    
    print(f"\n--- Existing Character Vocabulary (from dataset_splits.json) ---")
    print(f"  Unique characters:      {len(existing_chars)}")
    print(f"  num_classes (with blank): {existing_num_classes}")
    
    # Grapheme vocabulary from training data
    stats = tokenizer.get_vocab_stats()
    print(f"\n--- Grapheme Vocabulary (built from training data) ---")
    print(f"  Unique graphemes:       {stats['num_graphemes']}")
    print(f"  num_classes (with special): {stats['vocab_size']}")
    print(f"  Grapheme length distribution: {stats['grapheme_length_distribution']}")
    
    # Full comparison on all data
    comparison = compare_char_vs_grapheme(all_texts)
    print(f"\n--- Detailed Comparison (all {comparison['num_texts']} texts) ---")
    print(f"  Unique characters:     {comparison['unique_chars']}")
    print(f"  Unique graphemes:      {comparison['unique_graphemes']}")
    print(f"  Total char tokens:     {comparison['total_char_tokens']}")
    print(f"  Total grapheme tokens: {comparison['total_grapheme_tokens']}")
    print(f"  Avg chars/text:        {comparison['avg_chars_per_text']:.2f}")
    print(f"  Avg graphemes/text:    {comparison['avg_graphemes_per_text']:.2f}")
    print(f"  Compression ratio:     {comparison['compression_ratio']:.3f}x")
    print(f"  Vocab size change:     {comparison['vocab_reduction_pct']:+.1f}%")
    
    # -----------------------------------------------------------------------
    # 5. Show 10 example segmentations
    # -----------------------------------------------------------------------
    print("\n" + "=" * 70)
    print("EXAMPLES: Text -> Grapheme Segmentation (10 samples)")
    print("=" * 70)
    
    # Pick diverse examples: some short, some with conjuncts
    example_texts = [
        "বিয়ে",           # bi-ye
        "রবীন্দ্রনাথ",    # rabindranath (complex conjuncts)
        "স্কুল",          # skul (conjunct)
        "সাম্প্রদায়িক",   # samprodayik (multiple conjuncts)
        "ফার্ণিচার",      # furniture (halant)
        "বাংলা",          # bangla (anusvara)
        "অনুষ্ঠানে",      # anushthane (conjunct)
        "ছাত্রসাথী",      # chhatrosathi (conjunct + vowel sign)
        "ক্ষ",            # ksha (classic conjunct)
        "পাঞ্জাবী",       # panjabi (conjunct)
    ]
    
    for i, text in enumerate(example_texts):
        graphemes = segment_graphemes(text)
        encoded = tokenizer.encode(text)
        decoded = tokenizer.decode(encoded)
        
        print(f"\n  Example {i+1}: \"{text}\"")
        print(f"    Graphemes ({len(graphemes)}): {graphemes}")
        print(f"    Char count: {len(text)}")
        print(f"    Encoded:    {encoded}")
        print(f"    Decoded:    \"{decoded}\"")
        print(f"    Round-trip: {'✓' if decoded == text else '✗ MISMATCH'}")
    
    # -----------------------------------------------------------------------
    # 6. Show 10 examples from actual dataset
    # -----------------------------------------------------------------------
    print("\n" + "=" * 70)
    print("DATASET EXAMPLES: Actual texts from training data")
    print("=" * 70)
    
    import random
    random.seed(42)
    sample_indices = random.sample(range(len(train_texts)), min(10, len(train_texts)))
    
    for i, idx in enumerate(sample_indices):
        text = train_texts[idx]
        graphemes = segment_graphemes(text)
        encoded = tokenizer.encode(text)
        decoded = tokenizer.decode(encoded)
        
        print(f"\n  Dataset Example {i+1}: \"{text}\"")
        print(f"    Graphemes ({len(graphemes)}): {graphemes}")
        print(f"    Chars: {len(text)} -> Graphemes: {len(graphemes)}")
        print(f"    Round-trip: {'✓' if decoded == text else '✗ MISMATCH'}")
    
    # -----------------------------------------------------------------------
    # 7. Test encode/decode round-trip on all training data
    # -----------------------------------------------------------------------
    print("\n" + "=" * 70)
    print("ROUND-TRIP TEST: Encode -> Decode all training texts")
    print("=" * 70)
    
    mismatches = 0
    unk_count = 0
    for text in all_texts:
        encoded = tokenizer.encode(text)
        decoded = tokenizer.decode(encoded)
        if decoded != text:
            mismatches += 1
        if tokenizer.UNK_IDX in encoded:
            unk_count += 1
    
    print(f"  Total texts tested: {len(all_texts)}")
    print(f"  Successful round-trips: {len(all_texts) - mismatches}")
    print(f"  Mismatches: {mismatches}")
    print(f"  Texts with <unk> tokens: {unk_count}")
    print(f"  Round-trip accuracy: {(len(all_texts) - mismatches) / len(all_texts) * 100:.2f}%")
    
    # -----------------------------------------------------------------------
    # 8. Summary
    # -----------------------------------------------------------------------
    print("\n" + "=" * 70)
    print("SUMMARY")
    print("=" * 70)
    print(f"  Character vocabulary size:  {len(existing_chars)} unique chars ({existing_num_classes} with blank)")
    print(f"  Grapheme vocabulary size:   {stats['num_graphemes']} unique graphemes ({stats['vocab_size']} with special tokens)")
    
    if stats['num_graphemes'] > len(existing_chars):
        print(f"  Grapheme vocab is LARGER by {stats['num_graphemes'] - len(existing_chars)} "
              f"({(stats['num_graphemes'] / len(existing_chars) - 1) * 100:.1f}% increase)")
        print(f"  BUT sequence lengths are shorter by {comparison['compression_ratio']:.2f}x on average")
    else:
        print(f"  Grapheme vocab is SMALLER by {len(existing_chars) - stats['num_graphemes']} "
              f"({(1 - stats['num_graphemes'] / len(existing_chars)) * 100:.1f}% reduction)")
    
    print("\n  Done!")


if __name__ == "__main__":
    main()
