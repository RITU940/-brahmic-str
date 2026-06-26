"""
Split long multi-word GT sequences into individual word-level samples.

Instead of rejecting samples with GT > 25 chars, we:
1. Split the GT text by spaces into individual words
2. Proportionally divide the image horizontally based on character count
3. Create separate image-word pairs

This preserves ALL data while keeping each sample within CTC-friendly lengths.
"""
import os, sys, json
from PIL import Image

sys.stdout.reconfigure(encoding='utf-8', errors='replace')

IMG_DIR = 'Bengali'
GT_DIR = 'Bengali_gt'
MAX_WORD_LEN = 25  # Only split if GT exceeds this


def main():
    print("=" * 60)
    print("  Splitting Long Sequences into Word-Level Samples")
    print("=" * 60)

    # Get all existing pairs
    images = {f.replace('.jpg', ''): f for f in os.listdir(IMG_DIR) if f.endswith('.jpg')}
    gts = {f.replace('.txt', ''): f for f in os.listdir(GT_DIR) if f.endswith('.txt')}

    long_count = 0
    new_crops = 0
    skipped_single = 0
    errors = 0

    names = sorted(set(images.keys()) & set(gts.keys()))
    print(f"  Total existing pairs: {len(names)}")

    for name in names:
        gt_path = os.path.join(GT_DIR, gts[name])
        with open(gt_path, 'r', encoding='utf-8') as f:
            gt_text = f.read().strip()

        if len(gt_text) <= MAX_WORD_LEN:
            continue  # Already short enough

        words = gt_text.split()
        if len(words) <= 1:
            skipped_single += 1
            continue  # Single long word - can't split

        long_count += 1

        # Load and split image proportionally
        img_path = os.path.join(IMG_DIR, images[name])
        try:
            img = Image.open(img_path)
        except Exception as e:
            errors += 1
            continue

        img_w, img_h = img.size

        # Calculate proportional widths based on character count
        char_counts = [len(w) for w in words]
        total_chars = sum(char_counts)

        x_start = 0
        for wi, word in enumerate(words):
            # Skip empty words or ### markers
            if not word.strip() or word == '###':
                continue

            # Proportional width for this word
            word_width = int(img_w * char_counts[wi] / total_chars)
            # Add small padding overlap (2px each side) for better crops
            x_end = min(x_start + word_width, img_w)
            x_left = max(0, x_start - 2)
            x_right = min(img_w, x_end + 2)

            if x_right - x_left < 5:
                x_start = x_end
                continue

            # Crop word region
            word_img = img.crop((x_left, 0, x_right, img_h))

            # Save new word-level pair
            new_name = f"{name}_w{wi}"
            word_img.save(os.path.join(IMG_DIR, f"{new_name}.jpg"), 'JPEG', quality=95)
            with open(os.path.join(GT_DIR, f"{new_name}.txt"), 'w', encoding='utf-8') as f:
                f.write(word)

            new_crops += 1
            x_start = x_end

        # Remove the original long-sequence files
        os.remove(img_path)
        os.remove(gt_path)

    # Count final dataset
    final_count = len([f for f in os.listdir(IMG_DIR) if f.endswith('.jpg')])

    print(f"\n  Results:")
    print(f"    Long sequences found (>{MAX_WORD_LEN} chars): {long_count}")
    print(f"    Skipped (single long word):   {skipped_single}")
    print(f"    Errors:                       {errors}")
    print(f"    New word-level crops created:  {new_crops}")
    print(f"    Original long files removed:   {long_count}")
    print(f"    Final dataset size:            {final_count}")
    print("=" * 60)


if __name__ == '__main__':
    main()
