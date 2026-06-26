"""
Crop Bengali words from annotated_images_ritu1 using GT annotations in gt/.

GT format (Ritu1):
  Image: <filename>
  Image width height: <w>, <h>
  ...
  Bounding Boxes (clustered):
  Big Region ID: <id>
    Small Region ID: <id>
      Points: [(x1,y1), (x2,y2), ...]
      Transcript: <text>

For each Bengali word region, crops the bounding polygon from the source image
and saves it to Bengali/ with a matching GT text file in Bengali_gt/.
"""

import os
import re
import sys
import glob
import numpy as np
from PIL import Image

# --- Configuration ---
RITU1_IMG_DIR = r"C:\lstm_model\annotated_images_ritu1"
RITU1_GT_DIR = r"C:\lstm_model\gt"
OUT_IMG_DIR = r"C:\lstm_model\Bengali"
OUT_GT_DIR = r"C:\lstm_model\Bengali_gt"

# Start numbering after existing files
# We'll detect the max existing index dynamically


def configure_console():
    try:
        sys.stdout.reconfigure(encoding='utf-8', errors='replace')
        sys.stderr.reconfigure(encoding='utf-8', errors='replace')
    except Exception:
        pass


def get_next_index(out_img_dir):
    """Find the next available index for naming cropped images."""
    max_idx = 0
    for f in os.listdir(out_img_dir):
        if f.endswith('.jpg'):
            # Try to extract numeric suffix
            # Existing format: gt_img_XXXX_lineN.jpg or ritu1_NNNN.jpg
            pass
    # Just count existing files
    return len([f for f in os.listdir(out_img_dir) if f.endswith('.jpg')])


def is_bengali_text(text):
    """Check if text contains Bengali characters (Unicode range 0x0980-0x09FF)."""
    bengali_count = sum(1 for ch in text if '\u0980' <= ch <= '\u09FF')
    return bengali_count > 0


def parse_ritu1_gt(gt_path):
    """
    Parse a Ritu1 GT file and extract bounding box regions with transcripts.
    
    Returns: list of dicts with keys: 'points', 'transcript'
    """
    with open(gt_path, 'r', encoding='utf-8') as f:
        content = f.read()
    
    regions = []
    
    # Split by "Small Region ID:" to get individual regions
    # Each region has Points and Transcript
    region_blocks = re.split(r'Small Region ID:', content)
    
    for block in region_blocks[1:]:  # Skip the header part before first region
        # Extract points
        points_match = re.search(
            r'Points(?:\s*\(Pixel\))?:\s*\[(.*?)\](?:\s*$|\s*\n)',
            block, re.DOTALL
        )
        transcript_match = re.search(
            r'Transcript:\s*(.+?)(?:\n|$)',
            block
        )
        
        if points_match and transcript_match:
            points_str = points_match.group(1)
            transcript = transcript_match.group(1).strip()
            
            # Skip empty, ###, None, or non-Bengali transcripts
            if not transcript or transcript == '###' or transcript == 'None':
                continue
            if transcript == '#':
                continue
                
            # Parse points: (x, y), (x, y), ...
            point_pairs = re.findall(r'\(([^)]+)\)', points_str)
            points = []
            for pp in point_pairs:
                coords = pp.split(',')
                if len(coords) >= 2:
                    try:
                        x = float(coords[0].strip())
                        y = float(coords[1].strip())
                        points.append((x, y))
                    except ValueError:
                        continue
            
            if len(points) >= 3 and is_bengali_text(transcript):
                regions.append({
                    'points': points,
                    'transcript': transcript
                })
    
    return regions


def crop_polygon_region(img, points):
    """
    Crop a region defined by polygon points from an image.
    Uses the bounding rectangle of the polygon for simplicity.
    """
    img_w, img_h = img.size
    
    xs = [p[0] for p in points]
    ys = [p[1] for p in points]
    
    x_min = max(0, int(min(xs)) - 2)
    y_min = max(0, int(min(ys)) - 2)
    x_max = min(img_w, int(max(xs)) + 2)
    y_max = min(img_h, int(max(ys)) + 2)
    
    # Ensure valid crop dimensions
    if x_max <= x_min or y_max <= y_min:
        return None
    
    # Minimum size check (too small crops are useless)
    if (x_max - x_min) < 5 or (y_max - y_min) < 5:
        return None
    
    cropped = img.crop((x_min, y_min, x_max, y_max))
    return cropped


def find_image_for_gt(gt_basename, img_dir):
    """Find the matching image file for a GT file, handling .jpg/.jpeg/.JPG variants."""
    # The GT file basename matches the image basename (without extension)
    for ext in ['.jpg', '.jpeg', '.JPG', '.JPEG', '.png', '.PNG']:
        img_path = os.path.join(img_dir, gt_basename + ext)
        if os.path.exists(img_path):
            return img_path
    return None


def main():
    configure_console()
    
    print("=" * 60)
    print("  Cropping Bengali Words from Ritu1 Dataset")
    print("=" * 60)
    
    os.makedirs(OUT_IMG_DIR, exist_ok=True)
    os.makedirs(OUT_GT_DIR, exist_ok=True)
    
    # Get all GT files
    gt_files = sorted(glob.glob(os.path.join(RITU1_GT_DIR, '*.txt')))
    print(f"Found {len(gt_files)} GT files in {RITU1_GT_DIR}")
    
    # Find starting index
    existing_count = len([f for f in os.listdir(OUT_IMG_DIR) if f.endswith('.jpg')])
    print(f"Existing images in Bengali/: {existing_count}")
    
    crop_idx = 0
    total_crops = 0
    skipped_no_image = 0
    skipped_no_regions = 0
    skipped_crop_fail = 0
    processed_images = 0
    
    for gt_file in gt_files:
        gt_basename = os.path.splitext(os.path.basename(gt_file))[0]
        
        # Skip duplicate files like "224 (1).txt"
        if '(' in gt_basename:
            continue
        
        # Find matching image
        img_path = find_image_for_gt(gt_basename, RITU1_IMG_DIR)
        if not img_path:
            skipped_no_image += 1
            continue
        
        # Parse GT
        regions = parse_ritu1_gt(gt_file)
        if not regions:
            skipped_no_regions += 1
            continue
        
        # Open image
        try:
            img = Image.open(img_path).convert('RGB')
        except Exception as e:
            print(f"  ERROR opening {img_path}: {e}")
            continue
        
        processed_images += 1
        
        # Crop each region
        for region in regions:
            cropped = crop_polygon_region(img, region['points'])
            if cropped is None:
                skipped_crop_fail += 1
                continue
            
            # Clean transcript for file saving
            transcript = region['transcript'].strip()
            
            # Generate unique filename
            crop_name = f"ritu1_{gt_basename}_{crop_idx}"
            out_img_path = os.path.join(OUT_IMG_DIR, crop_name + '.jpg')
            out_gt_path = os.path.join(OUT_GT_DIR, crop_name + '.txt')
            
            # Save cropped image
            cropped.save(out_img_path, 'JPEG', quality=95)
            
            # Save GT text
            with open(out_gt_path, 'w', encoding='utf-8') as f:
                f.write(transcript)
            
            crop_idx += 1
            total_crops += 1
        
        if processed_images % 50 == 0:
            print(f"  Processed {processed_images} images, {total_crops} crops so far...")
    
    print(f"\n{'=' * 60}")
    print(f"  RESULTS:")
    print(f"  GT files found:          {len(gt_files)}")
    print(f"  Images processed:        {processed_images}")
    print(f"  Skipped (no image):      {skipped_no_image}")
    print(f"  Skipped (no Bengali):    {skipped_no_regions}")
    print(f"  Skipped (crop fail):     {skipped_crop_fail}")
    print(f"  Total words cropped:     {total_crops}")
    print(f"  New Bengali/ count:      {existing_count + total_crops}")
    print(f"{'=' * 60}")


if __name__ == '__main__':
    main()
