"""
Synthetic Bengali Text Image Generator
- Generates synthetic word images using Bengali Unicode text
- Uses multiple fonts, sizes, colors, backgrounds
- Augments the real dataset for better CRNN training
"""
import os
import sys
import json
import random
import string
import numpy as np
from PIL import Image, ImageDraw, ImageFont, ImageFilter, ImageEnhance

# ============================================================
#  Bengali text corpus for synthesis
# ============================================================

# Common Bengali words (from real dataset + extra common words)
BENGALI_WORDS = [
    # Common scene text words
    "দোকান", "হোটেল", "রেস্তোরাঁ", "ফার্মেসি", "ব্যাংক", "স্কুল", "কলেজ",
    "হাসপাতাল", "পুলিশ", "ডাক্তার", "ইঞ্জিনিয়ার", "উকিল", "শিক্ষক",
    "বাজার", "মার্কেট", "সুপার", "শপ", "স্টোর", "সেন্টার", "কমপ্লেক্স",
    "রোড", "লেন", "স্ট্রীট", "পথ", "গলি", "মোড়", "চৌরাস্তা",
    "নগর", "পুর", "গ্রাম", "পাড়া", "মহল্লা", "তলা", "ভবন",
    "বিয়ে", "বাড়ী", "বাড়ি", "ঘর", "মন্দির", "মসজিদ", "গীর্জা",
    "সেগুন", "মেলা", "উৎসব", "পূজা", "অনুষ্ঠান", "সভা", "সমিতি",
    "ফার্ণিচার", "ইলেকট্রনিক্স", "মোবাইল", "কম্পিউটার", "প্রিন্টার",
    "কাপড়", "শাড়ী", "শাড়ি", "পাঞ্জাবী", "লুঙ্গি", "জামা", "পোশাক",
    "ওষুধ", "চিকিৎসা", "পরীক্ষা", "রিপোর্ট", "প্রেসক্রিপশন",
    "খাবার", "রান্না", "মিষ্টি", "চা", "কফি", "জল", "পানি",
    "বই", "খাতা", "কলম", "পেন্সিল", "রাবার", "স্কেল",
    "টাকা", "পয়সা", "দাম", "মূল্য", "ছাড়", "অফার", "বিক্রি",
    "নতুন", "পুরাতন", "ভালো", "খারাপ", "বড়", "ছোট", "সুন্দর",
    "লাল", "নীল", "সবুজ", "হলুদ", "সাদা", "কালো", "গোলাপী",
    "এক", "দুই", "তিন", "চার", "পাঁচ", "ছয়", "সাত", "আট", "নয়", "দশ",
    "প্রথম", "দ্বিতীয়", "তৃতীয়", "চতুর্থ", "পঞ্চম",
    "সোমবার", "মঙ্গলবার", "বুধবার", "বৃহস্পতিবার", "শুক্রবার", "শনিবার", "রবিবার",
    "জানুয়ারি", "ফেব্রুয়ারি", "মার্চ", "এপ্রিল", "মে", "জুন",
    "জুলাই", "আগস্ট", "সেপ্টেম্বর", "অক্টোবর", "নভেম্বর", "ডিসেম্বর",
    "বাংলাদেশ", "ভারত", "কলকাতা", "ঢাকা", "চট্টগ্রাম", "রাজশাহী",
    "সরকার", "সরকারি", "বেসরকারি", "জাতীয়", "আন্তর্জাতিক",
    "বিশ্ববিদ্যালয়", "মহাবিদ্যালয়", "বিদ্যালয়", "প্রাথমিক", "মাধ্যমিক",
    "ডেকরেটর", "ক্যাটারিং", "ফটোগ্রাফি", "ভিডিও", "মিউজিক",
    "ট্রেন", "বাস", "রিকশা", "অটো", "ট্যাক্সি", "স্টেশন",
    "নদী", "পুকুর", "সাগর", "পাহাড়", "বন", "মাঠ", "উদ্যান",
    "রবীন্দ্রনাথ", "নজরুল", "বঙ্কিম", "শরৎচন্দ্র", "সুকান্ত",
    "সাম্প্রদায়িক", "সম্প্রীতি", "ঐক্য", "শান্তি", "প্রগতি",
    "স্বাস্থ্য", "শিক্ষা", "কৃষি", "শিল্প", "বাণিজ্য",
    "মৎস্য", "পশু", "পাখি", "ফুল", "ফল", "শাকসবজি",
    "জুয়েলার্স", "গোল্ড", "সিলভার", "ডায়মন্ড", "প্ল্যাটিনাম",
    "কালেকশন", "ড্রেস", "বেডসীট", "পর্দা", "গালিচা",
    "অঞ্জনা", "ফ্যান্সী", "চুড়িদার", "পিস", "সেট",
    "পৌরসভা", "নির্মিত", "স্মারক", "ভিত্তিপ্রস্তর",
    "টেলার্স", "জেন্টস", "লেডিস", "কিডস", "ফ্যামিলি",
    "প্লাজা", "আর্কেড", "গ্যালারি", "শোরুম", "ওয়ারহাউস",
    "ক্লিনিক", "ল্যাব", "ডায়াগনস্টিক", "প্যাথলজি",
    "ইন্টারনেট", "ওয়াইফাই", "ব্রডব্যান্ড", "কেবল",
    "বিদ্যুৎ", "গ্যাস", "পানি", "টেলিফোন", "ডাক",
    "নির্বাচন", "ভোট", "প্রার্থী", "দল", "সংসদ",
    "আদালত", "থানা", "কারাগার", "সেনানিবাস",
    "বিমানবন্দর", "নৌবন্দর", "বাসস্ট্যান্ড", "রেলস্টেশন",
    "পার্ক", "মিউজিয়াম", "লাইব্রেরি", "সিনেমা", "থিয়েটার",
    "ব্রিজ", "ফ্লাইওভার", "আন্ডারপাস", "টানেল",
    "মঙ্গলদীপ", "শারদোৎসব", "নবরাত্রি", "দীপাবলি",
    "ঈদ", "বড়দিন", "হোলি", "পৌষসংক্রান্তি",
    "বিশ্বকবি", "ছাত্রসাথী", "উপহার", "লিখন", "সামগ্রী",
    "ফিডার", "বিশাল", "সুভাষ", "নিবেদন", "জংশন",
    "উপলক্ষে", "বিউটি", "স্পা", "সেলুন", "পার্লার",
    "রক্ষায়", "হোক", "সরঞ্জাম", "ভাড়া", "দেওয়া",
    "বিভিন্ন", "অনুষ্ঠানে", "যোগাযোগ", "ঠিকানা", "নম্বর",
]

# Bengali digits
BENGALI_DIGITS = "০১২৩৪৫৬৭৮৯"

def generate_digit_string():
    """Generate a random Bengali digit string (phone numbers, prices, etc.)."""
    length = random.randint(2, 10)
    if random.random() < 0.3:
        # Phone number format
        digits = ''.join(random.choice(BENGALI_DIGITS) for _ in range(4))
        digits2 = ''.join(random.choice(BENGALI_DIGITS) for _ in range(4))
        return f"{digits}-{digits2}"
    elif random.random() < 0.3:
        # Price format
        digits = ''.join(random.choice(BENGALI_DIGITS) for _ in range(random.randint(2, 5)))
        return f"৳{digits}"
    else:
        return ''.join(random.choice(BENGALI_DIGITS) for _ in range(length))


def get_random_text():
    """Get random Bengali text for synthesis."""
    r = random.random()
    if r < 0.7:
        # Single word
        return random.choice(BENGALI_WORDS)
    elif r < 0.85:
        # Two words
        return ' '.join(random.choices(BENGALI_WORDS, k=2))
    elif r < 0.95:
        # Digit string
        return generate_digit_string()
    else:
        # Word + punctuation
        word = random.choice(BENGALI_WORDS)
        punct = random.choice([',', '.', '।', '-', '!', '?'])
        return word + punct


def find_bengali_fonts():
    """Find available Bengali-compatible fonts."""
    font_paths = []

    # Common font locations
    search_dirs = [
        '/usr/share/fonts',
        '/usr/local/share/fonts',
        '/content/fonts',
        'fonts',
        'C:\\Windows\\Fonts',
    ]

    for d in search_dirs:
        if os.path.isdir(d):
            for root, dirs, files in os.walk(d):
                for f in files:
                    if f.lower().endswith(('.ttf', '.otf')):
                        font_paths.append(os.path.join(root, f))

    return font_paths


def download_bengali_fonts(font_dir='fonts'):
    """Download Bengali fonts if not available."""
    os.makedirs(font_dir, exist_ok=True)

    # Try to use system fonts first
    fonts = find_bengali_fonts()

    # Filter for fonts that can render Bengali
    bengali_fonts = []
    test_text = "বাংলা"

    for fp in fonts:
        try:
            font = ImageFont.truetype(fp, 24)
            # Quick test: try to get the bounding box
            bbox = font.getbbox(test_text)
            if bbox and (bbox[2] - bbox[0]) > 5:  # Has some width
                bengali_fonts.append(fp)
        except Exception:
            continue

    if not bengali_fonts:
        # Download Noto Sans Bengali from Google
        print("  Downloading Bengali fonts...")
        try:
            import urllib.request
            urls = [
                ("https://github.com/google/fonts/raw/main/ofl/notosansbengali/NotoSansBengali%5Bwdth%2Cwght%5D.ttf",
                 "NotoSansBengali.ttf"),
                ("https://github.com/google/fonts/raw/main/ofl/notoserifbengali/NotoSerifBengali%5Bwdth%2Cwght%5D.ttf",
                 "NotoSerifBengali.ttf"),
            ]
            for url, fname in urls:
                out_path = os.path.join(font_dir, fname)
                if not os.path.exists(out_path):
                    try:
                        urllib.request.urlretrieve(url, out_path)
                        bengali_fonts.append(out_path)
                        print(f"    Downloaded: {fname}")
                    except Exception as e:
                        print(f"    Failed to download {fname}: {e}")
        except Exception as e:
            print(f"  Font download failed: {e}")

    # Fallback: try to install via apt
    if not bengali_fonts:
        print("  Installing Bengali fonts via apt...")
        os.system("apt-get install -y fonts-bengali-extra fonts-lohit-beng-bengali >/dev/null 2>&1")
        fonts = find_bengali_fonts()
        for fp in fonts:
            try:
                font = ImageFont.truetype(fp, 24)
                bbox = font.getbbox(test_text)
                if bbox and (bbox[2] - bbox[0]) > 5:
                    bengali_fonts.append(fp)
            except Exception:
                continue

    if bengali_fonts:
        print(f"  Found {len(bengali_fonts)} Bengali-compatible fonts")
    else:
        print("  WARNING: No Bengali fonts found! Using default font.")
        bengali_fonts = [None]  # Will use default

    return bengali_fonts


def render_text_image(text, font_path=None, font_size=None, img_height=48):
    """Render Bengali text as an image with random styling."""
    if font_size is None:
        font_size = random.randint(20, 40)

    # Load font
    try:
        if font_path:
            font = ImageFont.truetype(font_path, font_size)
        else:
            font = ImageFont.load_default()
    except Exception:
        font = ImageFont.load_default()

    # Get text size
    dummy_img = Image.new('RGB', (1, 1))
    dummy_draw = ImageDraw.Draw(dummy_img)
    bbox = dummy_draw.textbbox((0, 0), text, font=font)
    text_w = bbox[2] - bbox[0] + 20
    text_h = bbox[3] - bbox[1] + 16

    # Create image with random background
    bg_style = random.random()
    if bg_style < 0.4:
        # Light background (like signs)
        bg_color = tuple(random.randint(200, 255) for _ in range(3))
        text_color = tuple(random.randint(0, 80) for _ in range(3))
    elif bg_style < 0.7:
        # Dark background (like billboards)
        bg_color = tuple(random.randint(0, 60) for _ in range(3))
        text_color = tuple(random.randint(180, 255) for _ in range(3))
    elif bg_style < 0.85:
        # Colored background
        bg_color = (random.randint(100, 200), random.randint(50, 150), random.randint(50, 200))
        text_color = (255, 255, 255) if sum(bg_color) < 400 else (0, 0, 0)
    else:
        # Pure white/black
        if random.random() < 0.5:
            bg_color = (255, 255, 255)
            text_color = (0, 0, 0)
        else:
            bg_color = (0, 0, 0)
            text_color = (255, 255, 255)

    img = Image.new('RGB', (text_w, text_h), bg_color)
    draw = ImageDraw.Draw(img)

    # Draw text with slight offset
    x_offset = random.randint(4, 10)
    y_offset = random.randint(2, 8)
    draw.text((x_offset, y_offset), text, fill=text_color, font=font)

    # Apply random degradation (to simulate real scene conditions)
    # Blur
    if random.random() < 0.3:
        radius = random.uniform(0.3, 1.2)
        img = img.filter(ImageFilter.GaussianBlur(radius=radius))

    # Noise
    if random.random() < 0.3:
        arr = np.array(img, dtype=np.float32)
        noise = np.random.normal(0, random.uniform(3, 12), arr.shape)
        arr = np.clip(arr + noise, 0, 255).astype(np.uint8)
        img = Image.fromarray(arr)

    # Contrast variation
    if random.random() < 0.3:
        factor = random.uniform(0.6, 1.5)
        img = ImageEnhance.Contrast(img).enhance(factor)

    # Slight rotation
    if random.random() < 0.2:
        angle = random.uniform(-5, 5)
        img = img.rotate(angle, fillcolor=bg_color, expand=True)

    # Perspective-like distortion (horizontal shear)
    if random.random() < 0.15:
        w, h = img.size
        shear = random.uniform(-0.1, 0.1)
        data = (1, shear, -shear * h / 2, 0, 1, 0)
        img = img.transform((w, h), Image.AFFINE, data, fillcolor=bg_color)

    return img


def generate_synthetic_dataset(num_samples=10000, output_dir='synthetic_bengali',
                                gt_dir='synthetic_bengali_gt'):
    """Generate synthetic Bengali text images and ground truth."""
    os.makedirs(output_dir, exist_ok=True)
    os.makedirs(gt_dir, exist_ok=True)

    print("=" * 60)
    print("  Generating Synthetic Bengali Text Dataset")
    print("=" * 60)

    # Get fonts
    fonts = download_bengali_fonts()

    generated = 0
    failed = 0

    for i in range(num_samples):
        text = get_random_text()
        font_path = random.choice(fonts) if fonts[0] is not None else None

        try:
            img = render_text_image(text, font_path)

            # Save
            img_name = f"syn_{i:06d}.jpg"
            gt_name = f"syn_{i:06d}.txt"

            img.save(os.path.join(output_dir, img_name), 'JPEG', quality=90)
            with open(os.path.join(gt_dir, gt_name), 'w', encoding='utf-8') as f:
                f.write(text)

            generated += 1

        except Exception as e:
            failed += 1
            if failed <= 5:
                print(f"  Warning: Failed to generate sample {i}: {e}")

        if (i + 1) % 2000 == 0:
            print(f"  Progress: {i+1}/{num_samples} ({generated} generated, {failed} failed)")

    print(f"\n  Done! Generated {generated} synthetic samples in {output_dir}/")
    print(f"  Failed: {failed}")
    return generated


if __name__ == '__main__':
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument('--num', type=int, default=10000, help='Number of synthetic samples')
    parser.add_argument('--output', default='synthetic_bengali', help='Output image directory')
    parser.add_argument('--gt', default='synthetic_bengali_gt', help='Output GT directory')
    args = parser.parse_args()

    generate_synthetic_dataset(args.num, args.output, args.gt)
