"""
Prepare Bengali/Bengali_gt pairs for Tesseract LSTM fine-tuning.

Creates:
- <output>/<stem>.png
- <output>/<stem>.gt.txt
- <output>/manifest.json
- <output>/all_stems.txt
- <output>/train_stems.txt
- <output>/eval_stems.txt
- <output>/charset.txt
"""
import argparse
import json
import random
import sys
import unicodedata
from collections import Counter
from pathlib import Path

from PIL import Image


def configure_console():
    try:
        sys.stdout.reconfigure(encoding="utf-8", errors="replace")
        sys.stderr.reconfigure(encoding="utf-8", errors="replace")
    except Exception:
        pass


def normalize_text(text: str) -> str:
    text = unicodedata.normalize("NFC", text.strip())
    text = text.replace("\u200c", "").replace("\u200d", "")
    text = text.replace("\u200b", "").replace("\ufeff", "")
    return " ".join(text.split())


def load_pairs(image_dir: Path, gt_dir: Path):
    images = {p.stem: p for p in image_dir.glob("*.jpg")}
    images.update({p.stem: p for p in image_dir.glob("*.png")})
    texts = {p.stem: p for p in gt_dir.glob("*.txt")}

    pairs = []
    skipped = 0

    for stem in sorted(images):
        if stem not in texts:
            continue

        gt_text = normalize_text(texts[stem].read_text(encoding="utf-8"))
        if not gt_text or gt_text == "###":
            skipped += 1
            continue

        pairs.append(
            {
                "stem": stem,
                "image": str(images[stem]),
                "gt": gt_text,
            }
        )

    return pairs, skipped


def export_pairs(pairs, output_dir: Path):
    output_dir.mkdir(parents=True, exist_ok=True)

    char_counter = Counter()
    for pair in pairs:
        img = Image.open(pair["image"])
        if img.mode != "L":
            img = img.convert("L")

        png_path = output_dir / f"{pair['stem']}.png"
        gt_path = output_dir / f"{pair['stem']}.gt.txt"

        img.save(png_path)
        gt_path.write_text(pair["gt"], encoding="utf-8")

        char_counter.update(pair["gt"])

    return char_counter


def write_split_files(pairs, output_dir: Path, eval_ratio: float, seed: int):
    stems = [pair["stem"] for pair in pairs]
    rng = random.Random(seed)
    rng.shuffle(stems)

    eval_count = max(1, int(len(stems) * eval_ratio))
    eval_stems = sorted(stems[:eval_count])
    train_stems = sorted(stems[eval_count:])

    (output_dir / "all_stems.txt").write_text("\n".join(sorted(stems)) + "\n", encoding="utf-8")
    (output_dir / "train_stems.txt").write_text("\n".join(train_stems) + "\n", encoding="utf-8")
    (output_dir / "eval_stems.txt").write_text("\n".join(eval_stems) + "\n", encoding="utf-8")

    return train_stems, eval_stems


def main():
    configure_console()

    parser = argparse.ArgumentParser()
    parser.add_argument("--images", default="Bengali")
    parser.add_argument("--gt", default="Bengali_gt")
    parser.add_argument("--output", default="tesseract_ground_truth")
    parser.add_argument("--eval-ratio", type=float, default=0.10)
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()

    image_dir = Path(args.images)
    gt_dir = Path(args.gt)
    output_dir = Path(args.output)

    if not image_dir.exists() or not gt_dir.exists():
        raise FileNotFoundError("Input image or ground-truth directory not found.")

    pairs, skipped = load_pairs(image_dir, gt_dir)
    if not pairs:
        raise RuntimeError("No valid image/ground-truth pairs found.")

    char_counter = export_pairs(pairs, output_dir)
    train_stems, eval_stems = write_split_files(pairs, output_dir, args.eval_ratio, args.seed)

    charset = "".join(sorted(char_counter))
    (output_dir / "charset.txt").write_text(charset, encoding="utf-8")

    manifest = {
        "images_dir": str(image_dir),
        "gt_dir": str(gt_dir),
        "output_dir": str(output_dir),
        "total_pairs": len(pairs),
        "skipped_empty_or_hash": skipped,
        "train_count": len(train_stems),
        "eval_count": len(eval_stems),
        "eval_ratio": args.eval_ratio,
        "seed": args.seed,
        "charset_size": len(char_counter),
    }
    (output_dir / "manifest.json").write_text(
        json.dumps(manifest, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )

    print(f"Prepared {len(pairs)} pairs in {output_dir}")
    print(f"Train: {len(train_stems)} | Eval: {len(eval_stems)}")
    print(f"Charset size: {len(char_counter)}")


if __name__ == "__main__":
    main()
