"""
Generate .lstmf files for Tesseract fine-tuning from prepared ground-truth pairs.

Expected input directory contents:
- <stem>.png
- <stem>.gt.txt
- train_stems.txt
- eval_stems.txt
"""
import argparse
import subprocess
import sys
from pathlib import Path


def configure_console():
    try:
        sys.stdout.reconfigure(encoding="utf-8", errors="replace")
        sys.stderr.reconfigure(encoding="utf-8", errors="replace")
    except Exception:
        pass


def read_stems(path: Path):
    if not path.exists():
        return []
    return [line.strip() for line in path.read_text(encoding="utf-8").splitlines() if line.strip()]


def run_tesseract(image_path: Path, out_base: Path, lang: str, tessdata_dir: Path, psm: int):
    cmd = [
        "tesseract",
        str(image_path),
        str(out_base),
        "--psm",
        str(psm),
        "--oem",
        "1",
        "-l",
        lang,
        "--tessdata-dir",
        str(tessdata_dir),
        "lstm.train",
    ]
    result = subprocess.run(cmd, capture_output=True, text=True, encoding="utf-8", errors="replace")
    return result.returncode, result.stderr.strip()


def main():
    configure_console()

    parser = argparse.ArgumentParser()
    parser.add_argument("--ground-truth-dir", default="tesseract_ground_truth")
    parser.add_argument("--tessdata-dir", default="tessdata")
    parser.add_argument("--lang", default="ben")
    parser.add_argument("--psm", type=int, default=7)
    args = parser.parse_args()

    gt_dir = Path(args.ground_truth_dir)
    tessdata_dir = Path(args.tessdata_dir)

    if not gt_dir.exists():
        raise FileNotFoundError(f"Ground truth directory not found: {gt_dir}")
    if not (tessdata_dir / f"{args.lang}.traineddata").exists():
        raise FileNotFoundError(f"Missing starter model: {tessdata_dir / (args.lang + '.traineddata')}")

    stems = sorted(p.stem for p in gt_dir.glob("*.png"))
    failures = []

    for index, stem in enumerate(stems, start=1):
        image_path = gt_dir / f"{stem}.png"
        gt_path = gt_dir / f"{stem}.gt.txt"
        out_base = gt_dir / stem
        lstmf_path = gt_dir / f"{stem}.lstmf"

        if not gt_path.exists():
            failures.append((stem, "missing gt"))
            continue

        if not lstmf_path.exists():
            code, stderr = run_tesseract(image_path, out_base, args.lang, tessdata_dir, args.psm)
            if code != 0 or not lstmf_path.exists():
                failures.append((stem, stderr or "tesseract failed"))

        if index % 250 == 0 or index == len(stems):
            print(f"Processed {index}/{len(stems)}")

    train_stems = read_stems(gt_dir / "train_stems.txt")
    eval_stems = read_stems(gt_dir / "eval_stems.txt")

    all_lstmf = [str((gt_dir / f"{stem}.lstmf").resolve()) for stem in stems if (gt_dir / f"{stem}.lstmf").exists()]
    train_lstmf = [str((gt_dir / f"{stem}.lstmf").resolve()) for stem in train_stems if (gt_dir / f"{stem}.lstmf").exists()]
    eval_lstmf = [str((gt_dir / f"{stem}.lstmf").resolve()) for stem in eval_stems if (gt_dir / f"{stem}.lstmf").exists()]

    (gt_dir / "all_lstmf.txt").write_text("\n".join(all_lstmf) + "\n", encoding="utf-8")
    (gt_dir / "train_lstmf.txt").write_text("\n".join(train_lstmf) + "\n", encoding="utf-8")
    (gt_dir / "eval_lstmf.txt").write_text("\n".join(eval_lstmf) + "\n", encoding="utf-8")

    print(f"Generated/verified {len(all_lstmf)} lstmf files")
    print(f"Train list: {len(train_lstmf)} | Eval list: {len(eval_lstmf)}")

    if failures:
        failure_file = gt_dir / "lstmf_failures.txt"
        failure_file.write_text(
            "\n".join(f"{stem}\t{reason}" for stem, reason in failures),
            encoding="utf-8",
        )
        print(f"Failures: {len(failures)} | Details: {failure_file}")


if __name__ == "__main__":
    main()
