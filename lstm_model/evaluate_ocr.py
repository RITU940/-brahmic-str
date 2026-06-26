"""
Bengali OCR Final Evaluation - Optimized for Speed & Accuracy
- Excludes ### (unintelligible) GT entries
- Preprocesses ALL images with upscaling + contrast for better OCR
- Uses multiprocessing-like batch approach via pre-processing all images
- Generates processed_words.txt as proof
"""
import os
import subprocess
import sys
import unicodedata
from collections import defaultdict
import time
from PIL import Image, ImageEnhance, ImageFilter


def configure_console():
    try:
        sys.stdout.reconfigure(encoding='utf-8', errors='replace')
        sys.stderr.reconfigure(encoding='utf-8', errors='replace')
    except Exception:
        pass

# --- Configuration ---
BASE_DIR = os.path.abspath(os.environ.get("LSTM_MODEL_BASE", os.getcwd()))
BENGALI_IMG_DIR = os.environ.get("BENGALI_IMG_DIR", os.path.join(BASE_DIR, "Bengali"))
BENGALI_GT_DIR = os.environ.get("BENGALI_GT_DIR", os.path.join(BASE_DIR, "Bengali_gt"))
TESSDATA_DIR = os.environ.get("TESSDATA_DIR", os.path.join(BASE_DIR, "tessdata"))
PREPROCESSED_DIR = os.environ.get("PREPROCESSED_DIR", os.path.join(BASE_DIR, "Bengali_preprocessed"))
REPORT_FILE = os.environ.get("REPORT_FILE", os.path.join(BASE_DIR, "evaluation_report.txt"))
PROCESSED_WORDS_FILE = os.environ.get("PROCESSED_WORDS_FILE", os.path.join(BASE_DIR, "processed_words.txt"))

MODELS = {"ben_finetuned": "Fine-tuned Model", "ben": "Default Model"}

def normalize_text(text):
    text = unicodedata.normalize('NFC', text.strip())
    text = text.replace('\u200c', '').replace('\u200d', '').replace('\u200b', '').replace('\ufeff', '')
    return ' '.join(text.split())

def edit_distance(ref, hyp):
    m, n = len(ref), len(hyp)
    if m == 0: return n
    if n == 0: return m
    prev = list(range(n + 1))
    for i in range(1, m + 1):
        curr = [i] + [0] * n
        for j in range(1, n + 1):
            curr[j] = min(prev[j]+1, curr[j-1]+1, prev[j-1]+(0 if ref[i-1]==hyp[j-1] else 1))
        prev = curr
    return prev[n]

def preprocess_image(img_path, out_path):
    """Preprocess image: upscale + sharpen + contrast boost for better OCR."""
    try:
        img = Image.open(img_path)
        w, h = img.size
        
        # Always upscale to ensure minimum height of 64px
        target_h = max(64, h)
        scale = target_h / h
        if scale > 1:
            new_w, new_h = int(w * scale), int(h * scale)
            img = img.resize((new_w, new_h), Image.LANCZOS)
        
        # Convert to grayscale
        if img.mode != 'L':
            img = img.convert('L')
        
        # Sharpen
        img = img.filter(ImageFilter.SHARPEN)
        
        # Boost contrast
        enhancer = ImageEnhance.Contrast(img)
        img = enhancer.enhance(1.5)
        
        img.save(out_path)
        return True
    except:
        return False

def run_tesseract(image_path, lang, psm=7):
    try:
        result = subprocess.run(
            ["tesseract", image_path, "stdout", "-l", lang,
             "--tessdata-dir", TESSDATA_DIR, "--psm", str(psm)],
            capture_output=True, text=True, timeout=30,
            encoding='utf-8', errors='replace')
        return result.stdout.strip()
    except:
        return ""

def get_valid_pairs():
    images = {f.replace('.jpg', ''): f for f in os.listdir(BENGALI_IMG_DIR) if f.endswith('.jpg')}
    gts = {f.replace('.txt', ''): f for f in os.listdir(BENGALI_GT_DIR) if f.endswith('.txt')}
    
    pairs = []
    skipped_hash = 0
    
    for name in sorted(images.keys()):
        if name not in gts:
            continue
        gt_path = os.path.join(BENGALI_GT_DIR, gts[name])
        gt_text = open(gt_path, 'r', encoding='utf-8').read().strip()
        if gt_text == '###' or not gt_text:
            skipped_hash += 1
            continue
        pairs.append((os.path.join(BENGALI_IMG_DIR, images[name]), gt_path, name, gt_text))
    
    print(f"Total images: {len(images)}")
    print(f"Excluded (### / empty): {skipped_hash}")
    print(f"Valid pairs: {len(pairs)}")
    return pairs

def preprocess_all(pairs):
    """Preprocess all images upfront for speed."""
    os.makedirs(PREPROCESSED_DIR, exist_ok=True)
    print(f"\nPreprocessing {len(pairs)} images...")
    start = time.time()
    for i, (img_path, _, name, _) in enumerate(pairs):
        out_path = os.path.join(PREPROCESSED_DIR, name + '.png')
        if not os.path.exists(out_path):
            preprocess_image(img_path, out_path)
        if (i+1) % 1000 == 0:
            print(f"  Preprocessed {i+1}/{len(pairs)}")
    print(f"  Done in {time.time()-start:.1f}s")

def evaluate_model(pairs, lang, model_name):
    """Evaluate model: try preprocessed image first, fallback to original + PSM 8."""
    total_cer_num = 0; total_gt_chars = 0
    total_wer_num = 0; total_gt_words = 0
    total_correct_words = 0
    processed = 0; empty_ocr = 0; exact_match = 0
    results = []
    total = len(pairs)
    start_time = time.time()
    
    print(f"\n{'='*60}")
    print(f"  Evaluating: {model_name} (lang={lang})")
    print(f"  Valid pairs: {total}")
    print(f"{'='*60}")
    
    for i, (img_path, gt_path, name, gt_raw) in enumerate(pairs):
        gt_norm = normalize_text(gt_raw)
        if not gt_norm:
            continue
        
        # Try preprocessed image first (PSM 7)
        pp_path = os.path.join(PREPROCESSED_DIR, name + '.png')
        if os.path.exists(pp_path):
            ocr_text = run_tesseract(pp_path, lang, 7)
            # If empty, try PSM 8 on preprocessed
            if not ocr_text:
                ocr_text = run_tesseract(pp_path, lang, 8)
        else:
            ocr_text = run_tesseract(img_path, lang, 7)
        
        # Last resort: PSM 8 on original
        if not ocr_text:
            ocr_text = run_tesseract(img_path, lang, 8)
        
        ocr_norm = normalize_text(ocr_text) if ocr_text else ""
        
        gt_chars = list(gt_norm)
        ocr_chars = list(ocr_norm)
        char_ed = edit_distance(gt_chars, ocr_chars)
        total_cer_num += char_ed
        total_gt_chars += len(gt_chars)
        
        gt_words = gt_norm.split()
        ocr_words = ocr_norm.split()
        word_ed = edit_distance(gt_words, ocr_words)
        total_wer_num += word_ed
        total_gt_words += len(gt_words)
        
        correct = sum(1 for gw, ow in zip(gt_words, ocr_words) if gw == ow)
        total_correct_words += correct
        
        if not ocr_norm: empty_ocr += 1
        if gt_norm == ocr_norm: exact_match += 1
        
        processed += 1
        cer_val = char_ed / max(len(gt_chars), 1)
        results.append({
            'name': name, 'gt': gt_norm, 'ocr': ocr_norm,
            'cer': cer_val, 'gt_words': len(gt_words), 'correct': correct
        })
        
        if (i+1) % 500 == 0 or (i+1) == total:
            elapsed = time.time() - start_time
            cer = total_cer_num / max(total_gt_chars, 1)
            rate = (i+1) / max(elapsed, 0.1)
            eta = (total - i - 1) / max(rate, 0.1)
            print(f"  [{i+1}/{total}] CER={cer:.4f} CharAcc={(1-cer)*100:.1f}% "
                  f"Exact={exact_match} Empty={empty_ocr} "
                  f"({rate:.1f}img/s ETA:{eta:.0f}s)")
    
    cer = total_cer_num / max(total_gt_chars, 1)
    wer = total_wer_num / max(total_gt_words, 1)
    simple_wer = 1 - total_correct_words / max(total_gt_words, 1)
    elapsed = time.time() - start_time
    
    # Non-empty stats
    ne_results = [r for r in results if r['ocr']]
    ne_cer_num = sum(edit_distance(list(r['gt']), list(r['ocr'])) for r in ne_results)
    ne_gt_chars = sum(len(r['gt']) for r in ne_results)
    ne_cer = ne_cer_num / max(ne_gt_chars, 1)
    ne_correct = sum(r['correct'] for r in ne_results)
    ne_gt_words = sum(r['gt_words'] for r in ne_results)
    ne_exact = sum(1 for r in ne_results if r['gt'] == r['ocr'])
    
    print(f"\n--- {model_name} FINAL RESULTS ---")
    print(f"  Processed: {processed} | Empty OCR: {empty_ocr} ({empty_ocr/max(processed,1)*100:.1f}%)")
    print(f"  === OVERALL (all {processed} images) ===")
    print(f"  GT Words: {total_gt_words} | Correct: {total_correct_words}")
    print(f"  CER: {cer:.4f} | Char Accuracy: {(1-cer)*100:.2f}%")
    print(f"  Simplified WER: {simple_wer:.4f} | Word Accuracy: {(1-simple_wer)*100:.2f}%")
    print(f"  Exact Match: {exact_match}/{processed} ({exact_match/max(processed,1)*100:.1f}%)")
    print(f"  === NON-EMPTY OCR ({len(ne_results)} images) ===")
    print(f"  CER: {ne_cer:.4f} | Char Accuracy: {(1-ne_cer)*100:.2f}%")
    print(f"  Word Accuracy: {ne_correct/max(ne_gt_words,1)*100:.2f}%")
    print(f"  Exact Match: {ne_exact}/{len(ne_results)} ({ne_exact/max(len(ne_results),1)*100:.1f}%)")
    print(f"  Time: {elapsed:.1f}s")
    
    return {
        'model_name': model_name, 'lang': lang,
        'processed': processed, 'empty_ocr': empty_ocr,
        'total_gt_chars': total_gt_chars, 'total_gt_words': total_gt_words,
        'total_correct_words': total_correct_words,
        'cer': cer, 'wer': wer, 'simple_wer': simple_wer,
        'char_acc': (1-cer)*100, 'word_acc': (1-wer)*100,
        'simple_acc': (1-simple_wer)*100,
        'exact_match': exact_match, 'exact_pct': exact_match/max(processed,1)*100,
        'ne_count': len(ne_results), 'ne_cer': ne_cer,
        'ne_char_acc': (1-ne_cer)*100,
        'ne_correct': ne_correct, 'ne_gt_words': ne_gt_words,
        'ne_word_acc': ne_correct/max(ne_gt_words,1)*100,
        'ne_exact': ne_exact, 'ne_exact_pct': ne_exact/max(len(ne_results),1)*100,
        'results': results
    }

def generate_processed_words(ft_result):
    """Generate processed_words.txt with every image's GT vs OCR."""
    results = ft_result['results']
    with open(PROCESSED_WORDS_FILE, 'w', encoding='utf-8') as f:
        f.write("=" * 90 + "\n")
        f.write("  PROCESSED WORDS - Bengali OCR Evaluation\n")
        f.write("  Model: ben_finetuned.traineddata (Fine-tuned Tesseract LSTM)\n")
        f.write(f"  Total images processed: {ft_result['processed']}\n")
        f.write(f"  Overall Character Accuracy: {ft_result['char_acc']:.2f}%\n")
        f.write(f"  On recognized images - Char Accuracy: {ft_result['ne_char_acc']:.2f}%\n")
        f.write(f"  Simplified Word Error Rate: {ft_result['simple_wer']:.4f}\n")
        f.write("=" * 90 + "\n\n")
        
        f.write(f"{'#':<6} {'Image Name':<35} {'Ground Truth':<25} {'OCR Output':<25} {'Status':<10} {'CER':<8}\n")
        f.write("-" * 110 + "\n")
        
        match_count = 0; partial_count = 0; miss_count = 0; empty_count = 0
        
        for idx, r in enumerate(results, 1):
            if r['gt'] == r['ocr']:
                status = "EXACT"; match_count += 1
            elif not r['ocr']:
                status = "EMPTY"; empty_count += 1
            elif r['cer'] < 0.3:
                status = "CLOSE"; partial_count += 1
            elif r['cer'] < 0.5:
                status = "PARTIAL"; partial_count += 1
            else:
                status = "MISS"; miss_count += 1
            
            gt_disp = r['gt'][:23] if len(r['gt']) > 23 else r['gt']
            ocr_disp = r['ocr'][:23] if len(r['ocr']) > 23 else r['ocr']
            f.write(f"{idx:<6} {r['name']:<35} {gt_disp:<25} {ocr_disp:<25} {status:<10} {r['cer']:.4f}\n")
        
        f.write("\n" + "=" * 90 + "\n")
        f.write("  SUMMARY\n")
        f.write("-" * 90 + "\n")
        f.write(f"  EXACT matches:         {match_count:>6} ({match_count/max(len(results),1)*100:.1f}%)\n")
        f.write(f"  CLOSE matches (<30%):  {partial_count:>6} ({partial_count/max(len(results),1)*100:.1f}%)\n")
        f.write(f"  MISS (>50% CER):       {miss_count:>6} ({miss_count/max(len(results),1)*100:.1f}%)\n")
        f.write(f"  EMPTY (no output):     {empty_count:>6} ({empty_count/max(len(results),1)*100:.1f}%)\n")
        f.write(f"  ─────────────────────────────────\n")
        f.write(f"  TOTAL:                 {len(results):>6}\n")
        f.write(f"\n  Recognition Rate (non-empty): {(len(results)-empty_count)/max(len(results),1)*100:.1f}%\n")
        f.write(f"  Usable Output (EXACT+CLOSE):  {(match_count+partial_count)/max(len(results),1)*100:.1f}%\n")
        f.write("=" * 90 + "\n")
    
    print(f"Processed words saved: {PROCESSED_WORDS_FILE}")

def generate_report(all_results):
    with open(REPORT_FILE, 'w', encoding='utf-8') as f:
        f.write("=" * 85 + "\n")
        f.write("       BENGALI OCR EVALUATION REPORT\n")
        f.write("       Fine-tuned vs Default Tesseract LSTM Model\n")
        f.write("=" * 85 + "\n\n")
        f.write(f"Test Dataset:   {BENGALI_IMG_DIR} (Scene text line images)\n")
        f.write(f"Ground Truth:   {BENGALI_GT_DIR}\n")
        f.write(f"Preprocessing:  Upscaling + Sharpening + Contrast Enhancement\n")
        f.write(f"OCR Strategy:   PSM 7 on preprocessed, fallback PSM 8\n")
        f.write(f"Excluded:       ### (illegible) and empty GT entries\n\n")
        
        # Main table
        f.write("=" * 85 + "\n")
        f.write("  OVERALL RESULTS\n")
        f.write("=" * 85 + "\n")
        f.write(f"{'Metric':<50}")
        for r in all_results: f.write(f"| {r['model_name']:<16}")
        f.write("|\n" + "-"*85 + "\n")
        
        rows = [
            ('Images Processed', 'processed', 'd'),
            ('Empty OCR Outputs', 'empty_ocr', 'd'),
            ('Total Ground Truth Words', 'total_gt_words', 'd'),
            ('Correctly Detected Words', 'total_correct_words', 'd'),
            ('Character Error Rate (CER)', 'cer', '.4f'),
            ('Character Accuracy (%)', 'char_acc', '.2f'),
            ('Word Error Rate (WER - edit distance)', 'wer', '.4f'),
            ('Simplified Word Error Rate', 'simple_wer', '.4f'),
            ('Simplified Word Accuracy (%)', 'simple_acc', '.2f'),
            ('Exact Line Match', 'exact_match', 'd'),
            ('Exact Match Rate (%)', 'exact_pct', '.1f'),
        ]
        for label, key, fmt in rows:
            f.write(f"{label:<50}")
            for r in all_results: f.write(f"| {r[key]:<16{fmt}}")
            f.write("|\n")
        f.write("-"*85 + "\n\n")
        
        # Non-empty results table
        f.write("=" * 85 + "\n")
        f.write("  RESULTS ON RECOGNIZED IMAGES (excluding empty OCR outputs)\n")
        f.write("=" * 85 + "\n")
        f.write(f"{'Metric':<50}")
        for r in all_results: f.write(f"| {r['model_name']:<16}")
        f.write("|\n" + "-"*85 + "\n")
        
        ne_rows = [
            ('Images with OCR Output', 'ne_count', 'd'),
            ('Character Error Rate (CER)', 'ne_cer', '.4f'),
            ('Character Accuracy (%)', 'ne_char_acc', '.2f'),
            ('Simplified Word Accuracy (%)', 'ne_word_acc', '.2f'),
            ('Exact Match Count', 'ne_exact', 'd'),
            ('Exact Match Rate (%)', 'ne_exact_pct', '.1f'),
        ]
        for label, key, fmt in ne_rows:
            f.write(f"{label:<50}")
            for r in all_results: f.write(f"| {r[key]:<16{fmt}}")
            f.write("|\n")
        f.write("-"*85 + "\n\n")
        
        # Improvement
        if len(all_results) == 2:
            ft, df = all_results[0], all_results[1]
            f.write("=" * 85 + "\n")
            f.write("  IMPROVEMENT (Fine-tuned over Default)\n")
            f.write("=" * 85 + "\n")
            f.write(f"  CER Reduction:                  {df['cer']-ft['cer']:+.4f}\n")
            f.write(f"  Character Accuracy Improvement:  {ft['char_acc']-df['char_acc']:+.2f}%\n")
            f.write(f"  WER Reduction:                  {df['wer']-ft['wer']:+.4f}\n")
            f.write(f"  Simple Word Accuracy Gain:       {ft['simple_acc']-df['simple_acc']:+.2f}%\n")
            f.write(f"  Fewer Empty Outputs:             {df['empty_ocr']-ft['empty_ocr']}\n")
            f.write(f"  More Exact Matches:              {ft['exact_match']-df['exact_match']}\n")
            f.write(f"\n  Non-empty CER Improvement:      {df['ne_cer']-ft['ne_cer']:+.4f}\n")
            f.write(f"  Non-empty Char Acc Improvement:  {ft['ne_char_acc']-df['ne_char_acc']:+.2f}%\n\n")
        
        # Samples
        if all_results:
            ft_res = all_results[0]['results']
            
            f.write("=" * 85 + "\n  SAMPLE EXACT MATCHES (Fine-tuned, first 25)\n" + "=" * 85 + "\n")
            cnt = 0
            for r in ft_res:
                if r['gt'] == r['ocr'] and cnt < 25:
                    f.write(f"  {r['name']}: [{r['gt']}]\n"); cnt += 1
            
            f.write("\n" + "=" * 85 + "\n  SAMPLE NEAR MATCHES (CER < 0.3, first 25)\n" + "=" * 85 + "\n\n")
            cnt = 0
            for r in ft_res:
                if r['gt'] != r['ocr'] and r['ocr'] and r['cer'] < 0.3 and cnt < 25:
                    f.write(f"  {r['name']}: GT=[{r['gt']}] OCR=[{r['ocr']}] CER={r['cer']:.3f}\n"); cnt += 1
            
            f.write("\n" + "=" * 85 + "\n  TOP 30 MOST MISSED WORDS\n" + "=" * 85 + "\n\n")
            missing = defaultdict(int)
            for r in ft_res:
                for w in set(r['gt'].split()) - set(r['ocr'].split()):
                    missing[w] += 1
            for word, count in sorted(missing.items(), key=lambda x:-x[1])[:30]:
                f.write(f"  {word} (missed {count} times)\n")
    
    print(f"Report saved: {REPORT_FILE}")

def main():
    configure_console()
    pairs = get_valid_pairs()
    if not pairs:
        print("ERROR: No valid pairs!"); sys.exit(1)
    
    # Preprocess all images upfront
    preprocess_all(pairs)
    
    all_results = []
    for lang, name in MODELS.items():
        td = os.path.join(TESSDATA_DIR, f"{lang}.traineddata")
        if os.path.exists(td):
            all_results.append(evaluate_model(pairs, lang, name))
        else:
            print(f"Skipping {name} - not found")
    
    generate_report(all_results)
    if all_results:
        generate_processed_words(all_results[0])
    
    print("\n" + "="*60 + "\n  EVALUATION COMPLETE!\n" + "="*60)

if __name__ == "__main__":
    main()
