import subprocess
import os

test_imgs = ['gt_img_6401_line1', 'gt_img_6401_line5', 'gt_img_6402_line1', 'gt_img_6405_line1', 'gt_img_6410_line1']
GT_DIR = r'c:\lstm_model\Bengali_gt'
IMG_DIR = r'c:\lstm_model\Bengali'
TESSDATA = r'c:\lstm_model\tessdata'

for name in test_imgs:
    gt = open(os.path.join(GT_DIR, name + '.txt'), 'r', encoding='utf-8').read().strip()
    r = subprocess.run(
        ['tesseract', os.path.join(IMG_DIR, name + '.jpg'), 'stdout', '-l', 'ben_finetuned', '--tessdata-dir', TESSDATA, '--psm', '7'],
        capture_output=True, text=True, encoding='utf-8', errors='replace'
    )
    ocr = r.stdout.strip()
    
    # CER
    def edit_dist(a, b):
        m, n = len(a), len(b)
        d = [[0]*(n+1) for _ in range(m+1)]
        for i in range(m+1): d[i][0] = i
        for j in range(n+1): d[0][j] = j
        for i in range(1, m+1):
            for j in range(1, n+1):
                d[i][j] = min(d[i-1][j]+1, d[i][j-1]+1, d[i-1][j-1] + (0 if a[i-1]==b[j-1] else 1))
        return d[m][n]
    
    ed = edit_dist(gt, ocr)
    cer = ed / max(len(gt), 1)
    char_acc = (1 - cer) * 100
    
    print(f"Image: {name}")
    print(f"  GT:  [{gt}] (len={len(gt)})")
    print(f"  OCR: [{ocr}] (len={len(ocr)})")
    print(f"  Edit Distance: {ed}")
    print(f"  CER: {cer:.4f}")
    print(f"  Char Accuracy: {char_acc:.1f}%")
    print(f"  Exact Match: {gt == ocr}")
    print()
