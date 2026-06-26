"""
Unified Metrics Module for Bengali Scene Text Recognition
==========================================================
Computes all metrics used in the research paper:
  - WRR  (Word Recognition Rate)  — exact word match rate
  - CER  (Character Error Rate)   — Levenshtein at char level
  - NED  (Normalized Edit Distance)
  - 1-NED (similarity, higher=better)
  - Inference speed (FPS)

All metrics follow the definitions used in IndicSTR12 (ICDAR 2023)
and Bharat Scene Text Dataset (2025) for fair comparison.
"""
import time
import unicodedata
from typing import List, Dict, Tuple, Optional


def normalize_bengali(text: str) -> str:
    """Normalize Bengali text for fair evaluation.
    
    - NFC normalization (canonical composition)
    - Remove zero-width chars (ZWJ, ZWNJ, ZWSP, BOM)
    - Strip and collapse whitespace
    """
    text = unicodedata.normalize('NFC', text.strip())
    # Remove zero-width characters that don't affect visual rendering
    for zw in ['\u200c', '\u200d', '\u200b', '\ufeff']:
        text = text.replace(zw, '')
    return ' '.join(text.split())


def edit_distance(ref: str, hyp: str) -> int:
    """Compute Levenshtein edit distance between two strings.
    
    Uses O(min(m,n)) space dynamic programming.
    """
    if len(ref) < len(hyp):
        return edit_distance(hyp, ref)
    
    m, n = len(ref), len(hyp)
    if n == 0:
        return m
    
    prev = list(range(n + 1))
    for i in range(1, m + 1):
        curr = [i] + [0] * n
        for j in range(1, n + 1):
            cost = 0 if ref[i-1] == hyp[j-1] else 1
            curr[j] = min(prev[j] + 1, curr[j-1] + 1, prev[j-1] + cost)
        prev = curr
    return prev[n]


def compute_cer(gt: str, pred: str) -> float:
    """Character Error Rate = edit_distance(gt, pred) / len(gt).
    
    Returns 0.0 if gt is empty, 1.0 if pred is empty and gt is not.
    """
    gt_n = normalize_bengali(gt)
    pred_n = normalize_bengali(pred)
    
    if len(gt_n) == 0:
        return 0.0 if len(pred_n) == 0 else 1.0
    
    return edit_distance(gt_n, pred_n) / len(gt_n)


def compute_ned(gt: str, pred: str) -> float:
    """Normalized Edit Distance = edit_distance / max(len(gt), len(pred)).
    
    As used in IndicSTR12 and BSTD benchmarks.
    Returns 0.0 for identical strings (or both empty).
    """
    gt_n = normalize_bengali(gt)
    pred_n = normalize_bengali(pred)
    
    max_len = max(len(gt_n), len(pred_n))
    if max_len == 0:
        return 0.0
    
    return edit_distance(gt_n, pred_n) / max_len


def compute_wrr(gt: str, pred: str) -> float:
    """Word Recognition Rate — 1.0 if exact match after normalization, else 0.0.
    
    This is a per-sample metric; corpus WRR = mean of sample WRRs.
    """
    return 1.0 if normalize_bengali(gt) == normalize_bengali(pred) else 0.0


def evaluate_corpus(
    ground_truths: List[str],
    predictions: List[str],
    verbose: bool = False
) -> Dict:
    """Evaluate an entire corpus of predictions against ground truths.
    
    Args:
        ground_truths: List of ground truth strings
        predictions: List of predicted strings (same order)
        verbose: Print progress
    
    Returns:
        Dictionary with all aggregate metrics
    """
    assert len(ground_truths) == len(predictions), \
        f"Length mismatch: {len(ground_truths)} GTs vs {len(predictions)} preds"
    
    n = len(ground_truths)
    if n == 0:
        return {
            'WRR': 0.0, 'CER': 1.0, 'NED': 1.0, '1-NED': 0.0,
            'num_samples': 0, 'exact_matches': 0,
        }
    
    total_cer_num = 0
    total_gt_chars = 0
    total_ned = 0.0
    exact_matches = 0
    sample_results = []
    
    for i, (gt, pred) in enumerate(zip(ground_truths, predictions)):
        gt_n = normalize_bengali(gt)
        pred_n = normalize_bengali(pred)
        
        # CER components (corpus-level CER = total_edits / total_gt_chars)
        ed = edit_distance(gt_n, pred_n)
        total_cer_num += ed
        total_gt_chars += len(gt_n)
        
        # NED (averaged across samples)
        ned = compute_ned(gt, pred)
        total_ned += ned
        
        # WRR (exact match)
        is_exact = (gt_n == pred_n)
        if is_exact:
            exact_matches += 1
        
        sample_results.append({
            'gt': gt_n,
            'pred': pred_n,
            'cer': ed / max(len(gt_n), 1),
            'ned': ned,
            'exact_match': is_exact,
        })
        
        if verbose and (i + 1) % 500 == 0:
            print(f"  [{i+1}/{n}] Running CER={total_cer_num/max(total_gt_chars,1):.4f} "
                  f"WRR={exact_matches/(i+1)*100:.1f}%")
    
    corpus_cer = total_cer_num / max(total_gt_chars, 1)
    corpus_ned = total_ned / n
    corpus_wrr = exact_matches / n
    
    return {
        'WRR': corpus_wrr * 100,          # percentage
        'CER': corpus_cer * 100,           # percentage
        'NED': corpus_ned,                 # 0-1
        '1-NED': (1.0 - corpus_ned) * 100, # percentage (higher = better)
        'num_samples': n,
        'exact_matches': exact_matches,
        'char_accuracy': (1 - corpus_cer) * 100,
        'per_sample': sample_results,
    }


def format_results_table(results: Dict, model_name: str = "Model") -> str:
    """Format results as a nice text table for printing/logging."""
    lines = [
        f"{'='*50}",
        f"  Results: {model_name}",
        f"{'='*50}",
        f"  Samples:        {results['num_samples']}",
        f"  Exact Matches:  {results['exact_matches']}",
        f"  WRR:            {results['WRR']:.2f}%",
        f"  CER:            {results['CER']:.2f}%",
        f"  Char Accuracy:  {results['char_accuracy']:.2f}%",
        f"  NED:            {results['NED']:.4f}",
        f"  1-NED:          {results['1-NED']:.2f}%",
        f"{'='*50}",
    ]
    return '\n'.join(lines)


class FPSTimer:
    """Context manager to measure inference FPS."""
    
    def __init__(self):
        self.start_time = None
        self.end_time = None
        self.num_images = 0
    
    def __enter__(self):
        self.start_time = time.time()
        return self
    
    def __exit__(self, *args):
        self.end_time = time.time()
    
    @property
    def elapsed(self) -> float:
        if self.end_time is None:
            return time.time() - self.start_time
        return self.end_time - self.start_time
    
    @property
    def fps(self) -> float:
        elapsed = self.elapsed
        if elapsed == 0:
            return 0.0
        return self.num_images / elapsed


if __name__ == '__main__':
    # Quick test with Bengali text
    import sys
    sys.stdout.reconfigure(encoding='utf-8')
    
    gts = ['বাংলা', 'পরীক্ষা', 'ফলাফল', 'কম্পিউটার']
    preds = ['বাংলা', 'পরীক্ষা', 'ফলাফ',  'কম্পিউটা']
    
    results = evaluate_corpus(gts, preds)
    print(format_results_table(results, "Test"))
    
    # Test individual metrics
    print(f"\nPer-sample CER test:")
    for gt, pred in zip(gts, preds):
        print(f"  '{gt}' vs '{pred}' → CER={compute_cer(gt,pred):.3f} NED={compute_ned(gt,pred):.3f}")
