"""
Evaluation script for trained CRNN model
- Loads best model checkpoint
- Evaluates on test set (unseen during training)
- Compares with Tesseract baseline results
- Generates research-paper-quality report
"""
import os
import json
import time
import sys
import torch
from torch.utils.data import DataLoader

from dataset import BengaliSceneTextDataset, collate_fn, load_dataset_splits
from model import build_model
from train import decode_predictions, compute_metrics, Config


def configure_console():
    """Avoid Windows console crashes when printing Bengali text."""
    try:
        sys.stdout.reconfigure(encoding='utf-8', errors='replace')
        sys.stderr.reconfigure(encoding='utf-8', errors='replace')
    except Exception:
        pass


def load_model(checkpoint_path, device):
    """Load trained model from checkpoint."""
    checkpoint = torch.load(checkpoint_path, map_location=device, weights_only=False)
    
    num_classes = checkpoint['num_classes']
    config = checkpoint['config']
    
    model = build_model(
        num_classes=num_classes,
        img_height=config['img_height'],
        hidden_size=config['hidden_size']
    )
    model.load_state_dict(checkpoint['model_state_dict'])
    model = model.to(device)
    model.eval()
    
    char2idx = checkpoint['char2idx']
    idx2char = {int(k): v for k, v in checkpoint['idx2char'].items()}
    
    print(f"Model loaded from: {checkpoint_path}")
    print(f"  Best epoch: {checkpoint['epoch']}")
    print(f"  Val CER: {checkpoint.get('val_cer', 'N/A')}")
    print(f"  Val Word Acc: {checkpoint.get('val_word_acc', 'N/A')}%")
    
    return model, char2idx, idx2char, config


@torch.no_grad()
def evaluate_test_set(model, test_loader, device, idx2char):
    """Full evaluation on test set."""
    model.eval()
    all_preds = []
    all_gts = []
    
    for images, labels, label_lengths, texts in test_loader:
        images = images.to(device)
        preds = model(images)
        decoded = decode_predictions(preds, idx2char)
        all_preds.extend(decoded)
        all_gts.extend(texts)
    
    metrics = compute_metrics(all_preds, all_gts)
    return metrics, all_preds, all_gts


def generate_report(metrics, predictions, ground_truths, training_log=None,
                    output_file='crnn_evaluation_report.txt'):
    """Generate comprehensive evaluation report."""
    
    # Tesseract baseline (from previous evaluation)
    tesseract_baseline = {
        'char_acc': 45.50,
        'word_acc': 24.78,
        'exact_match_rate': 24.8,
        'cer': 0.545,
        'wer': 0.752,
    }
    
    tesseract_default = {
        'char_acc': 45.12,
        'word_acc': 24.37,
        'exact_match_rate': 24.4,
        'cer': 0.549,
        'wer': 0.756,
    }
    
    lines = []
    lines.append("=" * 80)
    lines.append("  BENGALI SCENE TEXT RECOGNITION - CRNN EVALUATION REPORT")
    lines.append("=" * 80)
    lines.append("")
    lines.append(f"  Generated: {time.strftime('%Y-%m-%d %H:%M:%S')}")
    lines.append(f"  Model: CRNN (CNN + BiLSTM + CTC)")
    lines.append(f"  Test samples: {metrics['total_samples']}")
    lines.append("")
    
    # === Comparison Table ===
    lines.append("=" * 80)
    lines.append("  COMPARATIVE RESULTS")
    lines.append("=" * 80)
    lines.append("")
    lines.append(f"  {'Metric':<25} {'Tesseract Default':>20} {'Tesseract Finetuned':>20} {'CRNN (Ours)':>15}")
    lines.append(f"  {'-'*25} {'-'*20} {'-'*20} {'-'*15}")
    lines.append(f"  {'Character Accuracy':<25} {tesseract_default['char_acc']:>19.2f}% {tesseract_baseline['char_acc']:>19.2f}% {metrics['char_acc']:>14.2f}%")
    lines.append(f"  {'Word Accuracy':<25} {tesseract_default['word_acc']:>19.2f}% {tesseract_baseline['word_acc']:>19.2f}% {metrics['word_acc']:>14.2f}%")
    lines.append(f"  {'Exact Match Rate':<25} {tesseract_default['exact_match_rate']:>19.2f}% {tesseract_baseline['exact_match_rate']:>19.2f}% {metrics['exact_match_rate']:>14.2f}%")
    lines.append(f"  {'CER':<25} {tesseract_default['cer']:>20.4f} {tesseract_baseline['cer']:>20.4f} {metrics['cer']:>15.4f}")
    lines.append(f"  {'WER':<25} {tesseract_default['wer']:>20.4f} {tesseract_baseline['wer']:>20.4f} {metrics['wer']:>15.4f}")
    lines.append("")
    
    # === Improvement over baseline ===
    char_improvement = metrics['char_acc'] - tesseract_baseline['char_acc']
    word_improvement = metrics['word_acc'] - tesseract_baseline['word_acc']
    exact_improvement = metrics['exact_match_rate'] - tesseract_baseline['exact_match_rate']
    
    lines.append("=" * 80)
    lines.append("  IMPROVEMENT OVER TESSERACT (FINETUNED)")
    lines.append("=" * 80)
    lines.append("")
    lines.append(f"  Character Accuracy: +{char_improvement:.2f}% ({tesseract_baseline['char_acc']:.2f}% → {metrics['char_acc']:.2f}%)")
    lines.append(f"  Word Accuracy:      +{word_improvement:.2f}% ({tesseract_baseline['word_acc']:.2f}% → {metrics['word_acc']:.2f}%)")
    lines.append(f"  Exact Match Rate:   +{exact_improvement:.2f}% ({tesseract_baseline['exact_match_rate']:.2f}% → {metrics['exact_match_rate']:.2f}%)")
    lines.append("")
    
    # === Detailed Per-Sample Results ===
    lines.append("=" * 80)
    lines.append("  DETAILED PER-SAMPLE RESULTS (Test Set)")
    lines.append("=" * 80)
    lines.append("")
    
    correct = []
    incorrect = []
    for pred, gt in zip(predictions, ground_truths):
        if pred == gt:
            correct.append((gt, pred))
        else:
            incorrect.append((gt, pred))
    
    lines.append(f"  Correct predictions: {len(correct)} / {len(predictions)}")
    lines.append(f"  Incorrect predictions: {len(incorrect)} / {len(predictions)}")
    lines.append("")
    
    # Show some correct examples
    lines.append("  --- Correct Predictions (sample) ---")
    for gt, pred in correct[:20]:
        lines.append(f"    ✓ [{gt}]")
    lines.append("")
    
    # Show incorrect examples
    lines.append("  --- Incorrect Predictions ---")
    for gt, pred in incorrect[:50]:
        lines.append(f"    ✗ GT: [{gt}] → Pred: [{pred}]")
    lines.append("")
    
    # === Training Summary ===
    if training_log and 'summary' in training_log:
        summary = training_log['summary']
        lines.append("=" * 80)
        lines.append("  TRAINING SUMMARY")
        lines.append("=" * 80)
        lines.append("")
        lines.append(f"  Total training time: {summary.get('total_time_seconds', 0)/60:.1f} minutes")
        lines.append(f"  Total epochs: {summary.get('total_epochs', 'N/A')}")
        lines.append(f"  Best val CER: {summary.get('best_val_cer', 'N/A')}")
        lines.append(f"  Best val Word Acc: {summary.get('best_val_word_acc', 'N/A')}%")
        
        config = training_log.get('config', {})
        lines.append(f"  Batch size: {config.get('batch_size', 'N/A')}")
        lines.append(f"  Learning rate: {config.get('learning_rate', 'N/A')}")
        lines.append(f"  Image size: {config.get('img_size', 'N/A')}")
        lines.append(f"  Train samples: {config.get('train_size', 'N/A')}")
        lines.append(f"  Val samples: {config.get('val_size', 'N/A')}")
        lines.append(f"  Test samples: {config.get('test_size', 'N/A')}")
        lines.append("")
    
    lines.append("=" * 80)
    lines.append("  END OF REPORT")
    lines.append("=" * 80)
    
    report = '\n'.join(lines)
    
    with open(output_file, 'w', encoding='utf-8') as f:
        f.write(report)
    
    print(f"\nReport saved: {output_file}")
    print(report)
    
    return report


def save_detailed_results(predictions, ground_truths, output_file='crnn_detailed_results.txt'):
    """Save detailed per-sample results."""
    with open(output_file, 'w', encoding='utf-8') as f:
        f.write("Ground Truth\tPrediction\tMatch\n")
        f.write("-" * 80 + "\n")
        for gt, pred in zip(ground_truths, predictions):
            match = "✓" if gt == pred else "✗"
            f.write(f"{gt}\t{pred}\t{match}\n")
    print(f"Detailed results saved: {output_file}")


def main():
    configure_console()
    cfg = Config()
    
    print("=" * 70)
    print("  Bengali CRNN Model - Test Set Evaluation")
    print("=" * 70)
    
    # Load model
    print("\nLoading model...")
    model, char2idx, idx2char, model_config = load_model(cfg.best_model_path, cfg.device)
    
    # Load test set
    print("\nLoading test set...")
    _, _, test_pairs, _, data = load_dataset_splits(cfg.dataset_json)
    print(f"  Test samples: {len(test_pairs)}")
    
    test_dataset = BengaliSceneTextDataset(
        test_pairs, char2idx, cfg.img_height, cfg.img_width, augment=False)
    test_loader = DataLoader(
        test_dataset, batch_size=cfg.batch_size, shuffle=False,
        num_workers=cfg.num_workers, collate_fn=collate_fn)
    
    # Evaluate
    print("\nEvaluating on test set...")
    metrics, predictions, ground_truths = evaluate_test_set(
        model, test_loader, cfg.device, idx2char)
    
    print(f"\n  Character Accuracy: {metrics['char_acc']:.2f}%")
    print(f"  Word Accuracy:      {metrics['word_acc']:.2f}%")
    print(f"  Exact Match:        {metrics['exact_match_rate']:.2f}%")
    print(f"  CER:                {metrics['cer']:.4f}")
    print(f"  WER:                {metrics['wer']:.4f}")
    
    # Load training log for report
    training_log = None
    if os.path.exists(cfg.log_file):
        with open(cfg.log_file, 'r') as f:
            training_log = json.load(f)
    
    # Generate comparative report
    generate_report(metrics, predictions, ground_truths, training_log)
    
    # Save detailed results
    save_detailed_results(predictions, ground_truths)
    
    # Save metrics as JSON for figures script
    metrics_output = {
        'crnn': metrics,
        'tesseract_finetuned': {
            'char_acc': 45.50, 'word_acc': 24.78,
            'exact_match_rate': 24.8, 'cer': 0.545, 'wer': 0.752,
        },
        'tesseract_default': {
            'char_acc': 45.12, 'word_acc': 24.37,
            'exact_match_rate': 24.4, 'cer': 0.549, 'wer': 0.756,
        }
    }
    with open('evaluation_metrics.json', 'w') as f:
        json.dump(metrics_output, f, indent=2)
    
    print("\nAll evaluation files generated successfully!")


if __name__ == '__main__':
    main()
