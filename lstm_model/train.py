"""
Training script for Bengali Scene Text CRNN Model
- CTC loss training
- Validation CER tracking
- Early stopping & model checkpointing
- Comprehensive logging for research paper
"""
import os
import json
import time
import sys
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader

from dataset import BengaliSceneTextDataset, collate_fn, load_dataset_splits
from model import build_model


def configure_console():
    """Avoid Windows console crashes when printing Bengali text."""
    try:
        sys.stdout.reconfigure(encoding='utf-8', errors='replace')
        sys.stderr.reconfigure(encoding='utf-8', errors='replace')
    except Exception:
        pass


# ===================== Configuration =====================
class Config:
    # Data
    dataset_json = 'dataset_splits.json'
    img_height = 32
    img_width = 128
    
    # Model
    hidden_size = 256
    
    # Training
    batch_size = 32
    num_epochs = 150
    learning_rate = 0.001
    weight_decay = 1e-4
    
    # Scheduler
    lr_patience = 8
    lr_factor = 0.5
    min_lr = 1e-6
    
    # Early stopping
    early_stop_patience = 20
    
    # Checkpointing
    checkpoint_dir = 'checkpoints'
    best_model_path = 'checkpoints/best_model.pth'
    
    # Logging
    log_file = 'training_log.json'
    
    # Device
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    num_workers = 2 if torch.cuda.is_available() else 0


def decode_predictions(preds, idx2char):
    """Greedy CTC decoding: collapse repeated chars and remove blanks."""
    pred_texts = []
    
    # preds: (seq_len, batch, num_classes)
    _, pred_indices = preds.max(2)  # (seq_len, batch)
    pred_indices = pred_indices.permute(1, 0).cpu().numpy()  # (batch, seq_len)
    
    for seq in pred_indices:
        chars = []
        prev = -1
        for idx in seq:
            if idx != 0 and idx != prev:  # 0 = CTC blank
                if idx in idx2char:
                    chars.append(idx2char[idx])
            prev = idx
        pred_texts.append(''.join(chars))
    
    return pred_texts


def edit_distance(ref, hyp):
    """Compute edit distance between two strings."""
    m, n = len(ref), len(hyp)
    if m == 0: return n
    if n == 0: return m
    dp = list(range(n + 1))
    for i in range(1, m + 1):
        curr = [i] + [0] * n
        for j in range(1, n + 1):
            curr[j] = min(dp[j] + 1, curr[j-1] + 1,
                         dp[j-1] + (0 if ref[i-1] == hyp[j-1] else 1))
        dp = curr
    return dp[n]


def compute_metrics(predictions, ground_truths):
    """Compute CER, WER, Word Accuracy, Exact Match."""
    total_char_dist = 0
    total_chars = 0
    total_word_dist = 0
    total_words = 0
    exact_matches = 0
    correct_words = 0
    
    for pred, gt in zip(predictions, ground_truths):
        # Character-level
        char_dist = edit_distance(list(gt), list(pred))
        total_char_dist += char_dist
        total_chars += max(len(gt), 1)
        
        # Word-level
        gt_words = gt.split()
        pred_words = pred.split()
        word_dist = edit_distance(gt_words, pred_words)
        total_word_dist += word_dist
        total_words += max(len(gt_words), 1)
        
        # Word accuracy (matching words in order)
        for gw, pw in zip(gt_words, pred_words):
            if gw == pw:
                correct_words += 1
        
        # Exact match
        if pred == gt:
            exact_matches += 1
    
    n = len(predictions)
    cer = total_char_dist / max(total_chars, 1)
    wer = total_word_dist / max(total_words, 1)
    word_acc = correct_words / max(total_words, 1) * 100
    char_acc = (1 - cer) * 100
    exact_match_rate = exact_matches / max(n, 1) * 100
    
    return {
        'cer': cer,
        'wer': wer,
        'char_acc': char_acc,
        'word_acc': word_acc,
        'exact_match': exact_matches,
        'exact_match_rate': exact_match_rate,
        'total_samples': n,
    }


def train_one_epoch(model, dataloader, criterion, optimizer, device, idx2char):
    """Train for one epoch."""
    model.train()
    total_loss = 0
    num_batches = 0
    all_preds = []
    all_gts = []
    
    for batch_idx, (images, labels, label_lengths, texts) in enumerate(dataloader):
        images = images.to(device)
        labels = labels.to(device)
        label_lengths = label_lengths.to(device)
        
        # Forward
        preds = model(images)  # (seq_len, batch, num_classes)
        seq_len = preds.size(0)
        batch_size = preds.size(1)
        
        # Input lengths for CTC (all same = seq_len)
        input_lengths = torch.full((batch_size,), seq_len, dtype=torch.long, device=device)
        
        # CTC loss
        loss = criterion(preds, labels, input_lengths, label_lengths)
        
        if torch.isnan(loss) or torch.isinf(loss):
            continue
        
        # Backward
        optimizer.zero_grad()
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=5.0)
        optimizer.step()
        
        total_loss += loss.item()
        num_batches += 1
        
        # Decode predictions for metrics (every 10th batch to save time)
        if batch_idx % 10 == 0:
            decoded = decode_predictions(preds.detach(), idx2char)
            all_preds.extend(decoded)
            all_gts.extend(texts)
    
    avg_loss = total_loss / max(num_batches, 1)
    
    # Compute training metrics on sampled predictions
    metrics = compute_metrics(all_preds, all_gts) if all_preds else {
        'cer': 1.0, 'wer': 1.0, 'char_acc': 0, 'word_acc': 0,
        'exact_match': 0, 'exact_match_rate': 0, 'total_samples': 0
    }
    
    return avg_loss, metrics


@torch.no_grad()
def evaluate(model, dataloader, criterion, device, idx2char):
    """Evaluate model on validation/test set."""
    model.eval()
    total_loss = 0
    num_batches = 0
    all_preds = []
    all_gts = []
    
    for images, labels, label_lengths, texts in dataloader:
        images = images.to(device)
        labels = labels.to(device)
        label_lengths = label_lengths.to(device)
        
        preds = model(images)
        seq_len = preds.size(0)
        batch_size = preds.size(1)
        
        input_lengths = torch.full((batch_size,), seq_len, dtype=torch.long, device=device)
        
        loss = criterion(preds, labels, input_lengths, label_lengths)
        if not (torch.isnan(loss) or torch.isinf(loss)):
            total_loss += loss.item()
            num_batches += 1
        
        # Decode all predictions
        decoded = decode_predictions(preds, idx2char)
        all_preds.extend(decoded)
        all_gts.extend(texts)
    
    avg_loss = total_loss / max(num_batches, 1)
    metrics = compute_metrics(all_preds, all_gts)
    
    return avg_loss, metrics, all_preds, all_gts


def train():
    """Main training function."""
    configure_console()
    cfg = Config()
    
    print("=" * 70)
    print("  Bengali Scene Text Recognition - CRNN Training")
    print("=" * 70)
    print(f"  Device: {cfg.device}")
    print(f"  Batch size: {cfg.batch_size}")
    print(f"  Learning rate: {cfg.learning_rate}")
    print(f"  Max epochs: {cfg.num_epochs}")
    print("=" * 70)
    
    # Load dataset
    print("\nLoading dataset...")
    train_pairs, val_pairs, test_pairs, char2idx, data = load_dataset_splits(cfg.dataset_json)
    
    num_classes = data['stats']['num_classes']
    chars = data['vocabulary']['chars']
    idx2char = {v: k for k, v in char2idx.items()}
    
    print(f"  Train: {len(train_pairs)}, Val: {len(val_pairs)}, Test: {len(test_pairs)}")
    print(f"  Vocabulary: {len(chars)} chars, {num_classes} classes (with blank)")
    
    # Create datasets
    train_dataset = BengaliSceneTextDataset(
        train_pairs, char2idx, cfg.img_height, cfg.img_width, augment=True)
    val_dataset = BengaliSceneTextDataset(
        val_pairs, char2idx, cfg.img_height, cfg.img_width, augment=False)
    
    train_loader = DataLoader(
        train_dataset, batch_size=cfg.batch_size, shuffle=True,
        num_workers=cfg.num_workers, collate_fn=collate_fn, pin_memory=True)
    val_loader = DataLoader(
        val_dataset, batch_size=cfg.batch_size, shuffle=False,
        num_workers=cfg.num_workers, collate_fn=collate_fn, pin_memory=True)
    
    # Build model
    print("\nBuilding model...")
    model = build_model(num_classes, cfg.img_height, cfg.hidden_size)
    model = model.to(cfg.device)
    
    # CTC Loss
    criterion = nn.CTCLoss(blank=0, zero_infinity=True)
    
    # Optimizer
    optimizer = optim.Adam(model.parameters(), lr=cfg.learning_rate,
                          weight_decay=cfg.weight_decay)
    
    # Learning rate scheduler
    scheduler = optim.lr_scheduler.ReduceLROnPlateau(
        optimizer, mode='min', factor=cfg.lr_factor,
        patience=cfg.lr_patience, min_lr=cfg.min_lr)
    
    # Checkpointing
    os.makedirs(cfg.checkpoint_dir, exist_ok=True)
    best_val_cer = float('inf')
    best_val_word_acc = 0
    epochs_without_improvement = 0
    
    # Training log
    training_log = {
        'config': {
            'batch_size': cfg.batch_size,
            'learning_rate': cfg.learning_rate,
            'hidden_size': cfg.hidden_size,
            'img_size': f"{cfg.img_height}x{cfg.img_width}",
            'num_classes': num_classes,
            'train_size': len(train_pairs),
            'val_size': len(val_pairs),
            'test_size': len(test_pairs),
            'device': cfg.device,
        },
        'epochs': []
    }
    
    print(f"\n{'='*70}")
    print(f"  Starting Training...")
    print(f"{'='*70}")
    
    training_start = time.time()
    
    for epoch in range(1, cfg.num_epochs + 1):
        epoch_start = time.time()
        
        # Train
        train_loss, train_metrics = train_one_epoch(
            model, train_loader, criterion, optimizer, cfg.device, idx2char)
        
        # Validate
        val_loss, val_metrics, val_preds, val_gts = evaluate(
            model, val_loader, criterion, cfg.device, idx2char)
        
        # Update scheduler
        scheduler.step(val_loss)
        
        epoch_time = time.time() - epoch_start
        current_lr = optimizer.param_groups[0]['lr']
        
        # Log
        epoch_log = {
            'epoch': epoch,
            'train_loss': round(train_loss, 4),
            'val_loss': round(val_loss, 4),
            'train_cer': round(train_metrics['cer'], 4),
            'val_cer': round(val_metrics['cer'], 4),
            'val_char_acc': round(val_metrics['char_acc'], 2),
            'val_word_acc': round(val_metrics['word_acc'], 2),
            'val_exact_match_rate': round(val_metrics['exact_match_rate'], 2),
            'lr': current_lr,
            'time': round(epoch_time, 1),
        }
        training_log['epochs'].append(epoch_log)
        
        # Print progress
        improved = ""
        if val_metrics['cer'] < best_val_cer:
            best_val_cer = val_metrics['cer']
            best_val_word_acc = val_metrics['word_acc']
            epochs_without_improvement = 0
            improved = " ★ BEST"
            
            # Save best model
            torch.save({
                'epoch': epoch,
                'model_state_dict': model.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'val_cer': val_metrics['cer'],
                'val_word_acc': val_metrics['word_acc'],
                'val_char_acc': val_metrics['char_acc'],
                'val_exact_match_rate': val_metrics['exact_match_rate'],
                'num_classes': num_classes,
                'char2idx': char2idx,
                'idx2char': {str(k): v for k, v in idx2char.items()},
                'config': {
                    'img_height': cfg.img_height,
                    'img_width': cfg.img_width,
                    'hidden_size': cfg.hidden_size,
                }
            }, cfg.best_model_path)
        else:
            epochs_without_improvement += 1
        
        print(f"Epoch {epoch:3d}/{cfg.num_epochs} | "
              f"Loss: {train_loss:.4f}/{val_loss:.4f} | "
              f"CER: {train_metrics['cer']:.4f}/{val_metrics['cer']:.4f} | "
              f"CharAcc: {val_metrics['char_acc']:.1f}% | "
              f"WordAcc: {val_metrics['word_acc']:.1f}% | "
              f"Exact: {val_metrics['exact_match_rate']:.1f}% | "
              f"LR: {current_lr:.6f} | "
              f"{epoch_time:.1f}s{improved}")
        
        # Show sample predictions every 10 epochs
        if epoch % 10 == 0 and val_preds:
            print(f"  --- Sample Predictions (epoch {epoch}) ---")
            for i in range(min(5, len(val_preds))):
                status = "✓" if val_preds[i] == val_gts[i] else "✗"
                print(f"  {status} GT: [{val_gts[i]}] → Pred: [{val_preds[i]}]")
        
        # Early stopping
        if epochs_without_improvement >= cfg.early_stop_patience:
            print(f"\n  Early stopping at epoch {epoch} "
                  f"(no improvement for {cfg.early_stop_patience} epochs)")
            break
        
        # Save log periodically
        if epoch % 5 == 0:
            with open(cfg.log_file, 'w', encoding='utf-8') as f:
                json.dump(training_log, f, indent=2)
    
    total_time = time.time() - training_start
    
    # Final log save
    training_log['summary'] = {
        'total_time_seconds': round(total_time, 1),
        'total_epochs': epoch,
        'best_val_cer': round(best_val_cer, 4),
        'best_val_word_acc': round(best_val_word_acc, 2),
        'best_val_char_acc': round((1 - best_val_cer) * 100, 2),
    }
    
    with open(cfg.log_file, 'w', encoding='utf-8') as f:
        json.dump(training_log, f, indent=2)
    
    print(f"\n{'='*70}")
    print(f"  Training Complete!")
    print(f"  Total time: {total_time/60:.1f} minutes")
    print(f"  Best validation CER: {best_val_cer:.4f}")
    print(f"  Best validation Char Accuracy: {(1-best_val_cer)*100:.2f}%")
    print(f"  Best validation Word Accuracy: {best_val_word_acc:.2f}%")
    print(f"  Best model saved: {cfg.best_model_path}")
    print(f"  Training log: {cfg.log_file}")
    print(f"{'='*70}")


if __name__ == '__main__':
    train()
