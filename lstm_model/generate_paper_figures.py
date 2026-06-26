"""
Generate publication-quality figures for the research paper.
- Training curves (loss, CER, accuracy per epoch)
- Accuracy comparison bar charts (CRNN vs Tesseract)
- Sample prediction visualizations
"""
import os
import json
import numpy as np
import matplotlib
matplotlib.use('Agg')  # Non-interactive backend
import matplotlib.pyplot as plt
from matplotlib import rcParams

# Publication-quality settings
rcParams['font.family'] = 'serif'
rcParams['font.size'] = 11
rcParams['axes.labelsize'] = 12
rcParams['axes.titlesize'] = 13
rcParams['legend.fontsize'] = 10
rcParams['figure.dpi'] = 300

OUTPUT_DIR = 'paper_figures'
os.makedirs(OUTPUT_DIR, exist_ok=True)


def plot_training_curves(log_file='training_log.json'):
    """Plot training and validation loss/CER curves."""
    with open(log_file, 'r') as f:
        log = json.load(f)
    
    epochs_data = log['epochs']
    epochs = [e['epoch'] for e in epochs_data]
    train_loss = [e['train_loss'] for e in epochs_data]
    val_loss = [e['val_loss'] for e in epochs_data]
    train_cer = [e.get('train_cer', 1.0) for e in epochs_data]
    val_cer = [e.get('val_cer', 1.0) for e in epochs_data]
    val_char_acc = [e.get('val_char_acc', 0) for e in epochs_data]
    val_word_acc = [e.get('val_word_acc', 0) for e in epochs_data]
    
    # --- Figure 1: Loss Curves ---
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 4.5))
    
    ax1.plot(epochs, train_loss, 'b-', label='Training Loss', linewidth=1.5, alpha=0.8)
    ax1.plot(epochs, val_loss, 'r-', label='Validation Loss', linewidth=1.5, alpha=0.8)
    ax1.set_xlabel('Epoch')
    ax1.set_ylabel('CTC Loss')
    ax1.set_title('(a) Training and Validation Loss')
    ax1.legend()
    ax1.grid(True, alpha=0.3)
    ax1.set_xlim(1, max(epochs))
    
    ax2.plot(epochs, [c * 100 for c in train_cer], 'b-', label='Train CER', linewidth=1.5, alpha=0.8)
    ax2.plot(epochs, [c * 100 for c in val_cer], 'r-', label='Val CER', linewidth=1.5, alpha=0.8)
    ax2.set_xlabel('Epoch')
    ax2.set_ylabel('Character Error Rate (%)')
    ax2.set_title('(b) Character Error Rate')
    ax2.legend()
    ax2.grid(True, alpha=0.3)
    ax2.set_xlim(1, max(epochs))
    
    plt.tight_layout()
    plt.savefig(os.path.join(OUTPUT_DIR, 'fig1_training_curves.png'), bbox_inches='tight')
    plt.close()
    print("  ✓ fig1_training_curves.png")
    
    # --- Figure 2: Accuracy Curves ---
    fig, ax = plt.subplots(figsize=(8, 4.5))
    
    ax.plot(epochs, val_char_acc, 'g-', label='Character Accuracy', linewidth=2, alpha=0.8)
    ax.plot(epochs, val_word_acc, 'm-', label='Word Accuracy', linewidth=2, alpha=0.8)
    ax.axhline(y=90, color='k', linestyle='--', alpha=0.3, label='90% Target')
    ax.set_xlabel('Epoch')
    ax.set_ylabel('Accuracy (%)')
    ax.set_title('Validation Accuracy over Training')
    ax.legend()
    ax.grid(True, alpha=0.3)
    ax.set_xlim(1, max(epochs))
    ax.set_ylim(0, 100)
    
    plt.tight_layout()
    plt.savefig(os.path.join(OUTPUT_DIR, 'fig2_accuracy_curves.png'), bbox_inches='tight')
    plt.close()
    print("  ✓ fig2_accuracy_curves.png")


def plot_comparison_charts(metrics_file='evaluation_metrics.json'):
    """Plot CRNN vs Tesseract comparison bar charts."""
    with open(metrics_file, 'r') as f:
        metrics = json.load(f)
    
    crnn = metrics['crnn']
    tess_ft = metrics['tesseract_finetuned']
    tess_def = metrics['tesseract_default']
    
    # --- Figure 3: Bar Chart Comparison ---
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 5))
    
    # Accuracy metrics
    categories = ['Character\nAccuracy', 'Word\nAccuracy', 'Exact\nMatch Rate']
    tess_def_vals = [tess_def['char_acc'], tess_def['word_acc'], tess_def['exact_match_rate']]
    tess_ft_vals = [tess_ft['char_acc'], tess_ft['word_acc'], tess_ft['exact_match_rate']]
    crnn_vals = [crnn['char_acc'], crnn['word_acc'], crnn['exact_match_rate']]
    
    x = np.arange(len(categories))
    width = 0.25
    
    bars1 = ax1.bar(x - width, tess_def_vals, width, label='Tesseract Default',
                     color='#3498db', alpha=0.8)
    bars2 = ax1.bar(x, tess_ft_vals, width, label='Tesseract Fine-tuned',
                     color='#e67e22', alpha=0.8)
    bars3 = ax1.bar(x + width, crnn_vals, width, label='CRNN (Ours)',
                     color='#2ecc71', alpha=0.8)
    
    ax1.set_ylabel('Accuracy (%)')
    ax1.set_title('(a) Recognition Accuracy Comparison')
    ax1.set_xticks(x)
    ax1.set_xticklabels(categories)
    ax1.legend(loc='upper left')
    ax1.set_ylim(0, 105)
    ax1.axhline(y=90, color='red', linestyle='--', alpha=0.4, label='90% Target')
    ax1.grid(axis='y', alpha=0.3)
    
    # Add value labels on bars
    for bars in [bars1, bars2, bars3]:
        for bar in bars:
            height = bar.get_height()
            ax1.annotate(f'{height:.1f}%',
                        xy=(bar.get_x() + bar.get_width() / 2, height),
                        xytext=(0, 3), textcoords="offset points",
                        ha='center', va='bottom', fontsize=8)
    
    # Error rate metrics
    categories2 = ['CER', 'WER']
    tess_def_err = [tess_def['cer'], tess_def['wer']]
    tess_ft_err = [tess_ft['cer'], tess_ft['wer']]
    crnn_err = [crnn['cer'], crnn['wer']]
    
    x2 = np.arange(len(categories2))
    
    bars4 = ax2.bar(x2 - width, tess_def_err, width, label='Tesseract Default',
                     color='#3498db', alpha=0.8)
    bars5 = ax2.bar(x2, tess_ft_err, width, label='Tesseract Fine-tuned',
                     color='#e67e22', alpha=0.8)
    bars6 = ax2.bar(x2 + width, crnn_err, width, label='CRNN (Ours)',
                     color='#2ecc71', alpha=0.8)
    
    ax2.set_ylabel('Error Rate')
    ax2.set_title('(b) Error Rate Comparison')
    ax2.set_xticks(x2)
    ax2.set_xticklabels(categories2)
    ax2.legend(loc='upper right')
    ax2.set_ylim(0, 1.0)
    ax2.grid(axis='y', alpha=0.3)
    
    for bars in [bars4, bars5, bars6]:
        for bar in bars:
            height = bar.get_height()
            ax2.annotate(f'{height:.3f}',
                        xy=(bar.get_x() + bar.get_width() / 2, height),
                        xytext=(0, 3), textcoords="offset points",
                        ha='center', va='bottom', fontsize=8)
    
    plt.tight_layout()
    plt.savefig(os.path.join(OUTPUT_DIR, 'fig3_comparison_charts.png'), bbox_inches='tight')
    plt.close()
    print("  ✓ fig3_comparison_charts.png")


def plot_architecture_diagram():
    """Create a simple architecture flow diagram."""
    fig, ax = plt.subplots(figsize=(14, 3))
    ax.set_xlim(0, 14)
    ax.set_ylim(0, 3)
    ax.axis('off')
    
    blocks = [
        (0.5, 'Input\n(32×128)\nGrayscale', '#3498db'),
        (2.5, 'Conv Block 1\n64 filters\nMaxPool', '#2ecc71'),
        (4.5, 'Conv Block 2\n128 filters\nMaxPool', '#2ecc71'),
        (6.5, 'Conv Block 3\n256 filters\n×2 layers', '#2ecc71'),
        (8.5, 'Conv Block 4\n512 filters\n×2 layers', '#2ecc71'),
        (10.5, 'BiLSTM\n256 hidden\n×2 layers', '#e74c3c'),
        (12.5, 'FC + CTC\nDecode\nOutput', '#9b59b6'),
    ]
    
    for i, (x, label, color) in enumerate(blocks):
        rect = plt.Rectangle((x, 0.4), 1.6, 2.2, facecolor=color, alpha=0.3,
                            edgecolor=color, linewidth=2, zorder=2)
        ax.add_patch(rect)
        ax.text(x + 0.8, 1.5, label, ha='center', va='center',
               fontsize=8, fontweight='bold', zorder=3)
        
        if i < len(blocks) - 1:
            ax.annotate('', xy=(x + 2.0, 1.5), xytext=(x + 1.7, 1.5),
                       arrowprops=dict(arrowstyle='->', lw=1.5, color='gray'))
    
    ax.set_title('CRNN Architecture for Bengali Scene Text Recognition', fontsize=13, pad=15)
    
    plt.tight_layout()
    plt.savefig(os.path.join(OUTPUT_DIR, 'fig4_architecture.png'), bbox_inches='tight')
    plt.close()
    print("  ✓ fig4_architecture.png")


def main():
    print("=" * 60)
    print("  Generating Research Paper Figures")
    print("=" * 60)
    print(f"  Output directory: {OUTPUT_DIR}/")
    print("")
    
    # Figure 1 & 2: Training curves
    if os.path.exists('training_log.json'):
        plot_training_curves()
    else:
        print("  ⚠ training_log.json not found, skipping training curves")
    
    # Figure 3: Comparison charts
    if os.path.exists('evaluation_metrics.json'):
        plot_comparison_charts()
    else:
        print("  ⚠ evaluation_metrics.json not found, skipping comparison charts")
    
    # Figure 4: Architecture diagram
    plot_architecture_diagram()
    
    print(f"\n  All figures saved to: {OUTPUT_DIR}/")
    print("  Figures are 300 DPI, suitable for publication.")


if __name__ == '__main__':
    main()
