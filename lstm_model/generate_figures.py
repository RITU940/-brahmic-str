"""
Paper Figure Generator for Bengali Scene Text Recognition
===========================================================
Generates publication-quality figures from training logs and results.

Figures:
  1. Training curves (loss + WRR per epoch)
  2. Baseline comparison bar chart
  3. Ablation: grapheme vs standard tokenizer
  4. CER distribution histogram
  5. Sample predictions (qualitative)

Usage:
  python generate_figures.py
"""
import os
import sys
import json
import argparse

sys.stdout.reconfigure(encoding='utf-8', errors='replace')

BASE_DIR = os.path.abspath(os.path.dirname(__file__))

try:
    import matplotlib
    matplotlib.use('Agg')  # Non-interactive backend
    import matplotlib.pyplot as plt
    import matplotlib.font_manager as fm
    import numpy as np
    HAS_MPL = True
except ImportError:
    HAS_MPL = False
    print("WARNING: matplotlib not installed. Install with: pip install matplotlib")


def setup_style():
    """Set up publication-quality matplotlib style."""
    plt.rcParams.update({
        'figure.figsize': (8, 5),
        'figure.dpi': 150,
        'savefig.dpi': 300,
        'savefig.bbox': 'tight',
        'axes.labelsize': 12,
        'axes.titlesize': 14,
        'xtick.labelsize': 10,
        'ytick.labelsize': 10,
        'legend.fontsize': 10,
        'font.family': 'serif',
        'axes.grid': True,
        'grid.alpha': 0.3,
    })


def fig1_training_curves(fig_dir):
    """Generate training loss and WRR curves for both modes."""
    print("  Generating Fig 1: Training curves...")
    
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 5))
    
    colors = {'standard': '#2196F3', 'grapheme': '#FF5722'}
    labels = {'standard': 'Standard Tokenizer', 'grapheme': 'Grapheme Tokenizer (Ours)'}
    
    any_data = False
    for mode in ['standard', 'grapheme']:
        log_path = os.path.join(BASE_DIR, f'training_log_florence2_{mode}.json')
        if not os.path.exists(log_path):
            continue
        
        with open(log_path, 'r', encoding='utf-8') as f:
            log = json.load(f)
        
        epochs_data = log['epochs']
        epochs = [e['epoch'] for e in epochs_data]
        train_loss = [e['train_loss'] for e in epochs_data]
        val_loss = [e['val_loss'] for e in epochs_data]
        val_wrr = [e['val_wrr'] for e in epochs_data]
        
        ax1.plot(epochs, train_loss, '-o', color=colors[mode], alpha=0.6,
                label=f'{labels[mode]} (train)', markersize=4)
        ax1.plot(epochs, val_loss, '-s', color=colors[mode],
                label=f'{labels[mode]} (val)', markersize=4)
        
        ax2.plot(epochs, val_wrr, '-o', color=colors[mode],
                label=labels[mode], markersize=5, linewidth=2)
        
        any_data = True
    
    if not any_data:
        print("    No training logs found. Skipping.")
        plt.close()
        return
    
    ax1.set_xlabel('Epoch')
    ax1.set_ylabel('Loss')
    ax1.set_title('(a) Training & Validation Loss')
    ax1.legend(loc='upper right', fontsize=8)
    
    ax2.set_xlabel('Epoch')
    ax2.set_ylabel('WRR (%)')
    ax2.set_title('(b) Validation Word Recognition Rate')
    ax2.legend(loc='lower right')
    
    plt.suptitle('Florence-2 Fine-Tuning on Bengali Scene Text', fontsize=14, y=1.02)
    plt.tight_layout()
    
    save_path = os.path.join(fig_dir, 'fig1_training_curves.png')
    plt.savefig(save_path)
    plt.close()
    print(f"    Saved: {save_path}")


def fig2_comparison_bar(fig_dir):
    """Generate comparison bar chart across all baselines."""
    print("  Generating Fig 2: Baseline comparison...")
    
    table_path = os.path.join(BASE_DIR, 'paper_comparison_table.json')
    if not os.path.exists(table_path):
        print("    No comparison table found. Run run_baselines.py first. Skipping.")
        return
    
    with open(table_path, 'r', encoding='utf-8') as f:
        rows = json.load(f)
    
    if not rows:
        print("    Empty comparison table. Skipping.")
        return
    
    models = [r['model'] for r in rows]
    wrr = [r['WRR'] for r in rows]
    cer = [r['CER'] for r in rows]
    one_ned = [r['1-NED'] for r in rows]
    
    x = np.arange(len(models))
    width = 0.25
    
    fig, ax = plt.subplots(figsize=(10, 6))
    
    bars1 = ax.bar(x - width, wrr, width, label='WRR (%)', color='#4CAF50', alpha=0.85)
    bars2 = ax.bar(x, one_ned, width, label='1-NED (%)', color='#2196F3', alpha=0.85)
    bars3 = ax.bar(x + width, [100 - c for c in cer], width, label='Char Acc (%)', color='#FF9800', alpha=0.85)
    
    ax.set_xlabel('Model')
    ax.set_ylabel('Score (%)')
    ax.set_title('Bengali Scene Text Recognition — Model Comparison')
    ax.set_xticks(x)
    ax.set_xticklabels(models, rotation=30, ha='right', fontsize=9)
    ax.legend()
    ax.set_ylim(0, 105)
    
    # Add value labels on bars
    for bars in [bars1, bars2, bars3]:
        for bar in bars:
            height = bar.get_height()
            if height > 0:
                ax.annotate(f'{height:.1f}', xy=(bar.get_x() + bar.get_width() / 2, height),
                           xytext=(0, 3), textcoords="offset points",
                           ha='center', va='bottom', fontsize=7)
    
    plt.tight_layout()
    save_path = os.path.join(fig_dir, 'fig2_comparison.png')
    plt.savefig(save_path)
    plt.close()
    print(f"    Saved: {save_path}")


def fig3_ablation(fig_dir):
    """Generate ablation study chart: grapheme vs standard tokenizer."""
    print("  Generating Fig 3: Ablation study...")
    
    results = {}
    for mode in ['standard', 'grapheme']:
        result_path = os.path.join(BASE_DIR, f'florence2_results_{mode}.json')
        if os.path.exists(result_path):
            with open(result_path, 'r', encoding='utf-8') as f:
                results[mode] = json.load(f)
    
    if len(results) < 2:
        print("    Need both standard and grapheme results. Skipping.")
        return
    
    metrics = ['WRR', 'CER', '1-NED', 'char_accuracy']
    metric_labels = ['WRR (%)', 'CER (%)', '1-NED (%)', 'Char Accuracy (%)']
    
    fig, axes = plt.subplots(1, 4, figsize=(16, 4))
    
    for i, (metric, label) in enumerate(zip(metrics, metric_labels)):
        ax = axes[i]
        std_val = results['standard'].get(metric, 0)
        grp_val = results['grapheme'].get(metric, 0)
        
        bars = ax.bar(['Standard', 'Grapheme\n(Ours)'], [std_val, grp_val],
                      color=['#90CAF9', '#FF5722'], edgecolor='black', linewidth=0.5)
        
        ax.set_title(label, fontsize=11)
        ax.set_ylabel(label)
        
        for bar, val in zip(bars, [std_val, grp_val]):
            ax.annotate(f'{val:.1f}', xy=(bar.get_x() + bar.get_width()/2, val),
                       xytext=(0, 5), textcoords='offset points',
                       ha='center', fontsize=10, fontweight='bold')
    
    plt.suptitle('Ablation Study: Grapheme vs Standard Tokenization', fontsize=14, y=1.02)
    plt.tight_layout()
    
    save_path = os.path.join(fig_dir, 'fig3_ablation.png')
    plt.savefig(save_path)
    plt.close()
    print(f"    Saved: {save_path}")


def fig4_cer_distribution(fig_dir):
    """Generate CER distribution histogram."""
    print("  Generating Fig 4: CER distribution...")
    
    any_data = False
    fig, ax = plt.subplots(figsize=(8, 5))
    
    for mode, color, label in [('grapheme', '#FF5722', 'Grapheme (Ours)'),
                                ('standard', '#2196F3', 'Standard')]:
        result_path = os.path.join(BASE_DIR, f'florence2_results_{mode}.json')
        if not os.path.exists(result_path):
            continue
        
        with open(result_path, 'r', encoding='utf-8') as f:
            data = json.load(f)
        
        if 'predictions' not in data:
            continue
        
        from metrics import compute_cer
        cers = [compute_cer(p['gt'], p['pred']) for p in data['predictions']]
        
        ax.hist(cers, bins=50, alpha=0.6, color=color, label=label, edgecolor='black', linewidth=0.5)
        any_data = True
    
    if not any_data:
        print("    No prediction data found. Skipping.")
        plt.close()
        return
    
    ax.set_xlabel('Character Error Rate')
    ax.set_ylabel('Number of Samples')
    ax.set_title('CER Distribution on Test Set')
    ax.legend()
    ax.axvline(x=0.0, color='green', linestyle='--', alpha=0.5, label='Perfect')
    
    plt.tight_layout()
    save_path = os.path.join(fig_dir, 'fig4_cer_distribution.png')
    plt.savefig(save_path)
    plt.close()
    print(f"    Saved: {save_path}")


def fig5_dataset_stats(fig_dir):
    """Generate dataset statistics figure."""
    print("  Generating Fig 5: Dataset statistics...")
    
    stats_path = os.path.join(BASE_DIR, 'data_statistics.json')
    if not os.path.exists(stats_path):
        print("    No data_statistics.json found. Skipping.")
        return
    
    with open(stats_path, 'r', encoding='utf-8') as f:
        data = json.load(f)
    
    ds = data['dataset_stats']
    
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 5))
    
    # Split sizes
    splits = ['Train', 'Validation', 'Test']
    sizes = [ds['train_samples'], ds['val_samples'], ds['test_samples']]
    colors = ['#4CAF50', '#FF9800', '#F44336']
    
    wedges, texts, autotexts = ax1.pie(
        sizes, labels=splits, colors=colors, autopct='%1.1f%%',
        startangle=90, textprops={'fontsize': 11}
    )
    ax1.set_title(f'Dataset Split (n={ds["total_samples"]})')
    
    # Char vs Grapheme vocab
    categories = ['Unique\nChars', 'Unique\nGraphemes']
    values = [ds['unique_chars'], ds['unique_graphemes']]
    bars = ax2.bar(categories, values, color=['#2196F3', '#FF5722'],
                   edgecolor='black', linewidth=0.5, width=0.5)
    
    for bar, val in zip(bars, values):
        ax2.annotate(str(val), xy=(bar.get_x() + bar.get_width()/2, val),
                    xytext=(0, 5), textcoords='offset points',
                    ha='center', fontsize=14, fontweight='bold')
    
    ax2.set_ylabel('Count')
    ax2.set_title('Character vs Grapheme Vocabulary Size')
    
    plt.suptitle('Bengali Scene Text Dataset Overview', fontsize=14, y=1.02)
    plt.tight_layout()
    
    save_path = os.path.join(fig_dir, 'fig5_dataset_stats.png')
    plt.savefig(save_path)
    plt.close()
    print(f"    Saved: {save_path}")


def main():
    if not HAS_MPL:
        print("ERROR: matplotlib is required. Install with: pip install matplotlib")
        sys.exit(1)
    
    setup_style()
    
    fig_dir = os.path.join(BASE_DIR, 'paper_figures')
    os.makedirs(fig_dir, exist_ok=True)
    
    print("=" * 60)
    print("  Generating Paper Figures")
    print("=" * 60)
    
    fig1_training_curves(fig_dir)
    fig2_comparison_bar(fig_dir)
    fig3_ablation(fig_dir)
    fig4_cer_distribution(fig_dir)
    fig5_dataset_stats(fig_dir)
    
    print(f"\n{'='*60}")
    print(f"  All figures saved to: {fig_dir}")
    print(f"{'='*60}")


if __name__ == '__main__':
    main()
