"""
Florence-2 Fine-Tuning for Bengali Scene Text Recognition
==========================================================
Fine-tunes Microsoft Florence-2 (base or large) on Bengali scene text data
using LoRA (Low-Rank Adaptation) for parameter-efficient training.

Two modes:
  --use_graphemes    → Inject custom Bengali graphemes into tokenizer (PROPOSED METHOD)
  (default)          → Standard tokenizer (BASELINE for ablation)

Usage:
  python train_florence2.py                           # Standard tokenizer baseline
  python train_florence2.py --use_graphemes            # Grapheme tokenizer (our method)
  python train_florence2.py --evaluate                 # Evaluate best checkpoint
  python train_florence2.py --use_graphemes --evaluate  # Evaluate grapheme model

Requirements:
  pip install torch transformers peft pillow accelerate bitsandbytes
"""
import os
import sys
import json
import time
import random
import argparse
from datetime import datetime
from typing import Optional, Dict, List, Tuple

import numpy as np
import torch
from torch.utils.data import Dataset, DataLoader
from PIL import Image

sys.stdout.reconfigure(encoding='utf-8', errors='replace')

# ── Configuration ──────────────────────────────────────────────────
BASE_DIR = os.path.abspath(os.path.dirname(__file__))
SPLITS_FILE = os.path.join(BASE_DIR, 'florence2_splits.json')
GRAPHEME_VOCAB_FILE = os.path.join(BASE_DIR, 'grapheme_vocab.json')

DEFAULT_SEED = 42


def set_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    os.environ['PYTHONHASHSEED'] = str(seed)

# Use microsoft/Florence-2-base — has proper auto_map in config.json
# flash_attn is handled via monkey-patch below (see _patch_flash_attn)
DEFAULT_MODEL_ID = "microsoft/Florence-2-base"
TASK_PROMPT = "<OCR>"  # Florence-2 OCR task prompt

# Ahana's Banglish tokenizer (already on HuggingFace — expands vocab 51k→71k)
BANGLISH_TOKENIZER_ID = "RocketFuel810/florence2-banglish-tokenizer"
HF_TOKEN = "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"

# Training hyperparameters — matched to Sharmistha's best config (r=16)
DEFAULT_CONFIG = {
    'epochs': 15,
    'batch_size': 4,
    'lr': 1e-4,
    'weight_decay': 0.01,
    'warmup_ratio': 0.1,
    'max_grad_norm': 1.0,
    'lora_r': 16,
    'lora_alpha': 32,
    'lora_dropout': 0.05,
    'img_max_size': 768,
    'max_length': 128,
    'save_every_n_epochs': 3,
    'patience': 7,  # early stopping
}


# ── Flash-Attn Bypass ──────────────────────────────────────────────
def _patch_flash_attn():
    """Create a real flash_attn stub package in site-packages if not installed.

    The dummy sys.modules approach fails because Python's inspect/importlib
    introspection checks __spec__, __file__, getfile() etc. The only bulletproof
    fix is a real package directory with a real __init__.py on disk.
    """
    try:
        import flash_attn
        return  # already installed (real or stub), nothing to do
    except (ImportError, ValueError):
        pass

    import site, os
    # Find site-packages directory
    sp_dirs = site.getsitepackages()
    sp = sp_dirs[0] if sp_dirs else os.path.join(os.path.dirname(os.__file__), 'site-packages')

    pkg_dir = os.path.join(sp, 'flash_attn')
    init_file = os.path.join(pkg_dir, '__init__.py')

    if not os.path.exists(init_file):
        os.makedirs(pkg_dir, exist_ok=True)
        with open(init_file, 'w') as f:
            f.write('# Stub package — provides import but no functionality\\n')
            f.write('# Florence-2 will use eager attention instead\\n')
            f.write('__version__ = "0.0.0"\\n')
        # Also create common submodules that Florence-2 might import
        for sub in ['flash_attn_interface', 'bert_padding']:
            with open(os.path.join(pkg_dir, f'{sub}.py'), 'w') as f:
                f.write('# stub\\n')
        print(f"  [info] Created flash_attn stub at {pkg_dir}")
    else:
        print(f"  [info] flash_attn stub already exists")


# ── Dataset ────────────────────────────────────────────────────────
class Florence2OCRDataset(Dataset):
    """Dataset for Florence-2 OCR fine-tuning."""
    
    def __init__(self, pairs, processor, max_length=128):
        self.pairs = pairs
        self.processor = processor
        self.max_length = max_length
    
    def __len__(self):
        return len(self.pairs)
    
    def __getitem__(self, idx):
        pair = self.pairs[idx]
        img_path = pair.get('resolved_image') or resolve_image_path(pair['image'])
        if not img_path:
            raise FileNotFoundError(
                f"Image not found for split entry: {pair.get('image')}"
            )

        gt_text = pair['gt']
        image = Image.open(img_path).convert('RGB')

        return image, gt_text, img_path


def _dedupe_paths(paths: List[str]) -> List[str]:
    """Keep candidate path order stable while removing duplicates."""
    seen = set()
    unique = []
    for path in paths:
        norm = os.path.abspath(path)
        if norm in seen:
            continue
        seen.add(norm)
        unique.append(norm)
    return unique


def resolve_image_path(raw_path: str) -> Optional[str]:
    """Resolve Windows/relative split paths to an image present on this server."""
    normalized = str(raw_path).replace('\\', '/')
    filename = os.path.basename(normalized)
    stem, ext = os.path.splitext(filename)

    candidates = []
    if os.path.isabs(normalized):
        candidates.append(normalized)
    else:
        candidates.extend([
            os.path.join(BASE_DIR, normalized),
            os.path.join(os.getcwd(), normalized),
        ])

    candidates.append(os.path.join(BASE_DIR, 'Bengali', filename))

    # Some server copies have only the preprocessed PNG for a split entry that
    # was originally written as .jpg.
    candidates.extend([
        os.path.join(BASE_DIR, 'Bengali_preprocessed', f'{stem}.png'),
        os.path.join(BASE_DIR, 'Bengali_preprocessed', filename),
    ])

    if ext.lower() != '.jpg':
        candidates.append(os.path.join(BASE_DIR, 'Bengali', f'{stem}.jpg'))

    for candidate in _dedupe_paths(candidates):
        if os.path.exists(candidate):
            return candidate
    return None


def validate_and_resolve_pairs(
    split_name: str,
    pairs: List[Dict],
    filter_missing: bool = False,
    max_examples: int = 8,
) -> Tuple[List[Dict], int]:
    """Validate image paths, optionally dropping entries missing on this server."""
    resolved_pairs = []
    missing = []

    for pair in pairs:
        img_path = resolve_image_path(pair['image'])
        if img_path:
            resolved = dict(pair)
            resolved['original_image'] = pair['image']
            resolved['image'] = img_path
            resolved['resolved_image'] = img_path
            resolved_pairs.append(resolved)
        else:
            missing.append(pair)

    print(f"  {split_name}: {len(resolved_pairs)}/{len(pairs)} images found")
    if missing:
        print(f"    Missing {len(missing)} {split_name} images. Examples:")
        for pair in missing[:max_examples]:
            print(f"      - {pair.get('image')}")

    if missing and not filter_missing:
        raise FileNotFoundError(
            f"{len(missing)} images are missing in the {split_name} split. "
            "Sync the correct florence2_splits.json/images, or run with "
            "--filter_missing_images to train only on files present here."
        )

    return resolved_pairs, len(missing)


def collate_fn_factory(processor, max_length=128):
    """Create a collate function that uses the Florence-2 processor."""
    
    def collate_fn(batch):
        images, texts, paths = zip(*batch)
        
        # Process images
        inputs = processor(
            images=list(images),
            text=[TASK_PROMPT] * len(images),
            return_tensors="pt",
            padding=True,
        )
        
        # Process labels
        labels = processor.tokenizer(
            list(texts),
            return_tensors="pt",
            padding=True,
            truncation=True,
            max_length=max_length,
        )
        
        # FIX: Florence-2 requires pad tokens to be -100 so they are ignored in cross-entropy loss
        input_ids = labels['input_ids']
        input_ids[input_ids == processor.tokenizer.pad_token_id] = -100
        
        inputs['labels'] = input_ids
        
        return inputs, list(texts), list(paths)
    
    return collate_fn


# ── Grapheme Token Injection ───────────────────────────────────────
def inject_grapheme_tokens(model, processor, vocab_path: str):
    """Inject Bengali grapheme clusters as new tokens into Florence-2 tokenizer.
    
    This is the KEY NOVELTY of our paper.
    Standard tokenizer: breaks বাংলা → multiple BPE pieces
    Grapheme tokenizer: adds বাং, লা as atomic tokens → better alignment
    """
    print("  Injecting Bengali grapheme tokens...")
    
    with open(vocab_path, 'r', encoding='utf-8') as f:
        vocab_data = json.load(f)
    
    grapheme2idx = vocab_data['grapheme2idx']
    
    # Filter out special tokens and single ASCII chars
    new_tokens = []
    for token in grapheme2idx.keys():
        if token in ('<blank>', '<unk>'):
            continue
        # Only add tokens not already in the vocabulary
        encoded = processor.tokenizer.encode(token, add_special_tokens=False)
        decoded_back = processor.tokenizer.decode(encoded)
        # If the token gets decomposed (encoded into multiple sub-tokens), add it
        if len(encoded) > 1 or decoded_back.strip() != token:
            new_tokens.append(token)
    
    if new_tokens:
        num_added = processor.tokenizer.add_tokens(new_tokens)
        model.resize_token_embeddings(len(processor.tokenizer))
        print(f"  Added {num_added} new grapheme tokens (total vocab: {len(processor.tokenizer)})")
    else:
        print("  All graphemes already in vocabulary (no new tokens needed)")
    
    return num_added if new_tokens else 0


# ── Training Loop (with AMP for 4x speedup) ────────────────────────
def train_one_epoch(model, dataloader, optimizer, scheduler, scaler, device, epoch, total_epochs):
    """Train for one epoch using Automatic Mixed Precision (AMP)."""
    model.train()
    total_loss = 0.0
    num_batches = 0
    start_time = time.time()
    
    for batch_idx, (inputs, texts, paths) in enumerate(dataloader):
        inputs = {k: v.to(device) if isinstance(v, torch.Tensor) else v
                  for k, v in inputs.items()}
        
        with torch.amp.autocast('cuda'):
            outputs = model(**inputs)
            loss = outputs.loss
        
        if torch.isnan(loss) or torch.isinf(loss):
            optimizer.zero_grad()
            continue
        
        scaler.scale(loss).backward()
        scaler.unscale_(optimizer)
        torch.nn.utils.clip_grad_norm_(model.parameters(), DEFAULT_CONFIG['max_grad_norm'])
        scaler.step(optimizer)
        scaler.update()
        scheduler.step()
        optimizer.zero_grad()
        
        total_loss += loss.item()
        num_batches += 1
        
        if (batch_idx + 1) % 50 == 0:
            avg_loss = total_loss / num_batches
            elapsed = time.time() - start_time
            speed = (batch_idx + 1) / elapsed
            print(f"  Epoch {epoch}/{total_epochs} [{batch_idx+1}/{len(dataloader)}] "
                  f"loss={avg_loss:.4f} lr={scheduler.get_last_lr()[0]:.2e} "
                  f"{speed:.1f} batch/s")
    
    return total_loss / max(num_batches, 1)


@torch.no_grad()
def validate(model, dataloader, processor, device):
    """Validate and compute WRR on validation set. Uses greedy decoding for speed."""
    model.eval()
    total_loss = 0.0
    num_batches = 0
    correct = 0
    total = 0
    
    for inputs, texts, paths in dataloader:
        inputs = {k: v.to(device) if isinstance(v, torch.Tensor) else v
                  for k, v in inputs.items()}
        
        with torch.cuda.amp.autocast():
            outputs = model(**inputs)
        total_loss += outputs.loss.item()
        num_batches += 1
        
        # Greedy decoding during training (fast) — beam search only for final test
        gen_inputs = {k: v for k, v in inputs.items() if k != 'labels'}
        generated_ids = model.generate(
            **gen_inputs,
            max_new_tokens=DEFAULT_CONFIG['max_length'],
            num_beams=1,  # greedy = fast
        )
        
        preds = processor.batch_decode(generated_ids, skip_special_tokens=True)
        
        for pred, gt in zip(preds, texts):
            if pred.strip() == gt.strip():
                correct += 1
            total += 1
    
    val_loss = total_loss / max(num_batches, 1)
    wrr = correct / max(total, 1) * 100
    
    return val_loss, wrr


# ── Main ───────────────────────────────────────────────────────────
def main():
    # Patch flash_attn FIRST — before any transformers import touches the model cache
    _patch_flash_attn()

    parser = argparse.ArgumentParser(description="Florence-2 Bengali OCR Fine-tuning")
    parser.add_argument('--use_graphemes', action='store_true',
                        help='Inject Bengali grapheme tokens into tokenizer (OUR METHOD)')
    parser.add_argument('--evaluate', action='store_true',
                        help='Run evaluation on test set using best checkpoint')
    parser.add_argument('--model_id', default=DEFAULT_MODEL_ID,
                        help=f'HuggingFace model ID (default: {DEFAULT_MODEL_ID})')
    parser.add_argument('--epochs', type=int, default=DEFAULT_CONFIG['epochs'])
    parser.add_argument('--batch_size', type=int, default=DEFAULT_CONFIG['batch_size'])
    parser.add_argument('--lr', type=float, default=DEFAULT_CONFIG['lr'])
    parser.add_argument('--lora_r', type=int, default=DEFAULT_CONFIG['lora_r'])
    parser.add_argument('--resume', type=str, default=None,
                        help='Path to checkpoint to resume from')
    parser.add_argument('--splits_file', default=SPLITS_FILE,
                        help='Path to florence2_splits.json on this server')
    parser.add_argument('--grapheme_vocab', default=None,
                        help='Path to grapheme vocab JSON (default: grapheme_vocab.json — Bengali)')
    parser.add_argument('--ckpt_dir', default=None,
                        help='Override checkpoint dir (keeps experiments from colliding)')
    parser.add_argument('--check_data_only', action='store_true',
                        help='Validate split image paths and exit before loading the model')
    parser.add_argument('--filter_missing_images', action='store_true',
                        help='Drop split entries whose images are not present on this server')
    parser.add_argument('--seed', type=int, default=DEFAULT_SEED,
                        help='Random seed for reproducibility (Python/NumPy/PyTorch/CuDNN)')
    args = parser.parse_args()

    set_seed(args.seed)
    print(f"  Seed set to {args.seed} (deterministic CuDNN enabled)")

    mode_str = "GRAPHEME" if args.use_graphemes else "STANDARD"
    ckpt_dir = os.path.abspath(args.ckpt_dir) if args.ckpt_dir else os.path.join(BASE_DIR, f'checkpoints_florence2_{mode_str.lower()}')
    log_path = os.path.join(BASE_DIR, f'training_log_florence2_{mode_str.lower()}.json')
    
    print("=" * 60)
    print(f"  Florence-2 Bengali OCR Fine-Tuning")
    print(f"  Mode: {mode_str} tokenizer")
    print(f"  Model: {args.model_id}")
    print("=" * 60)
    
    # ── Load data ──
    print("\n[1/5] Loading data...")
    split_path = os.path.abspath(args.splits_file)
    if not os.path.exists(split_path):
        print(f"ERROR: {split_path} not found. Run prepare_florence2_data.py first!")
        sys.exit(1)

    print(f"  Split file: {split_path}")
    with open(split_path, 'r', encoding='utf-8') as f:
        splits = json.load(f)

    raw_train_pairs = splits['train']
    raw_val_pairs = splits['val']
    raw_test_pairs = splits['test']

    print(f"  Raw split sizes: Train {len(raw_train_pairs)} | Val {len(raw_val_pairs)} | Test {len(raw_test_pairs)}")
    print("  Resolving image paths on this server...")
    try:
        train_pairs, train_missing = validate_and_resolve_pairs(
            'train', raw_train_pairs, args.filter_missing_images
        )
        val_pairs, val_missing = validate_and_resolve_pairs(
            'val', raw_val_pairs, args.filter_missing_images
        )
        test_pairs, test_missing = validate_and_resolve_pairs(
            'test', raw_test_pairs, args.filter_missing_images
        )
    except FileNotFoundError as e:
        print(f"ERROR: {e}")
        sys.exit(1)

    total_missing = train_missing + val_missing + test_missing
    print(f"  Usable split sizes: Train {len(train_pairs)} | Val {len(val_pairs)} | Test {len(test_pairs)}")
    print(f"  Missing images: {total_missing}")

    if args.check_data_only:
        if total_missing and not args.filter_missing_images:
            sys.exit(1)
        print("  Data check complete.")
        return

    if args.evaluate:
        if not test_pairs:
            print("ERROR: No usable test samples after resolving image paths.")
            sys.exit(1)
    elif not train_pairs or not val_pairs:
        print("ERROR: Train/val split is empty after resolving image paths.")
        sys.exit(1)

    # ── Load model & processor ──
    print(f"\n[2/5] Loading model: {args.model_id}")
    
    try:
        from transformers import AutoModelForCausalLM, AutoProcessor, AutoTokenizer
    except ImportError:
        print("ERROR: Install transformers: pip install transformers")
        sys.exit(1)
    
    # Device setup
    if torch.cuda.is_available():
        device = torch.device('cuda')
        print(f"  GPU: {torch.cuda.get_device_name(0)} ({torch.cuda.get_device_properties(0).total_memory / 1e9:.1f} GB)")
    else:
        device = torch.device('cpu')
        print("  WARNING: No GPU found, training will be very slow!")
    
    # Load florence-community model (no flash_attn dependency)
    processor = AutoProcessor.from_pretrained(args.model_id, trust_remote_code=True)

    # Load Ahana's Banglish tokenizer (vocab 51k → 71k) — built on top of Florence-2
    print(f"  Loading Banglish tokenizer ({BANGLISH_TOKENIZER_ID})...")
    try:
        banglish_tok = AutoTokenizer.from_pretrained(BANGLISH_TOKENIZER_ID, token=HF_TOKEN)
        processor.tokenizer = banglish_tok
        print(f"  ✓ Banglish tokenizer loaded (vocab size: {len(processor.tokenizer)})")
    except Exception as e:
        print(f"  WARNING: Could not load Banglish tokenizer ({e}). Using default.")

    # Pre-load the config with trust_remote_code — registers the 'florence2'
    # custom model type BEFORE AutoModelForCausalLM tries to look it up in its registry.
    from transformers import AutoConfig
    config = AutoConfig.from_pretrained(args.model_id, trust_remote_code=True)

    model = AutoModelForCausalLM.from_pretrained(
        args.model_id,
        config=config,
        trust_remote_code=True,
        torch_dtype=torch.float32,
        attn_implementation="eager",  # skip flash_attn; supported since transformers 4.36
    )
    # Resize embeddings to match Banglish tokenizer
    model.resize_token_embeddings(len(processor.tokenizer))
    
    # ── Inject grapheme tokens (if using our method) ──
    num_new_tokens = 0
    if args.use_graphemes:
        grapheme_vocab_file = os.path.abspath(args.grapheme_vocab) if args.grapheme_vocab else GRAPHEME_VOCAB_FILE
        print(f"\n[3/5] Injecting grapheme tokens ({os.path.basename(grapheme_vocab_file)})...")
        if not os.path.exists(grapheme_vocab_file):
            print(f"ERROR: {grapheme_vocab_file} not found. Run prepare_florence2_data.py first!")
            sys.exit(1)
        num_new_tokens = inject_grapheme_tokens(model, processor, grapheme_vocab_file)
    else:
        print(f"\n[3/5] Using standard tokenizer (baseline)")
    
    # ── Apply LoRA (training) or load saved adapter (eval) ──
    try:
        from peft import LoraConfig, get_peft_model, PeftModel
    except ImportError:
        print("ERROR: Install peft: pip install peft")
        sys.exit(1)

    if args.evaluate:
        best_path = os.path.join(ckpt_dir, 'best_model')
        if not os.path.exists(best_path):
            print(f"ERROR: best_model not found at {best_path}. Train first.")
            sys.exit(1)
        print(f"\n[4/5] Loading fine-tuned adapter from {best_path}")
        # Base model already has Banglish-resized embeddings; load adapter on top.
        model = PeftModel.from_pretrained(model, best_path, is_trainable=False)
        model.to(device)
        print(f"\n[5/5] Evaluating on test set (beam search)...")
        run_evaluation(model, processor, test_pairs, device, ckpt_dir, mode_str)
        return

    print(f"\n[4/5] Applying LoRA (r={args.lora_r})...")
    # LoRA config matched to Sharmistha's best-performing setup (r=16)
    # modules_to_save keeps embed_tokens & lm_head fully trainable (critical for new Bengali tokens)
    lora_config = LoraConfig(
        r=args.lora_r,
        lora_alpha=DEFAULT_CONFIG['lora_alpha'],
        target_modules=["q_proj", "v_proj", "k_proj", "out_proj", "fc1", "fc2"],
        modules_to_save=["embed_tokens", "lm_head"],
        lora_dropout=DEFAULT_CONFIG['lora_dropout'],
        bias="none",
        task_type="CAUSAL_LM",
    )

    model = get_peft_model(model, lora_config)
    model.print_trainable_parameters()
    model.to(device)
    
    # ── Create datasets ──
    train_dataset = Florence2OCRDataset(train_pairs, processor, DEFAULT_CONFIG['max_length'])
    val_dataset = Florence2OCRDataset(val_pairs, processor, DEFAULT_CONFIG['max_length'])
    
    collate_fn = collate_fn_factory(processor, DEFAULT_CONFIG['max_length'])
    
    train_loader = DataLoader(
        train_dataset, batch_size=args.batch_size, shuffle=True,
        collate_fn=collate_fn, num_workers=0, pin_memory=True
    )
    val_loader = DataLoader(
        val_dataset, batch_size=args.batch_size, shuffle=False,
        collate_fn=collate_fn, num_workers=0, pin_memory=True
    )
    
    # ── Optimizer, Scheduler & AMP Scaler ──
    optimizer = torch.optim.AdamW(
        model.parameters(), lr=args.lr,
        weight_decay=DEFAULT_CONFIG['weight_decay']
    )
    
    total_steps = len(train_loader) * args.epochs
    warmup_steps = int(total_steps * DEFAULT_CONFIG['warmup_ratio'])
    scheduler = torch.optim.lr_scheduler.OneCycleLR(
        optimizer, max_lr=args.lr,
        total_steps=total_steps,
        pct_start=DEFAULT_CONFIG['warmup_ratio'],
        anneal_strategy='cos',
    )
    
    # AMP GradScaler for Mixed Precision training (4x speedup on RTX 4000 Ada)
    scaler = torch.amp.GradScaler('cuda') if torch.cuda.is_available() else None
    
    # ── Training ──
    print(f"\n[5/5] Training...")
    print(f"  Epochs: {args.epochs}")
    print(f"  Batch size: {args.batch_size}")
    print(f"  Total steps: {total_steps}")
    print(f"  Warmup steps: {warmup_steps}")
    print(f"  Checkpoint dir: {ckpt_dir}")
    
    os.makedirs(ckpt_dir, exist_ok=True)
    
    training_log = {
        'config': {
            **DEFAULT_CONFIG,
            'model_id': args.model_id,
            'mode': mode_str,
            'num_new_tokens': num_new_tokens,
            'num_train': len(train_pairs),
            'num_val': len(val_pairs),
            'num_test': len(test_pairs),
        },
        'epochs': []
    }
    
    best_val_loss = float('inf')
    best_wrr = 0.0
    patience_counter = 0
    
    for epoch in range(1, args.epochs + 1):
        epoch_start = time.time()
        
        # Train
        train_loss = train_one_epoch(model, train_loader, optimizer, scheduler, scaler, device, epoch, args.epochs)
        
        # Validate
        val_loss, val_wrr = validate(model, val_loader, processor, device)
        
        epoch_time = time.time() - epoch_start
        
        print(f"\n  Epoch {epoch}/{args.epochs} Summary:")
        print(f"    Train Loss: {train_loss:.4f}")
        print(f"    Val Loss:   {val_loss:.4f}")
        print(f"    Val WRR:    {val_wrr:.2f}%")
        print(f"    Time:       {epoch_time:.1f}s")
        
        epoch_data = {
            'epoch': epoch,
            'train_loss': train_loss,
            'val_loss': val_loss,
            'val_wrr': val_wrr,
            'time': epoch_time,
            'lr': scheduler.get_last_lr()[0],
        }
        training_log['epochs'].append(epoch_data)
        
        # Save best model
        improved = False
        if val_wrr > best_wrr:
            best_wrr = val_wrr
            improved = True
            best_path = os.path.join(ckpt_dir, 'best_model')
            model.save_pretrained(best_path)
            processor.save_pretrained(best_path)
            print(f"    ★ New best WRR! Saved to {best_path}")
        
        if val_loss < best_val_loss:
            best_val_loss = val_loss
            improved = True
        
        if improved:
            patience_counter = 0
        else:
            patience_counter += 1
            if patience_counter >= DEFAULT_CONFIG['patience']:
                print(f"\n  Early stopping at epoch {epoch} (no improvement for {DEFAULT_CONFIG['patience']} epochs)")
                break
        
        # Periodic checkpoint
        if epoch % DEFAULT_CONFIG['save_every_n_epochs'] == 0:
            ckpt_path = os.path.join(ckpt_dir, f'epoch_{epoch}')
            model.save_pretrained(ckpt_path)
            print(f"    Checkpoint saved: {ckpt_path}")
        
        # Save training log
        with open(log_path, 'w', encoding='utf-8') as f:
            json.dump(training_log, f, indent=2)
    
    print(f"\n{'='*60}")
    print(f"  Training Complete!")
    print(f"  Best Val WRR: {best_wrr:.2f}%")
    print(f"  Best Val Loss: {best_val_loss:.4f}")
    print(f"  Checkpoints: {ckpt_dir}")
    print(f"  Log: {log_path}")
    print(f"{'='*60}")


def run_evaluation(model, processor, test_pairs, device, ckpt_dir, mode_str):
    """Run full evaluation on the test set and save results.

    NOTE: the fine-tuned adapter must already be loaded onto ``model`` by the
    caller (handled in ``main()`` for ``--evaluate`` mode).
    """
    sys.path.insert(0, BASE_DIR)
    from metrics import evaluate_corpus, format_results_table, FPSTimer, normalize_bengali

    model.eval()
    
    predictions = []
    ground_truths = []
    
    print(f"  Running inference on {len(test_pairs)} test samples...")
    
    timer = FPSTimer()
    timer.__enter__()
    
    for i, pair in enumerate(test_pairs):
        try:
            image = Image.open(pair['image']).convert('RGB')
        except Exception:
            predictions.append('')
            ground_truths.append(pair['gt'])
            continue
        
        inputs = processor(
            images=image,
            text=TASK_PROMPT,
            return_tensors="pt",
        ).to(device)
        
        with torch.no_grad():
            generated_ids = model.generate(
                **inputs,
                max_new_tokens=DEFAULT_CONFIG['max_length'],
                num_beams=3,
            )
        
        pred = processor.batch_decode(generated_ids, skip_special_tokens=True)[0]
        predictions.append(pred.strip())
        ground_truths.append(pair['gt'])
        timer.num_images += 1
        
        if (i + 1) % 100 == 0:
            print(f"    [{i+1}/{len(test_pairs)}] processed")
    
    timer.__exit__(None, None, None)
    
    # Compute all metrics
    results = evaluate_corpus(ground_truths, predictions, verbose=True)
    results['fps'] = timer.fps
    results['mode'] = mode_str
    
    print(format_results_table(results, f"Florence-2 ({mode_str})"))
    print(f"  Inference Speed: {timer.fps:.1f} FPS")
    
    # Save results
    results_path = os.path.join(BASE_DIR, f'florence2_results_{mode_str.lower()}.json')
    save_results = {k: v for k, v in results.items() if k != 'per_sample'}
    save_results['predictions'] = [
        {'gt': gt, 'pred': pred}
        for gt, pred in zip(ground_truths, predictions)
    ]
    
    with open(results_path, 'w', encoding='utf-8') as f:
        json.dump(save_results, f, ensure_ascii=False, indent=2)
    
    print(f"  Results saved: {results_path}")


if __name__ == '__main__':
    main()
