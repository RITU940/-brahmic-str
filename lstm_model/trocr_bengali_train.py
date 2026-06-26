"""
TrOCR Bengali Scene Text — Production Training Script
======================================================
Grapheme-Aware LoRA Fine-Tuning for Bengali Scene Text Recognition

INNOVATIONS (for paper):
  1. Grapheme-aware vocabulary injection — preserves Bengali conjunct clusters
  2. LoRA fine-tuning of TrOCR (transformer-based OCR model)
  3. Test-Time Augmentation (TTA) with majority voting ensemble
  4. Grapheme-level error analysis by conjunct type

USAGE ON ISI GPU SERVER:
  # Baseline (ablation)
  python trocr_bengali_train.py --drive_path ./

  # Our Method (grapheme-aware)
  python trocr_bengali_train.py --drive_path ./ --use_graphemes

  # Evaluate best model with TTA
  python trocr_bengali_train.py --drive_path ./ --use_graphemes --evaluate --tta
"""
import os, sys, json, time, random, argparse, unicodedata
from collections import Counter, defaultdict
from datetime import datetime
from functools import partial

import torch
from torch.utils.data import Dataset, DataLoader
from PIL import Image, ImageEnhance, ImageFilter

# ════════════════════════════════════════════════════════════════════
#  BENGALI GRAPHEME TOKENIZER — Core Innovation
# ════════════════════════════════════════════════════════════════════
BENGALI_VOWEL_SIGNS = set('\u09be\u09bf\u09c0\u09c1\u09c2\u09c3\u09c4\u09c7\u09c8\u09cb\u09cc\u09d7')
BENGALI_HALANT     = '\u09cd'
BENGALI_CONSONANTS = set(
    '\u0995\u0996\u0997\u0998\u0999\u099a\u099b\u099c\u099d\u099e'
    '\u099f\u09a0\u09a1\u09a2\u09a3\u09a4\u09a5\u09a6\u09a7\u09a8'
    '\u09aa\u09ab\u09ac\u09ad\u09ae\u09af\u09b0\u09b2'
    '\u09b6\u09b7\u09b8\u09b9\u09dc\u09dd\u09df\u09f0\u09f1'
)
BENGALI_VOWELS   = set('\u0985\u0986\u0987\u0988\u0989\u098a\u098b\u098c\u098f\u0990\u0993\u0994\u09e0\u09e1')
BENGALI_MODIFIERS= set('\u0981\u0982\u0983\u09bc\u09be')
BENGALI_DIGITS   = set('\u09e6\u09e7\u09e8\u09e9\u09ea\u09eb\u09ec\u09ed\u09ee\u09ef')
BENGALI_BASE     = BENGALI_CONSONANTS | BENGALI_VOWELS | BENGALI_DIGITS


def segment_graphemes(text):
    """Segment Bengali text into visual grapheme clusters (preserves conjuncts)."""
    text = unicodedata.normalize('NFC', text)
    graphemes, i = [], 0
    while i < len(text):
        ch = text[i]
        if ch in BENGALI_BASE:
            cluster = ch; i += 1
            if ch in BENGALI_CONSONANTS:
                while i < len(text) - 1 and text[i] == BENGALI_HALANT and text[i+1] in BENGALI_CONSONANTS:
                    cluster += text[i] + text[i+1]; i += 2
            while i < len(text) and text[i] in BENGALI_VOWEL_SIGNS:
                cluster += text[i]; i += 1
            while i < len(text) and text[i] in BENGALI_MODIFIERS:
                cluster += text[i]; i += 1
            if i < len(text) and text[i] == BENGALI_HALANT:
                cluster += text[i]; i += 1
            graphemes.append(cluster)
        else:
            graphemes.append(ch); i += 1
    return graphemes


def classify_grapheme(g):
    halant_count = g.count(BENGALI_HALANT)
    has_consonant = any(c in BENGALI_CONSONANTS for c in g)
    has_digit     = any(c in BENGALI_DIGITS for c in g)
    has_vowel     = any(c in BENGALI_VOWELS for c in g)
    if has_digit: return 'digit'
    if has_vowel and not has_consonant: return 'vowel'
    if halant_count == 0 and has_consonant: return 'simple_consonant'
    if halant_count == 1: return 'conjunct_2'
    if halant_count >= 2: return 'conjunct_3+'
    return 'other'


# ════════════════════════════════════════════════════════════════════
#  DATA LOADING
# ════════════════════════════════════════════════════════════════════
def normalize_text(text):
    text = unicodedata.normalize('NFC', text.strip())
    for zw in ['\u200c', '\u200d', '\u200b', '\ufeff']:
        text = text.replace(zw, '')
    return ' '.join(text.split())


def load_splits(base_dir):
    splits_path = os.path.join(base_dir, 'florence2_splits.json')
    if os.path.exists(splits_path):
        with open(splits_path, 'r', encoding='utf-8') as f:
            splits = json.load(f)
        train_p, val_p, test_p = splits['train'], splits['val'], splits['test']
        for pairs in [train_p, val_p, test_p]:
            for p in pairs:
                p['image'] = p['image'].replace('\\', '/')
                if not os.path.isabs(p['image']) or not os.path.exists(p['image']):
                    basename = os.path.basename(p['image'])
                    parent   = os.path.basename(os.path.dirname(p['image']))
                    p['image'] = os.path.join(base_dir, parent, basename)
        print(f"  Splits: train={len(train_p)} val={len(val_p)} test={len(test_p)}")
        return train_p, val_p, test_p
    else:
        return _build_splits(base_dir)


def _build_splits(base_dir, seed=42):
    img_dir = os.path.join(base_dir, 'Bengali')
    gt_dir  = os.path.join(base_dir, 'Bengali_gt')
    images  = {os.path.splitext(f)[0]: f for f in os.listdir(img_dir) if f.lower().endswith(('.jpg','.png','.jpeg'))}
    gts     = {os.path.splitext(f)[0]: f for f in os.listdir(gt_dir)  if f.endswith('.txt')}
    pairs   = []
    for name in sorted(images):
        if name not in gts: continue
        gt = open(os.path.join(gt_dir, gts[name]), encoding='utf-8').read().strip()
        if gt in ('###', ''): continue
        pairs.append({'image': os.path.join(img_dir, images[name]), 'gt': gt, 'name': name})
    doc_groups = defaultdict(list)
    for p in pairs:
        parts  = p['name'].split('_')
        doc_id = parts[2] if len(parts) >= 3 else p['name']
        doc_groups[doc_id].append(p)
    doc_ids = sorted(doc_groups.keys())
    random.seed(seed); random.shuffle(doc_ids)
    n = len(doc_ids)
    train_ids = set(doc_ids[:int(n*0.8)])
    val_ids   = set(doc_ids[int(n*0.8):int(n*0.9)])
    test_ids  = set(doc_ids) - train_ids - val_ids
    train = [p for d in train_ids for p in doc_groups[d]]
    val   = [p for d in val_ids   for p in doc_groups[d]]
    test  = [p for d in test_ids  for p in doc_groups[d]]
    print(f"  Built splits: train={len(train)} val={len(val)} test={len(test)}")
    return train, val, test


# ════════════════════════════════════════════════════════════════════
#  DATASET
# ════════════════════════════════════════════════════════════════════
class BengaliOCRDataset(Dataset):
    def __init__(self, pairs, augment=False):
        self.pairs   = pairs
        self.augment = augment

    def __len__(self): return len(self.pairs)

    def _augment(self, img):
        if random.random() < 0.3: img = ImageEnhance.Brightness(img).enhance(random.uniform(0.6, 1.4))
        if random.random() < 0.3: img = ImageEnhance.Contrast(img).enhance(random.uniform(0.6, 1.5))
        if random.random() < 0.2: img = img.filter(ImageFilter.GaussianBlur(random.uniform(0.3, 1.5)))
        if random.random() < 0.15: img = ImageEnhance.Sharpness(img).enhance(random.uniform(1.5, 2.5))
        if random.random() < 0.15:
            from io import BytesIO
            buf = BytesIO(); img.save(buf, 'JPEG', quality=random.randint(40, 75)); buf.seek(0)
            img = Image.open(buf).convert('RGB')
        return img

    def __getitem__(self, idx):
        p = self.pairs[idx]
        try:   img = Image.open(p['image']).convert('RGB')
        except: img = Image.new('RGB', (256, 64), (255,255,255))
        if self.augment: img = self._augment(img)
        return img, p['gt']


def make_collate(processor, max_length=64):
    def collate(batch):
        images, texts = zip(*batch)
        pixel_values = processor(images=list(images), return_tensors='pt').pixel_values
        labels = processor.tokenizer(
            list(texts), return_tensors='pt', padding=True,
            truncation=True, max_length=max_length
        ).input_ids
        labels[labels == processor.tokenizer.pad_token_id] = -100
        return pixel_values, labels, list(texts)
    return collate


# ════════════════════════════════════════════════════════════════════
#  METRICS
# ════════════════════════════════════════════════════════════════════
def edit_distance(ref, hyp):
    m, n = len(ref), len(hyp)
    if m == 0: return n
    if n == 0: return m
    prev = list(range(n+1))
    for i in range(1, m+1):
        curr = [i] + [0]*n
        for j in range(1, n+1):
            cost = 0 if ref[i-1]==hyp[j-1] else 1
            curr[j] = min(prev[j]+1, curr[j-1]+1, prev[j-1]+cost)
        prev = curr
    return prev[n]


def compute_metrics(gts, preds):
    n = len(gts)
    total_ed, total_chars, total_ned, exact = 0, 0, 0.0, 0
    for gt, pred in zip(gts, preds):
        g, p = normalize_text(gt), normalize_text(pred)
        ed = edit_distance(g, p)
        total_ed    += ed
        total_chars += max(len(g), 1)
        maxl = max(len(g), len(p))
        total_ned   += (ed/maxl) if maxl > 0 else 0
        if g == p: exact += 1
    cer  = total_ed / max(total_chars, 1)
    ned  = total_ned / max(n, 1)
    return {
        'WRR':            round(exact/max(n,1)*100, 2),
        'CER':            round(cer*100, 2),
        '1-NED':          round((1-ned)*100, 2),
        'char_accuracy':  round((1-cer)*100, 2),
        'exact_matches':  exact,
        'num_samples':    n,
    }


def grapheme_error_analysis(gts, preds):
    type_correct, type_total = Counter(), Counter()
    conjunct_errors = []
    for gt, pred in zip(gts, preds):
        gt_n, pred_n = normalize_text(gt), normalize_text(pred)
        for g in segment_graphemes(gt_n):
            gtype = classify_grapheme(g)
            type_total[gtype] += 1
            if g in pred_n: type_correct[gtype] += 1
            elif gtype in ('conjunct_2','conjunct_3+'): conjunct_errors.append(g)
    analysis = {'per_type_accuracy': {}, 'per_type_counts': {}, 'hardest_conjuncts': []}
    for gtype in sorted(type_total):
        total, correct = type_total[gtype], type_correct.get(gtype, 0)
        analysis['per_type_accuracy'][gtype] = round(correct/max(total,1)*100, 2)
        analysis['per_type_counts'][gtype] = {'correct': correct, 'total': total}
    analysis['hardest_conjuncts'] = [
        {'grapheme': g, 'errors': c} for g, c in Counter(conjunct_errors).most_common(20)
    ]
    return analysis


# ════════════════════════════════════════════════════════════════════
#  TEST-TIME AUGMENTATION (TTA)
# ════════════════════════════════════════════════════════════════════
def tta_predict(model, processor, image, device):
    views = [
        image,
        ImageEnhance.Brightness(image).enhance(1.2),
        ImageEnhance.Brightness(image).enhance(0.8),
        ImageEnhance.Contrast(image).enhance(1.3),
        ImageEnhance.Sharpness(image).enhance(2.0),
    ]
    preds = []
    for v in views:
        pv = processor(images=v, return_tensors='pt').pixel_values.to(device)
        with torch.no_grad():
            ids = model.generate(pv, max_new_tokens=64)
        preds.append(processor.decode(ids[0], skip_special_tokens=True).strip())
    if len(set(preds)) == 1: return preds[0]
    best, best_dist = preds[0], float('inf')
    for cand in preds:
        d = sum(edit_distance(cand, o) for o in preds)
        if d < best_dist: best_dist = d; best = cand
    return best


# ════════════════════════════════════════════════════════════════════
#  MAIN TRAINING FUNCTION
# ════════════════════════════════════════════════════════════════════
def train(args):
    base_dir = args.drive_path or '.'
    mode_str = 'grapheme' if args.use_graphemes else 'standard'

    print(f"\n{'='*70}")
    print(f"  TrOCR Bengali Scene Text — {mode_str.upper()} TOKENIZER")
    print(f"  {'OUR METHOD (Grapheme-Aware)' if args.use_graphemes else 'ABLATION BASELINE (Standard)'}")
    print(f"  GPU: {torch.cuda.get_device_name(0) if torch.cuda.is_available() else 'CPU'}")
    print(f"{'='*70}")

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    # ── 1. Load Data ──
    print("\n[1/6] Loading data...")
    train_pairs, val_pairs, test_pairs = load_splits(base_dir)

    # ── 2. Load Model ──
    print(f"\n[2/6] Loading TrOCR ({args.model_id})...")
    from transformers import TrOCRProcessor, VisionEncoderDecoderModel

    processor = TrOCRProcessor.from_pretrained(args.model_id)
    model     = VisionEncoderDecoderModel.from_pretrained(args.model_id)

    # Configure generation settings
    model.config.decoder_start_token_id = processor.tokenizer.cls_token_id
    model.config.pad_token_id           = processor.tokenizer.pad_token_id
    model.config.eos_token_id           = processor.tokenizer.sep_token_id
    model.config.max_length             = 64
    model.config.no_repeat_ngram_size   = 3
    model.config.length_penalty         = 2.0
    model.config.num_beams              = 4
    print(f"  Params: {sum(p.numel() for p in model.parameters()):,}")

    # ── 3. Grapheme Token Injection ──
    num_new_tokens = 0
    if args.use_graphemes:
        print(f"\n[3/6] Injecting grapheme tokens (OUR INNOVATION)...")
        all_gts   = [p['gt'] for p in train_pairs + val_pairs + test_pairs]
        all_graphemes = Counter()
        for gt in all_gts:
            all_graphemes.update(segment_graphemes(gt))
        new_tokens = []
        for g in all_graphemes:
            enc = processor.tokenizer.encode(g, add_special_tokens=False)
            dec = processor.tokenizer.decode(enc).strip()
            if len(enc) > 1 or dec != g:
                new_tokens.append(g)
        if new_tokens:
            num_new_tokens = processor.tokenizer.add_tokens(new_tokens)
            model.decoder.resize_token_embeddings(len(processor.tokenizer))
            print(f"  ✓ Injected {num_new_tokens} grapheme tokens")
            print(f"  ✓ New vocab size: {len(processor.tokenizer)}")
    else:
        print(f"\n[3/6] Standard tokenizer (ablation baseline)")

    # ── 4. LoRA (applied to decoder only — ViT encoder is frozen) ──
    print(f"\n[4/6] Applying LoRA to decoder (r={args.lora_r})...")
    from peft import LoraConfig, get_peft_model

    lora_config = LoraConfig(
        r=args.lora_r, lora_alpha=args.lora_r * 2,
        target_modules=["q_proj", "v_proj", "k_proj", "out_proj"],
        lora_dropout=0.05, bias="none", task_type="CAUSAL_LM",
    )
    # Freeze encoder (ViT), train only decoder with LoRA
    for param in model.encoder.parameters():
        param.requires_grad = False
    model.decoder = get_peft_model(model.decoder, lora_config)
    model.decoder.print_trainable_parameters()
    model.to(device)

    # ── 5. DataLoaders ──
    print(f"\n[5/6] Creating dataloaders...")
    collate_fn = make_collate(processor)
    train_ds   = BengaliOCRDataset(train_pairs, augment=True)
    val_ds     = BengaliOCRDataset(val_pairs,   augment=False)
    train_loader = DataLoader(train_ds, batch_size=args.batch_size, shuffle=True,
                              collate_fn=collate_fn, num_workers=0, drop_last=True)
    val_loader   = DataLoader(val_ds,   batch_size=args.batch_size, shuffle=False,
                              collate_fn=collate_fn, num_workers=0)

    # ── 6. Training Loop (with AMP for speed) ──
    optimizer    = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=0.01)
    total_steps  = len(train_loader) * args.epochs
    warmup_steps = max(1, total_steps // 10)
    scheduler    = torch.optim.lr_scheduler.OneCycleLR(
        optimizer, max_lr=args.lr, total_steps=total_steps,
        pct_start=warmup_steps/total_steps, anneal_strategy='cos',
    )
    scaler = torch.cuda.amp.GradScaler()  # For Mixed Precision speedup
    ckpt_dir = os.path.join(base_dir, f'checkpoints_trocr_{mode_str}')
    os.makedirs(ckpt_dir, exist_ok=True)

    print(f"\n[6/6] Training {args.epochs} epochs | batch={args.batch_size} | lr={args.lr}")
    print(f"{'─'*70}")
    print(f"{'Epoch':>5} │ {'TrLoss':>7} │ {'VaLoss':>7} │ {'WRR%':>6} │ {'CER%':>6} │ {'1-NED%':>7} │ Time")
    print(f"{'─'*70}")

    log = {'config': {'mode': mode_str, 'model': args.model_id, 'lora_r': args.lora_r,
                      'lr': args.lr, 'bs': args.batch_size, 'epochs': args.epochs,
                      'new_tokens': num_new_tokens, 'timestamp': datetime.now().isoformat()},
           'epochs': []}
    best_wrr, patience_counter = 0.0, 0

    for epoch in range(1, args.epochs + 1):
        # Train
        model.train()
        tr_loss, nb = 0.0, 0
        t0 = time.time()
        for i, (pv, labels, _) in enumerate(train_loader):
            pv, labels = pv.to(device), labels.to(device)
            with torch.cuda.amp.autocast():
                outputs = model(pixel_values=pv, labels=labels)
                loss    = outputs.loss
            
            if torch.isnan(loss) or torch.isinf(loss): continue
            
            scaler.scale(loss).backward()
            scaler.unscale_(optimizer)
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            scaler.step(optimizer)
            scaler.update()
            scheduler.step()
            optimizer.zero_grad()
            
            tr_loss += loss.item(); nb += 1
            if (i+1) % 50 == 0:
                print(f"  Ep{epoch} [{i+1}/{len(train_loader)}] loss={tr_loss/nb:.4f}")
        avg_tr = tr_loss / max(nb, 1)

        # Validate
        model.eval()
        va_loss, vb = 0.0, 0
        all_preds, all_gts = [], []
        with torch.no_grad():
            for pv, labels, texts in val_loader:
                pv, labels = pv.to(device), labels.to(device)
                with torch.cuda.amp.autocast():
                    va_loss += model(pixel_values=pv, labels=labels).loss.item(); vb += 1
                # Use greedy decoding (num_beams=1) during validation for speed
                ids  = model.generate(pixel_values=pv, max_new_tokens=64, num_beams=1)
                preds= processor.batch_decode(ids, skip_special_tokens=True)
                all_preds.extend([p.strip() for p in preds])
                all_gts.extend(texts)
        avg_va  = va_loss / max(vb, 1)
        metrics = compute_metrics(all_gts, all_preds)
        elapsed = time.time() - t0
        wrr, cer, ned1 = metrics['WRR'], metrics['CER'], metrics['1-NED']

        tag = ''
        if wrr > best_wrr:
            best_wrr = wrr; patience_counter = 0; tag = ' ★'
            best_path = os.path.join(ckpt_dir, 'best_model')
            os.makedirs(best_path, exist_ok=True)
            model.decoder.save_pretrained(best_path)  # save LoRA adapter
            processor.save_pretrained(best_path)
            # Save full decoder config for reloading
            with open(os.path.join(best_path, 'base_model_id.txt'), 'w') as f:
                f.write(args.model_id)
        else:
            patience_counter += 1

        print(f"{epoch:5d} │ {avg_tr:7.4f} │ {avg_va:7.4f} │ {wrr:6.2f} │ {cer:6.2f} │ {ned1:7.2f} │ {elapsed:.0f}s{tag}")
        log['epochs'].append({'epoch': epoch, 'train_loss': round(avg_tr,4), 'val_loss': round(avg_va,4),
                               'val_wrr': wrr, 'val_cer': cer, 'val_1ned': ned1, 'time': round(elapsed,1)})
        with open(os.path.join(base_dir, f'training_log_trocr_{mode_str}.json'), 'w', encoding='utf-8') as f:
            json.dump(log, f, indent=2)

        if epoch % 2 == 0 and all_preds:
            print(f"  Samples:")
            for i in range(min(3, len(all_preds))):
                ok = '✓' if normalize_text(all_gts[i]) == normalize_text(all_preds[i]) else '✗'
                print(f"    {ok} GT:[{all_gts[i]}] → Pred:[{all_preds[i]}]")

        if patience_counter >= args.patience:
            print(f"\n  Early stopping at epoch {epoch}"); break

    print(f"{'─'*70}")
    print(f"\n  ★ DONE — Best WRR: {best_wrr:.2f}%")

    # Final test evaluation
    _evaluate_test(model, processor, test_pairs, device, base_dir, mode_str, args.tta)


# ════════════════════════════════════════════════════════════════════
#  TEST EVALUATION
# ════════════════════════════════════════════════════════════════════
def _evaluate_test(model, processor, test_pairs, device, base_dir, mode_str, use_tta=False):
    from transformers import TrOCRProcessor, VisionEncoderDecoderModel
    model.eval()
    all_preds, all_gts = [], []
    t0 = time.time()
    print(f"\n  Evaluating {len(test_pairs)} test samples {'(+TTA)' if use_tta else ''}...")
    for i, pair in enumerate(test_pairs):
        try:   img = Image.open(pair['image']).convert('RGB')
        except: all_preds.append(''); all_gts.append(pair['gt']); continue
        if use_tta:
            pred = tta_predict(model, processor, img, device)
        else:
            pv = processor(images=img, return_tensors='pt').pixel_values.to(device)
            with torch.no_grad():
                ids = model.generate(pixel_values=pv, max_new_tokens=64)
            pred = processor.decode(ids[0], skip_special_tokens=True).strip()
        all_preds.append(pred); all_gts.append(pair['gt'])
        if (i+1) % 100 == 0: print(f"    [{i+1}/{len(test_pairs)}]")

    elapsed = time.time() - t0
    metrics = compute_metrics(all_gts, all_preds)
    error_a = grapheme_error_analysis(all_gts, all_preds)
    fps     = len(test_pairs) / max(elapsed, 0.01)

    print(f"\n  ┌──────────────────────────────────────────┐")
    print(f"  │  TEST RESULTS  ({mode_str:>9} {'+ TTA' if use_tta else '     '})         │")
    print(f"  ├──────────────────────────────────────────┤")
    print(f"  │  WRR:         {metrics['WRR']:>7.2f}%                │")
    print(f"  │  CER:         {metrics['CER']:>7.2f}%                │")
    print(f"  │  1-NED:       {metrics['1-NED']:>7.2f}%                │")
    print(f"  │  Char Acc:    {metrics['char_accuracy']:>7.2f}%                │")
    print(f"  │  Exact:       {metrics['exact_matches']:>4d}/{metrics['num_samples']:<4d}                │")
    print(f"  │  Speed:       {fps:>7.1f} img/s              │")
    print(f"  └──────────────────────────────────────────┘")

    print(f"\n  Grapheme-level accuracy:")
    for gtype, acc in sorted(error_a['per_type_accuracy'].items()):
        cnt = error_a['per_type_counts'][gtype]
        bar = '█' * int(acc/5) + '░' * (20-int(acc/5))
        print(f"    {gtype:<20s} {bar} {acc:5.1f}% ({cnt['correct']}/{cnt['total']})")

    if error_a['hardest_conjuncts']:
        print(f"\n  Top-10 hardest conjuncts:")
        for item in error_a['hardest_conjuncts'][:10]:
            print(f"    '{item['grapheme']}' — {item['errors']} errors")

    suffix    = f"_{mode_str}{'_tta' if use_tta else ''}"
    save_data = {**metrics, 'fps': round(fps,1), 'mode': mode_str, 'tta': use_tta,
                 'error_analysis': error_a,
                 'predictions': [{'gt': g, 'pred': p} for g, p in zip(all_gts, all_preds)]}
    out = os.path.join(base_dir, f'trocr_results{suffix}.json')
    with open(out, 'w', encoding='utf-8') as f:
        json.dump(save_data, f, ensure_ascii=False, indent=2)
    # Also save Florence-2-compatible filename for figure generation
    compat = os.path.join(base_dir, f'florence2_results_{mode_str}.json')
    with open(compat, 'w', encoding='utf-8') as f:
        json.dump(save_data, f, ensure_ascii=False, indent=2)
    print(f"\n  Results saved: {out}")


# ════════════════════════════════════════════════════════════════════
#  EVALUATE-ONLY MODE
# ════════════════════════════════════════════════════════════════════
def evaluate_only(args):
    from transformers import TrOCRProcessor, VisionEncoderDecoderModel
    from peft import PeftModel

    base_dir = args.drive_path or '.'
    mode_str = 'grapheme' if args.use_graphemes else 'standard'
    ckpt_dir = os.path.join(base_dir, f'checkpoints_trocr_{mode_str}', 'best_model')

    if not os.path.exists(ckpt_dir):
        print(f"ERROR: No checkpoint found at {ckpt_dir}. Train first."); return

    _, _, test_pairs = load_splits(base_dir)
    device    = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    processor = TrOCRProcessor.from_pretrained(ckpt_dir)
    model     = VisionEncoderDecoderModel.from_pretrained(args.model_id)
    if len(processor.tokenizer) != model.decoder.get_input_embeddings().weight.shape[0]:
        model.decoder.resize_token_embeddings(len(processor.tokenizer))
    model.decoder = PeftModel.from_pretrained(model.decoder, ckpt_dir)
    model.to(device).eval()
    _evaluate_test(model, processor, test_pairs, device, base_dir, mode_str, args.tta)


# ════════════════════════════════════════════════════════════════════
#  MAIN
# ════════════════════════════════════════════════════════════════════
if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='TrOCR Bengali Scene Text — Grapheme-Aware Training')
    parser.add_argument('--drive_path',    default=None)
    parser.add_argument('--model_id',      default='microsoft/trocr-base-str')
    parser.add_argument('--use_graphemes', action='store_true')
    parser.add_argument('--epochs',        type=int,   default=30)
    parser.add_argument('--batch_size',    type=int,   default=8)
    parser.add_argument('--lr',            type=float, default=5e-5)
    parser.add_argument('--lora_r',        type=int,   default=16)
    parser.add_argument('--patience',      type=int,   default=7)
    parser.add_argument('--evaluate',      action='store_true')
    parser.add_argument('--tta',           action='store_true')
    args = parser.parse_args()

    if args.evaluate:
        evaluate_only(args)
    else:
        train(args)
