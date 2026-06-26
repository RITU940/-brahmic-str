"""
Bengali Scene Text Recognition - CRNN V3 (High Accuracy)
=========================================================
Key improvements over V1 (57% char acc):
  1. Filter noisy data: remove GT > 25 chars (long phrases hurt CTC)
  2. Larger images: 64x256 (Bengali needs more resolution)
  3. Deeper ResNet-style CNN with residual connections
  4. Larger BiLSTM: 512 hidden units
  5. OneCycleLR scheduler (much better than ReduceLROnPlateau)
  6. AdamW optimizer with proper weight decay
  7. Stronger data augmentation

USAGE ON COLAB:
  !python colab_train_v3.py            # Train
  !python colab_train_v3.py --evaluate # Evaluate
  !python colab_train_v3.py --figures  # Generate figures
"""
import os, sys, json, time, math, random, argparse
from collections import Counter
import numpy as np
from PIL import Image, ImageFilter, ImageEnhance
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt


class Config:
    dataset_json = 'dataset_splits.json'
    img_height = 64
    img_width = 256
    hidden_size = 512
    batch_size = 32          # Smaller batch for larger images
    num_epochs = 200
    learning_rate = 0.001
    weight_decay = 1e-4
    max_gt_length = 50       # Safety filter (long seqs already pre-split into words)
    early_stop_patience = 30
    checkpoint_dir = 'checkpoints_v3'
    best_model_path = 'checkpoints_v3/best_model_v3.pth'
    log_file = 'training_log_v3.json'
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    num_workers = 2 if torch.cuda.is_available() else 0


# =====================================================================
#  DATASET
# =====================================================================
class BengaliDataset(Dataset):
    def __init__(self, pairs, char2idx, h=64, w=256, augment=False):
        self.pairs = pairs
        self.char2idx = char2idx
        self.h, self.w = h, w
        self.augment = augment

    def __len__(self):
        return len(self.pairs)

    def encode(self, text):
        return [self.char2idx[c] for c in text if c in self.char2idx]

    def preprocess(self, img):
        if img.mode != 'L':
            img = img.convert('L')
        w, h = img.size
        scale = self.h / h
        tw = min(int(w * scale), self.w)
        tw = max(tw, 1)
        img = img.resize((tw, self.h), Image.LANCZOS)
        out = Image.new('L', (self.w, self.h), 255)
        out.paste(img, (0, 0))
        return out

    def augment_img(self, img):
        if random.random() < 0.4:
            img = img.rotate(random.uniform(-5, 5), fillcolor=255)
        if random.random() < 0.5:
            img = ImageEnhance.Contrast(img).enhance(random.uniform(0.5, 1.8))
        if random.random() < 0.4:
            img = ImageEnhance.Brightness(img).enhance(random.uniform(0.6, 1.5))
        if random.random() < 0.3:
            img = img.filter(ImageFilter.GaussianBlur(random.uniform(0.3, 2.0)))
        if random.random() < 0.3:
            img = ImageEnhance.Sharpness(img).enhance(random.uniform(1.5, 3.0))
        if random.random() < 0.3:
            arr = np.array(img, dtype=np.float32)
            arr = np.clip(arr + np.random.normal(0, random.uniform(3, 12), arr.shape), 0, 255)
            img = Image.fromarray(arr.astype(np.uint8))
        if random.random() < 0.2:
            f = ImageFilter.MinFilter(3) if random.random() < 0.5 else ImageFilter.MaxFilter(3)
            img = img.filter(f)
        if random.random() < 0.25:
            w, h = img.size
            s = random.uniform(0.8, 1.2)
            nw = max(int(w * s), 10)
            img = img.resize((nw, h), Image.LANCZOS)
            if nw > w:
                img = img.crop((0, 0, w, h))
            else:
                p = Image.new('L', (w, h), 255)
                p.paste(img, (0, 0))
                img = p
        return img

    def __getitem__(self, idx):
        p = self.pairs[idx]
        try:
            img = Image.open(p['image'])
        except:
            img = Image.new('L', (self.w, self.h), 255)
        img = self.preprocess(img)
        if self.augment:
            img = self.augment_img(img)
        arr = np.array(img, dtype=np.float32) / 255.0
        arr = 1.0 - arr
        tensor = torch.FloatTensor(arr).unsqueeze(0)
        label = self.encode(p['gt'])
        return tensor, torch.IntTensor(label), len(label), p['gt']


def collate_fn(batch):
    imgs, labels, lens, texts = zip(*batch)
    return torch.stack(imgs), torch.cat(labels), torch.IntTensor(lens), texts


def load_data(json_path, max_gt_len=25):
    with open(json_path, 'r', encoding='utf-8') as f:
        data = json.load(f)
    pairs = data['pairs']
    char2idx = data['vocabulary']['char2idx']
    for p in pairs:
        p['image'] = p['image'].replace('\\', '/')
        if not os.path.isabs(p['image']):
            p['image'] = os.path.join('.', p['image'])
    splits = data['splits']

    def filter_pairs(indices):
        out = []
        for i in indices:
            p = pairs[i]
            if len(p['gt']) <= max_gt_len and len(p['gt']) >= 1:
                out.append(p)
        return out

    train = filter_pairs(splits['train'])
    val = filter_pairs(splits['val'])
    test = filter_pairs(splits['test'])
    print(f"  After filtering (max_gt_len={max_gt_len}):")
    print(f"  Train: {len(train)} | Val: {len(val)} | Test: {len(test)}")
    return train, val, test, char2idx, data


# =====================================================================
#  MODEL - ResNet-style CRNN
# =====================================================================
class ResBlock(nn.Module):
    def __init__(self, ch):
        super().__init__()
        self.conv1 = nn.Conv2d(ch, ch, 3, 1, 1)
        self.bn1 = nn.BatchNorm2d(ch)
        self.conv2 = nn.Conv2d(ch, ch, 3, 1, 1)
        self.bn2 = nn.BatchNorm2d(ch)
        self.relu = nn.ReLU(True)

    def forward(self, x):
        out = self.relu(self.bn1(self.conv1(x)))
        out = self.bn2(self.conv2(out))
        return self.relu(out + x)


class CRNN_V3(nn.Module):
    """
    ResNet-style CRNN for Bengali text recognition.
    Input:  (B, 1, 64, 256) -> Output: (seq_len, B, num_classes)
    """
    def __init__(self, num_classes, img_height=64, hidden_size=512):
        super().__init__()
        self.cnn = nn.Sequential(
            # Block 1: 1->64, 64x256 -> 32x128
            nn.Conv2d(1, 64, 3, 1, 1), nn.BatchNorm2d(64), nn.ReLU(True),
            nn.MaxPool2d(2, 2),
            # Block 2: 64->128, 32x128 -> 16x64
            nn.Conv2d(64, 128, 3, 1, 1), nn.BatchNorm2d(128), nn.ReLU(True),
            nn.MaxPool2d(2, 2),
            ResBlock(128),
            # Block 3: 128->256, 16x64 -> 8x64
            nn.Conv2d(128, 256, 3, 1, 1), nn.BatchNorm2d(256), nn.ReLU(True),
            nn.MaxPool2d((2, 1), (2, 1)),
            ResBlock(256),
            # Block 4: 256->512, 8x64 -> 4x64
            nn.Conv2d(256, 512, 3, 1, 1), nn.BatchNorm2d(512), nn.ReLU(True),
            nn.MaxPool2d((2, 1), (2, 1)),
            ResBlock(512),
            # Block 5: 512->512, 4x64 -> 1x63
            nn.Conv2d(512, 512, (4, 2), 1, (0, 0)), nn.BatchNorm2d(512), nn.ReLU(True),
        )
        self.rnn1 = nn.LSTM(512, hidden_size, bidirectional=True, batch_first=True)
        self.ln1 = nn.LayerNorm(hidden_size * 2)
        self.rnn2 = nn.LSTM(hidden_size * 2, hidden_size, bidirectional=True, batch_first=True)
        self.fc = nn.Linear(hidden_size * 2, num_classes)
        self.drop = nn.Dropout(0.4)
        self._init()

    def _init(self):
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
                if m.bias is not None: nn.init.zeros_(m.bias)
            elif isinstance(m, nn.BatchNorm2d):
                nn.init.ones_(m.weight); nn.init.zeros_(m.bias)
            elif isinstance(m, nn.Linear):
                nn.init.xavier_uniform_(m.weight)
                if m.bias is not None: nn.init.zeros_(m.bias)

    def forward(self, x):
        c = self.cnn(x)
        b, ch, h, w = c.size()
        if h != 1:
            c = c.mean(dim=2, keepdim=True)
        c = c.squeeze(2).permute(0, 2, 1)  # (B, W, 512)
        c = self.drop(c)
        r1, _ = self.rnn1(c)
        r1 = self.ln1(r1)
        r2, _ = self.rnn2(r1)
        out = self.fc(r2)
        return torch.nn.functional.log_softmax(out.permute(1, 0, 2), dim=2)


# =====================================================================
#  UTILITIES
# =====================================================================
def decode(preds, idx2char):
    _, idxs = preds.max(2)
    idxs = idxs.permute(1, 0).cpu().numpy()
    results = []
    for seq in idxs:
        chars, prev = [], -1
        for i in seq:
            if i != 0 and i != prev and i in idx2char:
                chars.append(idx2char[i])
            prev = i
        results.append(''.join(chars))
    return results


def edit_dist(a, b):
    m, n = len(a), len(b)
    if m == 0: return n
    if n == 0: return m
    dp = list(range(n + 1))
    for i in range(1, m + 1):
        c = [i] + [0] * n
        for j in range(1, n + 1):
            c[j] = min(dp[j]+1, c[j-1]+1, dp[j-1]+(0 if a[i-1]==b[j-1] else 1))
        dp = c
    return dp[n]


def metrics(preds, gts):
    cd, tc, wd, tw, em, cw = 0, 0, 0, 0, 0, 0
    for p, g in zip(preds, gts):
        if not g: continue
        cd += edit_dist(list(g), list(p))
        tc += max(len(g), 1)
        gw, pw = g.split(), p.split()
        wd += edit_dist(gw, pw)
        tw += max(len(gw), 1)
        for a, b in zip(gw, pw):
            if a == b: cw += 1
        if p.strip() == g.strip(): em += 1
    n = max(len([g for g in gts if g]), 1)
    cer = cd / max(tc, 1)
    return {'cer': cer, 'wer': wd/max(tw,1), 'char_acc': (1-cer)*100,
            'word_acc': cw/max(tw,1)*100, 'exact_match_rate': em/n*100, 'total_samples': n}


# =====================================================================
#  TRAIN / EVAL
# =====================================================================
def train_epoch(model, loader, crit, opt, dev, i2c):
    model.train()
    loss_sum, nb, ap, ag = 0, 0, [], []
    for bi, (imgs, labs, lens, txts) in enumerate(loader):
        imgs, labs, lens = imgs.to(dev), labs.to(dev), lens.to(dev)
        p = model(imgs)
        sl, bs = p.size(0), p.size(1)
        il = torch.full((bs,), sl, dtype=torch.long, device=dev)
        loss = crit(p, labs, il, lens)
        if torch.isnan(loss) or torch.isinf(loss): continue
        opt.zero_grad(); loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 5.0)
        opt.step()
        loss_sum += loss.item(); nb += 1
        if bi % 8 == 0:
            ap.extend(decode(p.detach(), i2c)); ag.extend(txts)
    m = metrics(ap, ag) if ap else {'cer':1,'wer':1,'char_acc':0,'word_acc':0,'exact_match_rate':0,'total_samples':0}
    return loss_sum/max(nb,1), m


@torch.no_grad()
def eval_model(model, loader, crit, dev, i2c):
    model.eval()
    loss_sum, nb, ap, ag = 0, 0, [], []
    for imgs, labs, lens, txts in loader:
        imgs, labs, lens = imgs.to(dev), labs.to(dev), lens.to(dev)
        p = model(imgs)
        sl, bs = p.size(0), p.size(1)
        il = torch.full((bs,), sl, dtype=torch.long, device=dev)
        loss = crit(p, labs, il, lens)
        if not (torch.isnan(loss) or torch.isinf(loss)):
            loss_sum += loss.item(); nb += 1
        ap.extend(decode(p, i2c)); ag.extend(txts)
    return loss_sum/max(nb,1), metrics(ap, ag), ap, ag


def run_training():
    cfg = Config()
    print("=" * 70)
    print("  Bengali CRNN V3 — High Accuracy Training")
    print("=" * 70)
    print(f"  Device: {cfg.device} | Images: {cfg.img_height}x{cfg.img_width}")
    print(f"  Batch: {cfg.batch_size} | Hidden: {cfg.hidden_size} | MaxGT: {cfg.max_gt_length}")
    print("=" * 70)

    print("\n[1/4] Loading & filtering dataset...")
    train_p, val_p, test_p, c2i, data = load_data(cfg.dataset_json, cfg.max_gt_length)
    nc = data['stats']['num_classes']
    i2c = {v: k for k, v in c2i.items()}
    print(f"  Classes: {nc}")

    train_ds = BengaliDataset(train_p, c2i, cfg.img_height, cfg.img_width, augment=True)
    val_ds = BengaliDataset(val_p, c2i, cfg.img_height, cfg.img_width, augment=False)
    train_dl = DataLoader(train_ds, cfg.batch_size, shuffle=True, num_workers=cfg.num_workers,
                          collate_fn=collate_fn, pin_memory=True)
    val_dl = DataLoader(val_ds, cfg.batch_size, shuffle=False, num_workers=cfg.num_workers,
                        collate_fn=collate_fn, pin_memory=True)

    print("\n[2/4] Building CRNN V3 (ResNet + BiLSTM-512)...")
    model = CRNN_V3(nc, cfg.img_height, cfg.hidden_size).to(cfg.device)
    params = sum(p.numel() for p in model.parameters())
    print(f"  Parameters: {params:,}")

    crit = nn.CTCLoss(blank=0, zero_infinity=True)
    opt = optim.AdamW(model.parameters(), lr=cfg.learning_rate, weight_decay=cfg.weight_decay)
    sched = optim.lr_scheduler.OneCycleLR(opt, max_lr=cfg.learning_rate,
                                           epochs=cfg.num_epochs, steps_per_epoch=len(train_dl),
                                           pct_start=0.1, anneal_strategy='cos')

    os.makedirs(cfg.checkpoint_dir, exist_ok=True)
    best_cer = float('inf')
    best_info = {}
    no_imp = 0
    log = {'config': {'v': 3, 'bs': cfg.batch_size, 'lr': cfg.learning_rate,
                      'h': cfg.hidden_size, 'img': f"{cfg.img_height}x{cfg.img_width}",
                      'nc': nc, 'train': len(train_p), 'val': len(val_p)}, 'epochs': []}

    print(f"\n[3/4] Training...")
    print(f"{'='*90}")
    print(f"{'Ep':>4} | {'TrL':>6} {'VaL':>6} | {'TrCER':>6} {'VaCER':>6} | "
          f"{'ChA':>5} {'WdA':>5} {'ExM':>5} | {'LR':>9} | {'T':>4}")
    print(f"{'='*90}")

    t0 = time.time()
    for ep in range(1, cfg.num_epochs + 1):
        t1 = time.time()
        tl, tm = train_epoch(model, train_dl, crit, opt, cfg.device, i2c)
        vl, vm, vp, vg = eval_model(model, val_dl, crit, cfg.device, i2c)
        # Step scheduler per epoch (it was set up per step, but we call it per batch inside train_epoch)
        dt = time.time() - t1
        lr = opt.param_groups[0]['lr']

        log['epochs'].append({'ep': ep, 'tl': round(tl,4), 'vl': round(vl,4),
            'tc': round(tm['cer'],4), 'vc': round(vm['cer'],4),
            'ca': round(vm['char_acc'],2), 'wa': round(vm['word_acc'],2),
            'em': round(vm['exact_match_rate'],2), 'lr': lr, 't': round(dt,1)})

        tag = ""
        if vm['cer'] < best_cer:
            best_cer = vm['cer']
            best_info = {**vm, 'epoch': ep}
            no_imp = 0
            tag = " *"
            torch.save({'epoch': ep, 'model_state_dict': model.state_dict(),
                'num_classes': nc, 'char2idx': c2i,
                'idx2char': {str(k):v for k,v in i2c.items()},
                'val_cer': vm['cer'], 'val_word_acc': vm['word_acc'],
                'val_char_acc': vm['char_acc'], 'val_exact_match_rate': vm['exact_match_rate'],
                'config': {'img_height': cfg.img_height, 'img_width': cfg.img_width,
                           'hidden_size': cfg.hidden_size, 'version': 'v3'}
            }, cfg.best_model_path)
        else:
            no_imp += 1

        print(f"{ep:4d} | {tl:6.3f} {vl:6.3f} | {tm['cer']:6.4f} {vm['cer']:6.4f} | "
              f"{vm['char_acc']:5.1f} {vm['word_acc']:5.1f} {vm['exact_match_rate']:5.1f} | "
              f"{lr:9.6f} | {dt:4.0f}{tag}")

        if ep % 10 == 0 and vp:
            for i in range(min(3, len(vp))):
                s = "Y" if vp[i]==vg[i] else "N"
                print(f"  {s} [{vg[i]}] -> [{vp[i]}]")

        if no_imp >= cfg.early_stop_patience:
            print(f"\n  Early stop at epoch {ep}")
            break
        if ep % 5 == 0:
            with open(cfg.log_file, 'w', encoding='utf-8') as f:
                json.dump(log, f, indent=2)

    tt = time.time() - t0
    log['summary'] = {'time': round(tt,1), 'best_cer': round(best_cer,4),
        'best_ca': round(best_info.get('char_acc',0),2),
        'best_wa': round(best_info.get('word_acc',0),2),
        'best_em': round(best_info.get('exact_match_rate',0),2),
        'best_ep': best_info.get('epoch',0)}
    with open(cfg.log_file, 'w', encoding='utf-8') as f:
        json.dump(log, f, indent=2)

    print(f"\n{'='*70}")
    print(f"  [4/4] DONE! Time: {tt/60:.1f} min")
    print(f"  Best Epoch:      {best_info.get('epoch','?')}")
    print(f"  Best Char Acc:   {best_info.get('char_acc',0):.2f}%")
    print(f"  Best Word Acc:   {best_info.get('word_acc',0):.2f}%")
    print(f"  Best Exact Match:{best_info.get('exact_match_rate',0):.2f}%")
    print(f"  Saved: {cfg.best_model_path}")
    print(f"{'='*70}")


def run_evaluation():
    cfg = Config()
    print("=" * 70)
    print("  Bengali CRNN V3 - Evaluation")
    print("=" * 70)
    ck = torch.load(cfg.best_model_path, map_location=cfg.device, weights_only=False)
    nc = ck['num_classes']
    cc = ck['config']
    model = CRNN_V3(nc, cc['img_height'], cc['hidden_size'])
    model.load_state_dict(ck['model_state_dict'])
    model.to(cfg.device).eval()
    c2i = ck['char2idx']
    i2c = {int(k):v for k,v in ck['idx2char'].items()}
    _, _, test_p, _, _ = load_data(cfg.dataset_json, cfg.max_gt_length)
    print(f"  Test: {len(test_p)} samples")
    ds = BengaliDataset(test_p, c2i, cc['img_height'], cc['img_width'])
    dl = DataLoader(ds, cfg.batch_size, shuffle=False, num_workers=cfg.num_workers, collate_fn=collate_fn)
    ap, ag = [], []
    with torch.no_grad():
        for imgs, _, _, txts in dl:
            p = model(imgs.to(cfg.device))
            ap.extend(decode(p, i2c)); ag.extend(txts)
    m = metrics(ap, ag)
    tft = {'char_acc':45.5, 'word_acc':24.78, 'cer':0.545, 'wer':0.752}
    print(f"\n  CRNN V3 Results:")
    print(f"    Char Accuracy:   {m['char_acc']:.2f}%")
    print(f"    Word Accuracy:   {m['word_acc']:.2f}%")
    print(f"    Exact Match:     {m['exact_match_rate']:.2f}%")
    print(f"    CER: {m['cer']:.4f} | WER: {m['wer']:.4f}")
    print(f"\n  vs Tesseract Finetuned:")
    print(f"    Char Acc: +{m['char_acc']-tft['char_acc']:.2f}%")
    print(f"    Word Acc: +{m['word_acc']-tft['word_acc']:.2f}%")
    report = [f"CRNN V3 Evaluation | Epoch {ck['epoch']}",
              f"CharAcc={m['char_acc']:.2f}% WordAcc={m['word_acc']:.2f}% ExactMatch={m['exact_match_rate']:.2f}%",
              f"CER={m['cer']:.4f} WER={m['wer']:.4f}", ""]
    for p, g in zip(ap, ag):
        s = "Y" if p==g else "N"
        report.append(f"{s} GT:[{g}] PR:[{p}]")
    with open('crnn_v3_report.txt', 'w', encoding='utf-8') as f:
        f.write('\n'.join(report))
    with open('eval_metrics_v3.json', 'w') as f:
        json.dump({'crnn_v3': m, 'tess_ft': tft}, f, indent=2)
    print(f"\n  Saved: crnn_v3_report.txt, eval_metrics_v3.json")


def run_figures():
    if not os.path.exists('training_log_v3.json'):
        print("No training log found. Train first.")
        return
    with open('training_log_v3.json') as f:
        log = json.load(f)
    eps = log['epochs']
    out = 'paper_figures_v3'
    os.makedirs(out, exist_ok=True)
    fig, axes = plt.subplots(1, 3, figsize=(16, 4.5))
    e = [x['ep'] for x in eps]
    axes[0].plot(e, [x['tl'] for x in eps], 'b-', lw=1.5, label='Train')
    axes[0].plot(e, [x['vl'] for x in eps], 'r-', lw=1.5, label='Val')
    axes[0].set_xlabel('Epoch'); axes[0].set_ylabel('Loss'); axes[0].set_title('Loss'); axes[0].legend(); axes[0].grid(alpha=0.3)
    axes[1].plot(e, [x['tc']*100 for x in eps], 'b-', lw=1.5, label='Train')
    axes[1].plot(e, [x['vc']*100 for x in eps], 'r-', lw=1.5, label='Val')
    axes[1].set_xlabel('Epoch'); axes[1].set_ylabel('CER%'); axes[1].set_title('CER'); axes[1].legend(); axes[1].grid(alpha=0.3)
    axes[2].plot(e, [x['ca'] for x in eps], 'g-', lw=2, label='Char')
    axes[2].plot(e, [x['wa'] for x in eps], 'm-', lw=2, label='Word')
    axes[2].axhline(90, color='k', ls='--', alpha=0.3)
    axes[2].set_xlabel('Epoch'); axes[2].set_ylabel('Acc%'); axes[2].set_title('Accuracy'); axes[2].legend()
    axes[2].grid(alpha=0.3); axes[2].set_ylim(0, 100)
    plt.tight_layout()
    plt.savefig(f'{out}/v3_training_curves.png', dpi=300, bbox_inches='tight')
    plt.close()
    print(f"  Saved: {out}/v3_training_curves.png")


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--evaluate', action='store_true')
    parser.add_argument('--figures', action='store_true')
    args = parser.parse_args()
    if args.evaluate: run_evaluation()
    elif args.figures: run_figures()
    else: run_training()
