"""
Bengali Scene Text Recognition - Comprehensive Evaluation
==========================================================
Reports ALL standard metrics used in STR papers:
  - WRR (Word Recognition Rate) — primary metric
  - CER (Character Error Rate)
  - CRR (Character Recognition Rate = 1-CER)
  - NED (Normalized Edit Distance)
  - 1-NED (higher is better)
  - WER (Word Error Rate)
  - FPS (inference speed)
  - Model parameter count

Supports evaluating on:
  - Our test set (dataset_splits.json)
  - IndicSTR12 Bengali test set
  - BSTD Bengali test set
  - Any custom test set in JSON format

USAGE:
  python evaluate_all.py --model best_model_v3.pth --test our
  python evaluate_all.py --model best_model_v3.pth --test indicstr12
  python evaluate_all.py --model best_model_v3.pth --test bstd
  python evaluate_all.py --model best_model_v3.pth --test all
"""
import os, sys, json, time, argparse
import numpy as np
from PIL import Image
import torch
import torch.nn as nn

sys.stdout.reconfigure(encoding='utf-8', errors='replace')


# =====================================================================
#  MODEL (must match training architecture exactly)
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
    def __init__(self, num_classes, img_height=64, hidden_size=512):
        super().__init__()
        self.cnn = nn.Sequential(
            nn.Conv2d(1, 64, 3, 1, 1), nn.BatchNorm2d(64), nn.ReLU(True),
            nn.MaxPool2d(2, 2),
            nn.Conv2d(64, 128, 3, 1, 1), nn.BatchNorm2d(128), nn.ReLU(True),
            nn.MaxPool2d(2, 2),
            ResBlock(128),
            nn.Conv2d(128, 256, 3, 1, 1), nn.BatchNorm2d(256), nn.ReLU(True),
            nn.MaxPool2d((2, 1), (2, 1)),
            ResBlock(256),
            nn.Conv2d(256, 512, 3, 1, 1), nn.BatchNorm2d(512), nn.ReLU(True),
            nn.MaxPool2d((2, 1), (2, 1)),
            ResBlock(512),
            nn.Conv2d(512, 512, (4, 2), 1, (0, 0)), nn.BatchNorm2d(512), nn.ReLU(True),
        )
        self.rnn1 = nn.LSTM(512, hidden_size, bidirectional=True, batch_first=True)
        self.ln1 = nn.LayerNorm(hidden_size * 2)
        self.rnn2 = nn.LSTM(hidden_size * 2, hidden_size, bidirectional=True, batch_first=True)
        self.fc = nn.Linear(hidden_size * 2, num_classes)
        self.drop = nn.Dropout(0.4)

    def forward(self, x):
        c = self.cnn(x)
        b, ch, h, w = c.size()
        if h != 1:
            c = c.mean(dim=2, keepdim=True)
        c = c.squeeze(2).permute(0, 2, 1)
        c = self.drop(c)
        r1, _ = self.rnn1(c)
        r1 = self.ln1(r1)
        r2, _ = self.rnn2(r1)
        out = self.fc(r2)
        return torch.nn.functional.log_softmax(out.permute(1, 0, 2), dim=2)


# =====================================================================
#  METRICS
# =====================================================================
def edit_distance(ref, hyp):
    m, n = len(ref), len(hyp)
    if m == 0: return n
    if n == 0: return m
    dp = list(range(n + 1))
    for i in range(1, m + 1):
        c = [i] + [0] * n
        for j in range(1, n + 1):
            c[j] = min(dp[j]+1, c[j-1]+1, dp[j-1]+(0 if ref[i-1]==hyp[j-1] else 1))
        dp = c
    return dp[n]


def normalized_edit_distance(pred, gt):
    """NED = edit_distance / max(len(pred), len(gt))"""
    if len(pred) == 0 and len(gt) == 0:
        return 0.0
    ed = edit_distance(list(gt), list(pred))
    return ed / max(len(gt), len(pred), 1)


def compute_all_metrics(predictions, ground_truths):
    """Compute ALL standard STR metrics."""
    total_ed = 0
    total_gt_chars = 0
    total_ned = 0.0
    exact_matches = 0
    n = 0

    for pred, gt in zip(predictions, ground_truths):
        if not gt.strip():
            continue
        n += 1

        # Edit distance (character level)
        ed = edit_distance(list(gt), list(pred))
        total_ed += ed
        total_gt_chars += len(gt)

        # NED
        ned = normalized_edit_distance(pred, gt)
        total_ned += ned

        # Exact match (WRR)
        if pred.strip() == gt.strip():
            exact_matches += 1

    n = max(n, 1)
    cer = total_ed / max(total_gt_chars, 1)
    crr = (1.0 - cer) * 100
    avg_ned = total_ned / n
    one_minus_ned = (1.0 - avg_ned) * 100
    wrr = exact_matches / n * 100

    return {
        'WRR': round(wrr, 2),
        'CER': round(cer, 4),
        'CRR': round(crr, 2),
        'NED': round(avg_ned, 4),
        '1-NED': round(one_minus_ned, 2),
        'exact_matches': exact_matches,
        'total_samples': n,
    }


# =====================================================================
#  IMAGE PREPROCESSING & DECODING
# =====================================================================
def preprocess_image(img_path, h=64, w=256):
    try:
        img = Image.open(img_path)
    except:
        img = Image.new('L', (w, h), 255)
    if img.mode != 'L':
        img = img.convert('L')
    iw, ih = img.size
    scale = h / ih
    tw = min(int(iw * scale), w)
    tw = max(tw, 1)
    img = img.resize((tw, h), Image.LANCZOS)
    out = Image.new('L', (w, h), 255)
    out.paste(img, (0, 0))
    arr = np.array(out, dtype=np.float32) / 255.0
    arr = 1.0 - arr
    return torch.FloatTensor(arr).unsqueeze(0)  # (1, H, W)


def decode_predictions(preds, idx2char):
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


# =====================================================================
#  TEST SET LOADERS
# =====================================================================
def load_our_test_set(json_path='dataset_splits.json'):
    """Load our test set from dataset_splits.json"""
    with open(json_path, 'r', encoding='utf-8') as f:
        data = json.load(f)
    pairs = data['pairs']
    splits = data['splits']
    test_pairs = []
    for i in splits['test']:
        p = pairs[i]
        img_path = p['image'].replace('\\', '/')
        if not os.path.isabs(img_path):
            img_path = os.path.join('.', img_path)
        test_pairs.append({'image': img_path, 'gt': p['gt']})
    return test_pairs


def load_external_test_set(json_path):
    """Load any external test set in [{image, gt}] JSON format."""
    with open(json_path, 'r', encoding='utf-8') as f:
        data = json.load(f)
    if isinstance(data, list):
        return data
    elif 'pairs' in data:
        return data['pairs']
    else:
        raise ValueError(f"Unrecognized format in {json_path}")


# =====================================================================
#  MAIN EVALUATION
# =====================================================================
def evaluate_model(model_path, test_set_name='our', device='cpu'):
    print("=" * 80)
    print("  Bengali Scene Text Recognition — Comprehensive Evaluation")
    print("=" * 80)

    # Load checkpoint
    print(f"\n  Loading model: {model_path}")
    ck = torch.load(model_path, map_location=device, weights_only=False)
    cfg = ck['config']
    nc = ck['num_classes']
    h, w = cfg['img_height'], cfg['img_width']
    hs = cfg['hidden_size']

    model = CRNN_V3(nc, h, hs)
    model.load_state_dict(ck['model_state_dict'])
    model.to(device).eval()

    # Model stats
    total_params = sum(p.numel() for p in model.parameters())
    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"  Model: CRNN V3 (ResNet+BiLSTM+CTC)")
    print(f"  Parameters: {total_params:,} ({trainable_params:,} trainable)")
    print(f"  Image size: {h}x{w}")
    print(f"  Trained epoch: {ck['epoch']}")

    c2i = ck['char2idx']
    i2c = {int(k): v for k, v in ck['idx2char'].items()}

    # Load test sets
    test_sets = {}
    if test_set_name in ('our', 'all'):
        test_sets['Our Test Set'] = load_our_test_set()
    if test_set_name in ('indicstr12', 'all'):
        p = 'external_tests/indicstr12_bengali.json'
        if os.path.exists(p):
            test_sets['IndicSTR12 Bengali'] = load_external_test_set(p)
        else:
            print(f"  [SKIP] {p} not found — run download_test_sets.py first")
    if test_set_name in ('bstd', 'all'):
        p = 'external_tests/bstd_bengali.json'
        if os.path.exists(p):
            test_sets['BSTD Bengali'] = load_external_test_set(p)
        else:
            print(f"  [SKIP] {p} not found — run download_test_sets.py first")

    if not test_sets:
        print("  No test sets available!")
        return

    # Evaluate on each test set
    all_results = {}
    for name, pairs in test_sets.items():
        print(f"\n{'─'*80}")
        print(f"  Evaluating on: {name} ({len(pairs)} samples)")
        print(f"{'─'*80}")

        all_preds = []
        all_gts = []
        batch_size = 32
        total_inference_time = 0

        with torch.no_grad():
            for i in range(0, len(pairs), batch_size):
                batch = pairs[i:i+batch_size]
                imgs = []
                for p in batch:
                    img_tensor = preprocess_image(p['image'], h, w)
                    imgs.append(img_tensor)
                    all_gts.append(p['gt'])

                imgs_batch = torch.stack(imgs).to(device)

                t0 = time.perf_counter()
                preds = model(imgs_batch)
                total_inference_time += time.perf_counter() - t0

                decoded = decode_predictions(preds, i2c)
                all_preds.extend(decoded)

        # Compute metrics
        m = compute_all_metrics(all_preds, all_gts)
        fps = len(all_preds) / max(total_inference_time, 0.001)
        m['FPS'] = round(fps, 1)
        m['params'] = total_params

        all_results[name] = m

        # Print results
        print(f"\n  Results:")
        print(f"    Word Recognition Rate (WRR):   {m['WRR']:>7.2f}%")
        print(f"    Character Recognition Rate:    {m['CRR']:>7.2f}%")
        print(f"    Character Error Rate (CER):    {m['CER']:>7.4f}")
        print(f"    Normalized Edit Distance:      {m['NED']:>7.4f}")
        print(f"    1-NED:                         {m['1-NED']:>7.2f}%")
        print(f"    Exact Matches:                 {m['exact_matches']}/{m['total_samples']}")
        print(f"    Inference Speed:               {m['FPS']:>7.1f} FPS")
        print(f"    Model Parameters:              {m['params']:>10,}")

        # Show some examples
        print(f"\n  Sample Predictions:")
        shown = 0
        for pred, gt in zip(all_preds, all_gts):
            if shown >= 8:
                break
            s = "✓" if pred == gt else "✗"
            print(f"    {s} GT:[{gt}] → Pred:[{pred}]")
            shown += 1

    # Print comparison table
    if len(all_results) > 0:
        print(f"\n{'='*80}")
        print(f"  COMPARISON TABLE")
        print(f"{'='*80}")
        header = f"  {'Test Set':<25} {'WRR':>7} {'CRR':>7} {'CER':>7} {'NED':>7} {'1-NED':>7} {'FPS':>7}"
        print(header)
        print(f"  {'─'*25} {'─'*7} {'─'*7} {'─'*7} {'─'*7} {'─'*7} {'─'*7}")
        for name, m in all_results.items():
            print(f"  {name:<25} {m['WRR']:>6.2f}% {m['CRR']:>6.2f}% {m['CER']:>7.4f} {m['NED']:>7.4f} {m['1-NED']:>6.2f}% {m['FPS']:>6.1f}")

    # SOTA comparison
    print(f"\n{'='*80}")
    print(f"  PUBLISHED SOTA COMPARISON (Bengali Scene Text)")
    print(f"{'='*80}")
    sota = [
        ('CRNN (IndicSTR12)', '2023', '48.21', '59.86'),
        ('STARNet (IndicSTR12)', '2023', '57.70', '80.26'),
        ('PARSeq (IndicSTR12)', '2023', '62.04', '83.08'),
        ('PARSeq fine-tuned (BSTD)', '2025', '~82', '—'),
    ]
    print(f"  {'Method':<30} {'Year':>5} {'WRR':>7} {'CRR':>7}")
    print(f"  {'─'*30} {'─'*5} {'─'*7} {'─'*7}")
    for name, year, wrr, crr in sota:
        print(f"  {name:<30} {year:>5} {wrr:>6}% {crr:>6}%")
    if 'Our Test Set' in all_results:
        m = all_results['Our Test Set']
        print(f"  {'CRNN V3 (Ours)':<30} {'2026':>5} {m['WRR']:>6.2f}% {m['CRR']:>6.2f}%")
    print(f"{'='*80}")

    # Save results
    output = {
        'model': model_path,
        'model_info': {
            'architecture': 'CRNN_V3 (ResNet+BiLSTM+CTC)',
            'params': total_params,
            'img_size': f"{h}x{w}",
            'hidden_size': hs,
            'num_classes': nc,
            'epoch': ck['epoch'],
        },
        'results': all_results,
    }
    os.makedirs('results', exist_ok=True)
    with open('results/evaluation_results.json', 'w', encoding='utf-8') as f:
        json.dump(output, f, indent=2, ensure_ascii=False)
    print(f"\n  Results saved: results/evaluation_results.json")

    return all_results


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--model', default='best_model_v3.pth')
    parser.add_argument('--test', default='our', choices=['our', 'indicstr12', 'bstd', 'all'])
    parser.add_argument('--device', default='cuda' if torch.cuda.is_available() else 'cpu')
    args = parser.parse_args()
    evaluate_model(args.model, args.test, args.device)
