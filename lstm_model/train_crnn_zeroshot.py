"""
Train + evaluate CRNN_V3 on one zero-shot LOSO rung (architecture-generality
test, Amendment 6 / ``PROSPECTIVE_PREDICTION_ARCHITECTURE.md``).

Reuses ``colab_train_v3.py``'s training loop unchanged — only patches
``Config`` to point at the rung's CRNN-format splits and a per-rung checkpoint
dir, with the training budget frozen in the prediction file (60 epochs max,
early-stop patience 10).

Evaluation deliberately does NOT reuse ``run_evaluation``: it scores the
**unfiltered** test split with ``metrics.evaluate_corpus`` — the identical
metric every Florence-2 rung used — and writes
``conf_crnn_zs_rung{R}_{tag}.json`` / ``result_crnn_zs_rung{R}_{tag}.json``
in the same schema as the Florence-2 results.

USAGE (after prepare_crnn_zeroshot.py):
    python train_crnn_zeroshot.py --rung B --tag tamil
    python train_crnn_zeroshot.py --rung B --tag tamil --evaluate_only
"""
import argparse
import json
import os
import sys
import types

sys.stdout.reconfigure(encoding='utf-8', errors='replace')
BASE = os.path.abspath(os.path.dirname(__file__))
sys.path.insert(0, BASE)

# colab_train_v3 imports matplotlib at module top; stub it so training runs in
# environments without it (same pattern as train_crnn_on_florence.py).
if 'matplotlib' not in sys.modules:
    mpl_stub = types.ModuleType('matplotlib')
    mpl_stub.use = lambda *a, **k: None
    sys.modules['matplotlib'] = mpl_stub
    pyplot_stub = types.ModuleType('matplotlib.pyplot')
    for fn in ('figure', 'plot', 'savefig', 'close', 'subplot', 'title',
               'xlabel', 'ylabel', 'legend', 'tight_layout', 'show', 'subplots'):
        setattr(pyplot_stub, fn, lambda *a, **k: None)
    sys.modules['matplotlib.pyplot'] = pyplot_stub

import torch  # noqa: E402
from torch.utils.data import DataLoader  # noqa: E402

import colab_train_v3 as ctv  # noqa: E402
from metrics import evaluate_corpus  # noqa: E402

# Frozen budget — declared in PROSPECTIVE_PREDICTION_ARCHITECTURE.md before any run.
MAX_EPOCHS = 60
PATIENCE = 10


def paths(rung, tag):
    return {
        'data': os.path.join(BASE, f'splits_crnn_zs_rung{rung}_{tag}.json'),
        'ckpt': os.path.join(BASE, f'checkpoints_crnn_zs_rung{rung}_{tag}'),
        'best': os.path.join(BASE, f'checkpoints_crnn_zs_rung{rung}_{tag}',
                             'best_model.pth'),
        'log': os.path.join(BASE, f'training_log_crnn_zs_rung{rung}_{tag}.json'),
        'conf': os.path.join(BASE, f'conf_crnn_zs_rung{rung}_{tag}.json'),
        'result': os.path.join(BASE, f'result_crnn_zs_rung{rung}_{tag}.json'),
    }


def patch_config(p):
    if not os.path.exists(p['data']):
        sys.exit(f"ERROR: {p['data']} missing — run prepare_crnn_zeroshot.py first.")
    ctv.Config.dataset_json = p['data']
    ctv.Config.checkpoint_dir = p['ckpt']
    ctv.Config.best_model_path = p['best']
    ctv.Config.log_file = p['log']
    ctv.Config.num_epochs = MAX_EPOCHS
    ctv.Config.early_stop_patience = PATIENCE
    os.makedirs(p['ckpt'], exist_ok=True)


def evaluate(rung, tag, p):
    """Score the UNFILTERED test split with the Florence-2 metric."""
    cfg = ctv.Config()
    ck = torch.load(p['best'], map_location=cfg.device, weights_only=False)
    cc = ck['config']
    model = ctv.CRNN_V3(ck['num_classes'], cc['img_height'], cc['hidden_size'])
    model.load_state_dict(ck['model_state_dict'])
    model.to(cfg.device).eval()
    c2i = ck['char2idx']
    i2c = {int(k): v for k, v in ck['idx2char'].items()}

    with open(p['data'], 'r', encoding='utf-8') as f:
        data = json.load(f)
    pairs = data['pairs']
    test_pairs = [pairs[i] for i in data['splits']['test']]   # no length filter
    print(f"  Test: {len(test_pairs)} images (unfiltered — matches the Florence-2 rung)")

    ds = ctv.BengaliDataset(test_pairs, c2i, cc['img_height'], cc['img_width'])
    dl = DataLoader(ds, cfg.batch_size, shuffle=False,
                    num_workers=cfg.num_workers, collate_fn=ctv.collate_fn)
    preds, gts = [], []
    with torch.no_grad():
        for imgs, _, _, txts in dl:
            out = model(imgs.to(cfg.device))
            preds.extend(ctv.decode(out, i2c))
            gts.extend(txts)

    recs = [{'name': tp['name'], 'gt': g, 'pred': pr}
            for tp, g, pr in zip(test_pairs, gts, preds)]
    with open(p['conf'], 'w', encoding='utf-8') as f:
        json.dump(recs, f, ensure_ascii=False)

    m = evaluate_corpus(gts, preds)
    rec = {'backbone': 'CRNN_V3', 'rung': rung, 'script': tag, 'N': len(recs),
           'WRR': round(m['WRR'], 2), 'CharAcc': round(m['char_accuracy'], 2),
           'CER': round(m['CER'], 2), 'epochs_max': MAX_EPOCHS,
           'best_epoch': ck.get('epoch')}
    with open(p['result'], 'w', encoding='utf-8') as f:
        json.dump(rec, f)
    print(f"[RESULT CRNN RUNG {rung} {tag}] N={rec['N']} WRR={rec['WRR']} "
          f"CharAcc={rec['CharAcc']} CER={rec['CER']}")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--rung', required=True, choices=['A', 'B'])
    ap.add_argument('--tag', required=True)
    ap.add_argument('--evaluate_only', action='store_true')
    a = ap.parse_args()

    p = paths(a.rung, a.tag)
    patch_config(p)
    done = os.path.join(p['ckpt'], '.train_done')
    if not a.evaluate_only:
        if os.path.exists(done):
            print(f"  {p['ckpt']} already trained (.train_done) — skipping training")
        else:
            ctv.run_training()
            open(done, 'w').close()
    evaluate(a.rung, a.tag, p)


if __name__ == '__main__':
    main()
