"""
Retrain CRNN_V3 on the SAME splits used by Florence-2, so the master
paper comparison table is leakage-free.

Reuses ``colab_train_v3.py``'s training loop unchanged — only patches
``Config`` to point at ``florence2_splits_for_crnn.json`` and a
distinct checkpoint dir (``checkpoints_crnn_florence/``) so the
original ``best_model_v3.pth`` is untouched.

USAGE (run after ``prepare_crnn_florence_splits.py``):
    python train_crnn_on_florence.py            # train
    python train_crnn_on_florence.py --evaluate # eval (loads new best)
"""
import os
import sys
import types
import argparse

sys.stdout.reconfigure(encoding='utf-8', errors='replace')
BASE_DIR = os.path.abspath(os.path.dirname(__file__))
sys.path.insert(0, BASE_DIR)

# colab_train_v3 imports matplotlib at module top; stub it so training
# can run in environments without matplotlib (figures script is a no-op).
if 'matplotlib' not in sys.modules:
    mpl_stub = types.ModuleType('matplotlib')
    mpl_stub.use = lambda *a, **k: None
    sys.modules['matplotlib'] = mpl_stub
    pyplot_stub = types.ModuleType('matplotlib.pyplot')
    for fn in ('figure', 'plot', 'savefig', 'close', 'subplot', 'title',
               'xlabel', 'ylabel', 'legend', 'tight_layout', 'show', 'subplots'):
        setattr(pyplot_stub, fn, lambda *a, **k: None)
    sys.modules['matplotlib.pyplot'] = pyplot_stub

import colab_train_v3 as ctv  # noqa: E402

DATASET_JSON = os.path.join(BASE_DIR, 'florence2_splits_for_crnn.json')
CKPT_DIR = os.path.join(BASE_DIR, 'checkpoints_crnn_florence')
BEST_PATH = os.path.join(CKPT_DIR, 'best_model_crnn_florence.pth')
LOG_PATH = os.path.join(BASE_DIR, 'training_log_crnn_florence.json')


def patch_config():
    if not os.path.exists(DATASET_JSON):
        print(f"ERROR: {DATASET_JSON} missing — run prepare_crnn_florence_splits.py first.")
        sys.exit(1)
    ctv.Config.dataset_json = DATASET_JSON
    ctv.Config.checkpoint_dir = CKPT_DIR
    ctv.Config.best_model_path = BEST_PATH
    ctv.Config.log_file = LOG_PATH
    print(f"  Dataset    : {DATASET_JSON}")
    print(f"  Checkpoints: {CKPT_DIR}")
    print(f"  Log        : {LOG_PATH}")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--evaluate', action='store_true')
    args = ap.parse_args()
    patch_config()
    if args.evaluate:
        ctv.run_evaluation()
    else:
        ctv.run_training()


if __name__ == '__main__':
    main()
