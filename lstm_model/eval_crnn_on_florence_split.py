"""
Re-evaluate the existing CRNN baseline (best_model_v3.pth) on the SAME
test split used by the Florence-2 evals, so the paper's master table is
apples-to-apples.

Reuses ``run_crnn_baseline`` from ``run_baselines.py`` and writes to
``baseline_crnn_results_florence_split.json`` so the existing
``baseline_crnn_results.json`` (legacy 717-sample) is preserved.
"""
import os
import sys
import json
import shutil
import types

sys.stdout.reconfigure(encoding='utf-8', errors='replace')
BASE_DIR = os.path.abspath(os.path.dirname(__file__))
sys.path.insert(0, BASE_DIR)

# colab_train_v3 imports matplotlib for training-time plots; we don't need it
# for inference. Stub it so the import doesn't fail in this minimal env.
if 'matplotlib' not in sys.modules:
    mpl_stub = types.ModuleType('matplotlib')
    mpl_stub.use = lambda *a, **k: None
    sys.modules['matplotlib'] = mpl_stub
    pyplot_stub = types.ModuleType('matplotlib.pyplot')
    for fn in ('figure', 'plot', 'savefig', 'close', 'subplot', 'title',
               'xlabel', 'ylabel', 'legend', 'tight_layout', 'show', 'subplots'):
        setattr(pyplot_stub, fn, lambda *a, **k: None)
    sys.modules['matplotlib.pyplot'] = pyplot_stub

from train_florence2 import SPLITS_FILE, validate_and_resolve_pairs  # noqa: E402
import run_baselines as rb  # noqa: E402


def main():
    print("=" * 60)
    print("  CRNN re-eval on the Florence-2 test split (307 images)")
    print("=" * 60)

    with open(SPLITS_FILE, 'r', encoding='utf-8') as f:
        splits = json.load(f)
    test_pairs, missing = validate_and_resolve_pairs(
        'test', splits['test'], filter_missing=False
    )
    assert missing == 0, f"{missing} test images missing"
    print(f"\n  Test set: {len(test_pairs)} images")

    # Back up the legacy results so run_crnn_baseline can save freely.
    legacy = os.path.join(BASE_DIR, 'baseline_crnn_results.json')
    legacy_bak = os.path.join(BASE_DIR, 'baseline_crnn_results_legacy717.json')
    if os.path.exists(legacy) and not os.path.exists(legacy_bak):
        shutil.copy2(legacy, legacy_bak)
        print(f"  Backed up legacy CRNN results -> {legacy_bak}")

    results = rb.run_crnn_baseline(test_pairs)
    if results is None:
        print("  ERROR: CRNN baseline failed (see warnings above)")
        sys.exit(1)

    # Move the freshly-written file to a distinct name and restore the legacy file.
    new_results = os.path.join(BASE_DIR, 'baseline_crnn_results_florence_split.json')
    shutil.move(legacy, new_results)
    if os.path.exists(legacy_bak):
        shutil.copy2(legacy_bak, legacy)
    print(f"\n  Florence-split CRNN results saved: {new_results}")
    print(f"  Legacy CRNN results preserved at:  {legacy}")


if __name__ == '__main__':
    main()
