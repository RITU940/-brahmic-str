#!/bin/bash
# Wave 1: zero-shot baseline, fusion-on-BSTD, leakage-free CRNN baseline.
cd /home/ujjwal/ritu1/lstm_model
PY=/home/ujjwal/miniconda3/envs/ritu_scenetext/bin/python
NEED=4000
gpu_wait () { while true; do f=$(nvidia-smi --query-gpu=memory.free --format=csv,noheader,nounits|head -1); [ "${f:-0}" -ge "$NEED" ] && break; sleep 120; done; }

# 1) Zero-shot Florence-2 (baseline row) on both test sets
gpu_wait; $PY -u zeroshot_florence2_eval.py --splits_file florence2_splits.json > eval_zs_ours.log 2>&1
[ -f florence2_results_zeroshot.json ] && mv florence2_results_zeroshot.json results_zeroshot_ours.json
gpu_wait; $PY -u zeroshot_florence2_eval.py --splits_file florence2_splits_bstd.json > eval_zs_bstd.log 2>&1
[ -f florence2_results_zeroshot.json ] && mv florence2_results_zeroshot.json results_zeroshot_bstd.json
echo "[$(date)] zero-shot done"

# 2) Fusion on BSTD public test (combined-trained models)
gpu_wait; $PY -u predict_with_conf.py --tag std_bstd  --ckpt_dir checkpoints_combined_standard \
            --splits_file florence2_splits_bstd.json > conf_std_bstd.log 2>&1
gpu_wait; $PY -u predict_with_conf.py --tag grph_bstd --ckpt_dir checkpoints_combined_grapheme --use_graphemes \
            --splits_file florence2_splits_bstd.json > conf_grph_bstd.log 2>&1
$PY -u fusion_analysis.py --std std_bstd --grph grph_bstd --self _none_ > fusion_bstd.log 2>&1
echo "[$(date)] fusion-on-BSTD done"

# 3) Leakage-free CRNN baseline on our splits
$PY -u prepare_crnn_florence_splits.py > crnn_prep.log 2>&1
gpu_wait; $PY -u train_crnn_on_florence.py > crnn_train.log 2>&1
$PY -u train_crnn_on_florence.py --evaluate > crnn_eval.log 2>&1
echo "[$(date)] CRNN done"
echo "[$(date)] STAGE 3a COMPLETE"
