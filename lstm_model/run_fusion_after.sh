#!/bin/bash
# 1) wait for combined-standard BSTD result; 2) run dual-tokenization fusion.
cd /home/ujjwal/ritu1/lstm_model
PY=/home/ujjwal/miniconda3/envs/ritu_scenetext/bin/python
NEED=4000
gpu_wait () { while true; do f=$(nvidia-smi --query-gpu=memory.free --format=csv,noheader,nounits|head -1); [ "${f:-0}" -ge "$NEED" ] && break; sleep 120; done; }

echo "[$(date)] waiting for results_std_combined.json (combined-standard BSTD)..."
while [ ! -f results_std_combined.json ]; do sleep 120; done
echo "[$(date)] combined-standard done. Running dual-tokenization fusion."

# Per-model predictions WITH confidence on OUR test set
gpu_wait; $PY -u predict_with_conf.py --tag std_ours       --ckpt_dir checkpoints_florence2_standard                          > conf_std.log 2>&1
gpu_wait; $PY -u predict_with_conf.py --tag grph_ours      --ckpt_dir checkpoints_florence2_grapheme       --use_graphemes   > conf_grph.log 2>&1
gpu_wait; $PY -u predict_with_conf.py --tag selftrain_ours --ckpt_dir checkpoints_selftrain_grapheme       --use_graphemes   > conf_self.log 2>&1

$PY -u fusion_analysis.py > fusion_result.log 2>&1
echo "[$(date)] FUSION DONE"
cat fusion_result.log
