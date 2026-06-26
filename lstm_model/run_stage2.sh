#!/bin/bash
# Stage 2: self-training + combined-data experiments. GPU-aware, checkpoints each step.
cd /home/ujjwal/ritu1/lstm_model
PY=/home/ujjwal/miniconda3/envs/ritu_scenetext/bin/python
NEED=4000

gpu_wait () {
  while true; do
    free=$(nvidia-smi --query-gpu=memory.free --format=csv,noheader,nounits | head -1)
    if [ "${free:-0}" -ge "$NEED" ]; then break; fi
    echo "[$(date)] GPU busy (${free} MiB free), waiting..."; sleep 120
  done
}
ev () {  # tag splits flags ckpt
  tag=$1; splits=$2; flags=$3; ckpt=$4
  mode=$([[ "$flags" == *use_graphemes* ]] && echo grapheme || echo standard)
  gpu_wait
  $PY -u train_florence2.py --evaluate --splits_file "$splits" $flags --ckpt_dir "$ckpt" > eval_${tag}.log 2>&1
  [ -f florence2_results_${mode}.json ] && mv florence2_results_${mode}.json results_${tag}.json && echo "[$(date)] saved results_${tag}.json"
}

# ---- 1. SELF-TRAINING (novelty #2) ----
echo "[$(date)] === self-training: pseudo-label unlabeled crops ==="
gpu_wait
$PY -u self_training_pseudolabel.py --conf 0.75 > stage2_pseudolabel.log 2>&1
echo "[$(date)] === self-train grapheme on augmented split ==="
gpu_wait
$PY -u train_florence2.py --use_graphemes --splits_file florence2_splits_selftrain.json \
    --ckpt_dir checkpoints_selftrain_grapheme > stage2_train_selftrain.log 2>&1
ev selftrain_ours florence2_splits.json "--use_graphemes" checkpoints_selftrain_grapheme

# ---- 2. COMBINED ours+BSTD: grapheme ----
echo "[$(date)] === combined grapheme (ours+BSTD) ==="
gpu_wait
$PY -u train_florence2.py --use_graphemes --splits_file florence2_splits_combined.json \
    --ckpt_dir checkpoints_combined_grapheme > stage2_train_comb_grph.log 2>&1
ev grph_combined florence2_splits_combined.json "--use_graphemes" checkpoints_combined_grapheme

# ---- 3. COMBINED ours+BSTD: standard ----
echo "[$(date)] === combined standard (ours+BSTD) ==="
gpu_wait
$PY -u train_florence2.py --splits_file florence2_splits_combined.json \
    --ckpt_dir checkpoints_combined_standard > stage2_train_comb_std.log 2>&1
ev std_combined florence2_splits_combined.json "" checkpoints_combined_standard

echo "[$(date)] STAGE 2 COMPLETE"
