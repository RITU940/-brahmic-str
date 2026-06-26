#!/bin/bash
# Wave 2 RETRY: synthetic-curriculum grapheme training + eval.
# Fix vs original: require 12 GB free (not 4) so a competing lab job can't OOM us
# mid-run, and enable expandable_segments to reduce fragmentation.
cd /home/ujjwal/ritu1/lstm_model
PY=/home/ujjwal/miniconda3/envs/ritu_scenetext/bin/python
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
NEED=12000
gpu_wait () { while true; do f=$(nvidia-smi --query-gpu=memory.free --format=csv,noheader,nounits|head -1); [ "${f:-0}" -ge "$NEED" ] && break; sleep 120; done; }

gpu_wait
$PY -u train_florence2.py --use_graphemes --splits_file florence2_splits_synthaug.json \
    --ckpt_dir checkpoints_synthaug_grapheme > stage3b_train.log 2>&1

if [ ! -d checkpoints_synthaug_grapheme/best_model ]; then
  echo "[$(date)] RETRY TRAIN FAILED (no best_model)"; tail -5 stage3b_train.log; exit 1
fi

gpu_wait
$PY -u train_florence2.py --evaluate --use_graphemes --splits_file florence2_splits.json \
    --ckpt_dir checkpoints_synthaug_grapheme > stage3b_eval.log 2>&1
[ -f florence2_results_grapheme.json ] && mv florence2_results_grapheme.json results_synthaug_ours.json

gpu_wait
$PY -u predict_with_conf.py --tag synthaug_ours --ckpt_dir checkpoints_synthaug_grapheme --use_graphemes \
    --splits_file florence2_splits.json > conf_synthaug.log 2>&1

# 4-way fusion: standard + grapheme + self-train + synthaug
$PY -u fusion_analysis.py --std std_ours --grph grph_ours --self selftrain_ours > fusion_final.log 2>&1

echo "[$(date)] STAGE 3b RETRY COMPLETE"
[ -f results_synthaug_ours.json ] && $PY -c "import json;d=json.load(open('results_synthaug_ours.json'));print('synthaug WRR',round(d['WRR'],2),'CER',round(d['CER'],2),'CharAcc',round(d['char_accuracy'],2))"
