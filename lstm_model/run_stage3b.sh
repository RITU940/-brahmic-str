#!/bin/bash
# Wave 2: grapheme-rarity-weighted synthetic augmentation training + eval.
cd /home/ujjwal/ritu1/lstm_model
PY=/home/ujjwal/miniconda3/envs/ritu_scenetext/bin/python
NEED=4000
gpu_wait () { while true; do f=$(nvidia-smi --query-gpu=memory.free --format=csv,noheader,nounits|head -1); [ "${f:-0}" -ge "$NEED" ] && break; sleep 120; done; }

echo "[$(date)] waiting for Wave 1 (stage 3a) to complete..."
while ! grep -q "STAGE 3a COMPLETE" run_stage3a_runner.log 2>/dev/null; do sleep 120; done
echo "[$(date)] Wave 1 done. Training synthetic-augmented grapheme model."

gpu_wait
$PY -u train_florence2.py --use_graphemes --splits_file florence2_splits_synthaug.json \
    --ckpt_dir checkpoints_synthaug_grapheme > stage3b_train.log 2>&1

gpu_wait
$PY -u train_florence2.py --evaluate --use_graphemes --splits_file florence2_splits.json \
    --ckpt_dir checkpoints_synthaug_grapheme > stage3b_eval.log 2>&1
[ -f florence2_results_grapheme.json ] && mv florence2_results_grapheme.json results_synthaug_ours.json

# predictions+confidence so the synth model can also join the fusion
gpu_wait
$PY -u predict_with_conf.py --tag synthaug_ours --ckpt_dir checkpoints_synthaug_grapheme --use_graphemes \
    --splits_file florence2_splits.json > conf_synthaug.log 2>&1

echo "[$(date)] STAGE 3b COMPLETE"
[ -f results_synthaug_ours.json ] && $PY -c "import json;d=json.load(open('results_synthaug_ours.json'));print('synthaug WRR',d['WRR'],'CER',d['CER'],'CharAcc',d['char_accuracy'])"
