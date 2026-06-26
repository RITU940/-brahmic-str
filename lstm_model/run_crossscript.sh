#!/bin/bash
# Cross-script generalization: Assamese (Bengali script) + Hindi (Devanagari).
# Trains standard + grapheme Florence-2 per language, gets beam-confidence
# predictions, and runs the dual-tokenization fusion on each public test set.
cd /home/ujjwal/ritu1/lstm_model
PY=/home/ujjwal/miniconda3/envs/ritu_scenetext/bin/python
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
NEED=12000
gpu_wait () { while true; do f=$(nvidia-smi --query-gpu=memory.free --format=csv,noheader,nounits|head -1); [ "${f:-0}" -ge "$NEED" ] && break; sleep 120; done; }

run_lang () {
  lang=$1
  splits=florence2_splits_bstd_${lang}.json
  vocab=grapheme_vocab_${lang}.json

  gpu_wait
  $PY -u train_florence2.py --splits_file $splits \
      --ckpt_dir checkpoints_bstd_${lang}_standard > train_${lang}_std.log 2>&1
  [ -d checkpoints_bstd_${lang}_standard/best_model ] || { echo "[$(date)] ${lang} STD TRAIN FAILED"; return 1; }

  gpu_wait
  $PY -u train_florence2.py --use_graphemes --grapheme_vocab $vocab --splits_file $splits \
      --ckpt_dir checkpoints_bstd_${lang}_grapheme > train_${lang}_grph.log 2>&1
  [ -d checkpoints_bstd_${lang}_grapheme/best_model ] || { echo "[$(date)] ${lang} GRPH TRAIN FAILED"; return 1; }

  gpu_wait
  $PY -u predict_with_conf.py --tag std_${lang} --ckpt_dir checkpoints_bstd_${lang}_standard \
      --splits_file $splits > conf_${lang}_std.log 2>&1
  gpu_wait
  $PY -u predict_with_conf.py --tag grph_${lang} --ckpt_dir checkpoints_bstd_${lang}_grapheme \
      --use_graphemes --grapheme_vocab $vocab --splits_file $splits > conf_${lang}_grph.log 2>&1

  $PY -u fusion_analysis.py --std std_${lang} --grph grph_${lang} --self _none_ > fusion_${lang}.log 2>&1
  echo "[$(date)] ${lang} DONE"; cat fusion_${lang}.log
}

run_lang assamese
run_lang hindi
echo "[$(date)] CROSS-SCRIPT COMPLETE"
