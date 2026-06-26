#!/bin/bash
# ============================================================================
# Part ① (the LAW) — train BPE + grapheme branches for ALL 12 BSTD languages
# at the EQUAL 1,800-word budget (so script effect is not confounded by data
# volume). Produces per-language WRR, per-sample confidences, and fusion.
#
# Resumable: skips any (lang, branch) whose best_model already exists.
# Each run waits until >=12 GB GPU is free, so it is safe to launch on a shared GPU.
#
# Usage:   bash run_law_training.sh              # all 12 languages
#          bash run_law_training.sh tamil telugu # just these
# Outputs: checkpoints_law_<lang>_{standard,grapheme}/best_model
#          conf_law_<lang>_{std,grph}.json , results_law_<lang>_*.json
#          fusion_law_<lang>.log  (the key per-language numbers)
# ============================================================================
cd /home/ujjwal/ritu1/lstm_model
PY=/home/ujjwal/miniconda3/envs/ritu_scenetext/bin/python
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
NEED=12000   # MB of free GPU memory required before starting a run

LANGS=("$@")
if [ ${#LANGS[@]} -eq 0 ]; then
  LANGS=(bengali assamese hindi marathi gujarati punjabi kannada malayalam odia tamil telugu english)
fi

gpu_wait () {
  while true; do
    f=$(nvidia-smi --query-gpu=memory.free --format=csv,noheader,nounits | head -1)
    [ "${f:-0}" -ge "$NEED" ] && break
    echo "[$(date '+%H:%M:%S')] waiting for GPU (free=${f:-?}MB, need ${NEED}MB)"; sleep 120
  done
}

run_lang () {
  lang=$1
  splits=florence2_splits_bstd_${lang}_b1800.json
  vocab=grapheme_vocab_${lang}_b1800.json
  std_ckpt=checkpoints_law_${lang}_standard
  grph_ckpt=checkpoints_law_${lang}_grapheme

  echo "==================== [$lang] $(date) ===================="

  # 1) STANDARD (BPE) branch
  if [ -d ${std_ckpt}/best_model ]; then
    echo "[$lang] standard already trained, skip"
  else
    gpu_wait
    $PY -u train_florence2.py --splits_file $splits --ckpt_dir $std_ckpt \
        --filter_missing_images > train_law_${lang}_std.log 2>&1
    [ -d ${std_ckpt}/best_model ] || { echo "[$lang] STD TRAIN FAILED (see train_law_${lang}_std.log)"; return 1; }
  fi

  # 2) GRAPHEME branch
  if [ -d ${grph_ckpt}/best_model ]; then
    echo "[$lang] grapheme already trained, skip"
  else
    gpu_wait
    $PY -u train_florence2.py --use_graphemes --grapheme_vocab $vocab \
        --splits_file $splits --ckpt_dir $grph_ckpt \
        --filter_missing_images > train_law_${lang}_grph.log 2>&1
    [ -d ${grph_ckpt}/best_model ] || { echo "[$lang] GRPH TRAIN FAILED (see train_law_${lang}_grph.log)"; return 1; }
  fi

  # 3) confidence-scored predictions for both branches
  gpu_wait
  $PY -u predict_with_conf.py --tag std_law_${lang} --ckpt_dir $std_ckpt \
      --splits_file $splits > conf_law_${lang}_std.log 2>&1
  gpu_wait
  $PY -u predict_with_conf.py --tag grph_law_${lang} --ckpt_dir $grph_ckpt \
      --use_graphemes --grapheme_vocab $vocab \
      --splits_file $splits > conf_law_${lang}_grph.log 2>&1

  # 4) dual-tokenization fusion -> the per-language numbers the LAW is fit on
  $PY -u fusion_analysis.py --std std_law_${lang} --grph grph_law_${lang} \
      --self _none_ > fusion_law_${lang}.log 2>&1
  echo "[$lang] DONE $(date)"; cat fusion_law_${lang}.log
}

for L in "${LANGS[@]}"; do run_lang "$L"; done
echo "[$(date)] LAW TRAINING COMPLETE for: ${LANGS[*]}"
