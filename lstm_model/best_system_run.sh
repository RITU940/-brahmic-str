#!/bin/bash
# Best-system track (leaderboard, NOT the controlled law): synthetic + FULL
# uncapped data + dual-tokenization fusion, Florence-2-base. Self-contained in
# /c/ujjwalb. COURTEOUS GPU GATE: waits for >=NEED MB free AND for the GPU to be
# free of OTHER users before each heavy stage (never competes with a colleague).
cd /c/ujjwalb/ritu1/lstm_model
PY=/c/ujjwalb/.conda/envs/ritu_scenetext/bin/python
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
export HF_HOME=/c/ujjwalb/.cache/huggingface
NEED=12000
ME=$(whoami)

LANGS=("$@")
if [ ${#LANGS[@]} -eq 0 ]; then
  LANGS=(bengali tamil hindi telugu kannada malayalam punjabi gujarati odia marathi assamese)
fi

others_on_gpu () {
  for p in $(nvidia-smi --query-compute-apps=pid --format=csv,noheader 2>/dev/null); do
    u=$(ps -o user= -p "$p" 2>/dev/null | tr -d " ")
    [ -n "$u" ] && [ "$u" != "$ME" ] && return 0
  done
  return 1
}

gpu_wait () {
  while true; do
    f=$(nvidia-smi --query-gpu=memory.free --format=csv,noheader,nounits | head -1)
    if [ "${f:-0}" -ge "$NEED" ] && ! others_on_gpu; then break; fi
    why=$([ "${f:-0}" -lt "$NEED" ] && echo "free=${f:-?}MB<${NEED}" || echo "other-user-active")
    echo "[$(date "+%H:%M:%S")] courteous-wait ($why)"; sleep 120
  done
}

run_lang () {
  lang=$1
  splits=florence2_splits_synth_${lang}_full.json
  vocab=grapheme_vocab_${lang}_synthfull.json
  std=checkpoints_best_${lang}_standard
  grph=checkpoints_best_${lang}_grapheme
  echo "==================== [best:$lang] $(date) ===================="
  if [ -d ${std}/best_model ]; then echo "[$lang] std done, skip"; else
    gpu_wait
    $PY -u train_florence2.py --splits_file $splits --ckpt_dir $std --filter_missing_images > train_best_${lang}_std.log 2>&1
    [ -d ${std}/best_model ] || { echo "[$lang] STD FAILED"; return 1; }
  fi
  if [ -d ${grph}/best_model ]; then echo "[$lang] grph done, skip"; else
    gpu_wait
    $PY -u train_florence2.py --use_graphemes --grapheme_vocab $vocab --splits_file $splits --ckpt_dir $grph --filter_missing_images > train_best_${lang}_grph.log 2>&1
    [ -d ${grph}/best_model ] || { echo "[$lang] GRPH FAILED"; return 1; }
  fi
  gpu_wait
  $PY -u predict_with_conf.py --tag std_best_${lang} --ckpt_dir $std --splits_file $splits > conf_best_${lang}_std.log 2>&1
  gpu_wait
  $PY -u predict_with_conf.py --tag grph_best_${lang} --ckpt_dir $grph --use_graphemes --grapheme_vocab $vocab --splits_file $splits > conf_best_${lang}_grph.log 2>&1
  $PY -u fusion_analysis.py --std std_best_${lang} --grph grph_best_${lang} --self _none_ > fusion_best_${lang}.log 2>&1
  echo "[$lang] DONE $(date)"; cat fusion_best_${lang}.log
}

for L in "${LANGS[@]}"; do run_lang "$L"; done
echo "[$(date)] BEST-SYSTEM COMPLETE for: ${LANGS[*]}"
