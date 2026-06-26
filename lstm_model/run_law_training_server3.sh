#!/bin/bash
# Part ① LAW training — SERVER3 (A5000) variant. Self-contained in /c/ujjwalb.
# Uses the existing ritu_scenetext env (torch2.5.1+cu121, tf4.44.2, peft0.13).
# Dravidian-first: tamil/telugu/kannada/malayalam = decisive test of the fertility
# law (Finding 2: highest fertility => largest grapheme win). Resumable; waits for
# >=12GB free GPU; never touches other users' jobs.
cd /c/ujjwalb/ritu1/lstm_model
PY=/c/ujjwalb/.conda/envs/ritu_scenetext/bin/python
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
export HF_HOME=/c/ujjwalb/.cache/huggingface
NEED=12000

LANGS=("$@")
if [ ${#LANGS[@]} -eq 0 ]; then
  LANGS=(tamil telugu kannada malayalam bengali assamese hindi marathi gujarati punjabi odia english)
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
  if [ -d ${std_ckpt}/best_model ]; then echo "[$lang] standard done, skip"; else
    gpu_wait
    $PY -u train_florence2.py --splits_file $splits --ckpt_dir $std_ckpt --filter_missing_images > train_law_${lang}_std.log 2>&1
    [ -d ${std_ckpt}/best_model ] || { echo "[$lang] STD FAILED (train_law_${lang}_std.log)"; return 1; }
  fi
  if [ -d ${grph_ckpt}/best_model ]; then echo "[$lang] grapheme done, skip"; else
    gpu_wait
    $PY -u train_florence2.py --use_graphemes --grapheme_vocab $vocab --splits_file $splits --ckpt_dir $grph_ckpt --filter_missing_images > train_law_${lang}_grph.log 2>&1
    [ -d ${grph_ckpt}/best_model ] || { echo "[$lang] GRPH FAILED (train_law_${lang}_grph.log)"; return 1; }
  fi
  gpu_wait
  $PY -u predict_with_conf.py --tag std_law_${lang} --ckpt_dir $std_ckpt --splits_file $splits > conf_law_${lang}_std.log 2>&1
  gpu_wait
  $PY -u predict_with_conf.py --tag grph_law_${lang} --ckpt_dir $grph_ckpt --use_graphemes --grapheme_vocab $vocab --splits_file $splits > conf_law_${lang}_grph.log 2>&1
  $PY -u fusion_analysis.py --std std_law_${lang} --grph grph_law_${lang} --self _none_ > fusion_law_${lang}.log 2>&1
  echo "[$lang] DONE $(date)"; cat fusion_law_${lang}.log
}

for L in "${LANGS[@]}"; do run_lang "$L"; done
echo "[$(date)] LAW TRAINING COMPLETE for: ${LANGS[*]}"
