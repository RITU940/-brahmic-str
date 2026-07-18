#!/usr/bin/env bash
# Amendment-4a/5 synthetic-exposure scaling sweep — server3.
# 12 runs: {malayalam, kannada, telugu} x budgets {6480, 1620, 810, 12960}
# (3240 = existing Rung-B result, reused as the middle point).
# Recipe identical to Rung B (grapheme injection ON; same trainer/eval/metric).
# Run order fixed by PREREGISTRATION Amendment 5.5: 6480 -> 1620 -> 810 -> 12960,
# each phase over (malayalam, kannada, telugu).
# Resumable exactly like run_zeroshot_loso.sh (result JSON skip + .train_done sentinel).
#
# Usage: nohup bash run_scaling_sweep.sh > scaling_sweep.log 2>&1 &
set -uo pipefail
cd /c/ujjwalb/ritu1/lstm_model
PY=/c/ujjwalb/.conda/envs/ritu_scenetext/bin/python
export HF_HOME=/c/ujjwalb/.cache/huggingface
export HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1

BUDGETS=(6480 1620 810 12960)          # Amendment 5.5 order
TAGS=(malayalam kannada telugu)

wait_gpu(){ while true; do f=$(nvidia-smi --query-gpu=memory.free --format=csv,noheader,nounits | head -1); [ "${f:-0}" -ge 12000 ] && break; echo "[scale] waiting GPU (${f} MB free) $(date)"; sleep 120; done; }

run_scale(){
  local B=$1 TAG=$2
  local splits=splits_zs_scale${B}_${TAG}.json
  local vocab=grapheme_vocab_zs_scale${B}_${TAG}.json
  local ckpt=checkpoints_zs_scale${B}_${TAG}
  local result=result_zs_scale${B}_${TAG}.json
  echo "==================== SCALE $B ($TAG) $(date) ===================="
  if [ -f "$result" ]; then echo "[scale] $result exists, skip"; return; fi
  if [ ! -f "$splits" ]; then echo "[scale] MISSING $splits — run prepare_scaling_sweep.py"; return; fi
  if [ ! -f ${ckpt}/.train_done ]; then
    wait_gpu
    if ! $PY -u train_florence2.py --use_graphemes --grapheme_vocab "$vocab" \
        --splits_file $splits --ckpt_dir $ckpt --filter_missing_images \
        > train_zs_scale${B}_${TAG}.log 2>&1; then
      echo "[scale] TRAIN FAILED $B $TAG — skipping eval (see train_zs_scale${B}_${TAG}.log)"
      return
    fi
    touch ${ckpt}/.train_done
  else
    echo "[scale] $ckpt training complete, skip train"
  fi
  $PY -u predict_with_conf.py --tag zs_scale${B}_${TAG} --ckpt_dir $ckpt \
      --use_graphemes --grapheme_vocab "$vocab" --splits_file $splits \
      > conf_zs_scale${B}_${TAG}.log 2>&1
  $PY - "$B" "$TAG" <<'PYEOF'
import json, sys
from metrics import evaluate_corpus
B, TAG = sys.argv[1], sys.argv[2]
d = json.load(open(f'conf_zs_scale{B}_{TAG}.json'))
gts = [r['gt'] for r in d]; preds = [r['pred'] for r in d]
m = evaluate_corpus(gts, preds)
rec = {'rung': f'scale{B}', 'script': TAG, 'N': len(d), 'WRR': round(m['WRR'],2),
       'CharAcc': round(m['char_accuracy'],2), 'CER': round(m['CER'],2)}
print(f"[RESULT SCALE {B} {TAG}] N={rec['N']} WRR={rec['WRR']} "
      f"CharAcc={rec['CharAcc']} CER={rec['CER']}")
open(f'result_zs_scale{B}_{TAG}.json','w').write(json.dumps(rec))
PYEOF
}

echo "===== SCALING SWEEP START $(date) ====="
for B in "${BUDGETS[@]}"; do
  for TAG in "${TAGS[@]}"; do
    run_scale "$B" "$TAG"
  done
done
echo "===== SCALING SWEEP COMPLETE $(date) ====="
echo "Collect: cat result_zs_scale*_*.json ; score vs PROSPECTIVE_PREDICTION_SCALING.md"
