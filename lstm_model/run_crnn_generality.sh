#!/usr/bin/env bash
# Architecture-generality test (Amendment 6 / PROSPECTIVE_PREDICTION_ARCHITECTURE.md):
# re-run the zero-shot LOSO rungs with a CRNN_V3 backbone instead of Florence-2,
# on the SAME splits, the SAME pivot output space and the SAME metric.
# Scripts: the three equal-synth (3240) scripts spanning the observed range.
# Resumable (skips any rung whose result JSON already exists); mirrors the
# audited run_zeroshot_loso.sh orchestrator (result skip + .train_done sentinel
# + wait_gpu).
#
# Usage:
#   nohup bash run_crnn_generality.sh > crnn_generality.log 2>&1 &
#   nohup bash run_crnn_generality.sh telugu > ... &      # subset
set -uo pipefail
cd /c/ujjwalb/ritu1/lstm_model
PY=/c/ujjwalb/.conda/envs/ritu_scenetext/bin/python

ALL_TAGS=(tamil telugu oriya)
TAGS=("$@"); [ ${#TAGS[@]} -eq 0 ] && TAGS=("${ALL_TAGS[@]}")

wait_gpu(){ while true; do f=$(nvidia-smi --query-gpu=memory.free --format=csv,noheader,nounits | head -1); [ "${f:-0}" -ge 6000 ] && break; echo "[crnn] waiting GPU (${f} MB free) $(date)"; sleep 120; done; }

run_rung(){
  local R=$1 TAG=$2
  local splits=splits_zeroshot_loso_rung${R}_${TAG}.json
  local data=splits_crnn_zs_rung${R}_${TAG}.json
  local result=result_crnn_zs_rung${R}_${TAG}.json

  if [ -f "$result" ]; then echo "[crnn] $result exists — skip"; return; fi
  if [ ! -f "$splits" ]; then echo "[crnn] MISSING $splits — skip"; return; fi

  if [ ! -f "$data" ]; then
    $PY -u prepare_crnn_zeroshot.py --rung "$R" --tag "$TAG" || return
  fi

  echo "[crnn] ===== rung $R $TAG $(date) ====="
  wait_gpu
  $PY -u train_crnn_zeroshot.py --rung "$R" --tag "$TAG" \
      > train_crnn_zs_rung${R}_${TAG}.log 2>&1 \
    || { echo "[crnn] FAILED rung $R $TAG (see train_crnn_zs_rung${R}_${TAG}.log)"; return; }
  grep -h "^\[RESULT CRNN" train_crnn_zs_rung${R}_${TAG}.log || true
}

echo "===== CRNN GENERALITY START $(date) — tags: ${TAGS[*]} ====="
for TAG in "${TAGS[@]}"; do
  run_rung A "$TAG"
  run_rung B "$TAG"
done
echo "===== CRNN GENERALITY COMPLETE $(date) ====="
ls result_crnn_zs_rung*_*.json 2>/dev/null | wc -l
