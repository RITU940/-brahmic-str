#!/usr/bin/env bash
# Supervised-ceiling anchor (IndicPhotoOCR PARSeq) on the 9 LOSO test sets.
# GPU-gated AND chained AFTER the Khmer rungs so it never contends with our own
# training for the card. One-shot (not critical-path for the Aug-8 deadline).
#   nohup bash run_parseq_anchor.sh > parseq_anchor.log 2>&1 &
set -uo pipefail
cd /c/ujjwalb/ritu1/lstm_model
PY=/c/ujjwalb/.conda/envs/ritu_scenetext/bin/python
export HF_HOME=/c/ujjwalb/.cache/huggingface
# 1) let the Khmer rungs finish first (don't split the GPU with our own job)
while [ "$(ls result_zs_loso_rungA_khmer.json result_zs_loso_rungB_khmer.json 2>/dev/null | wc -l)" -lt 2 ]; do
  echo "[parseq] waiting for Khmer rungs to finish $(date)"; sleep 300; done
# 2) then wait for GPU headroom (PARSeq is small; 6 GB is ample)
while true; do f=$(nvidia-smi --query-gpu=memory.free --format=csv,noheader,nounits | head -1); [ "${f:-0}" -ge 6000 ] && break; echo "[parseq] waiting GPU (${f} MB free) $(date)"; sleep 120; done
echo "[parseq] running supervised-ceiling anchor (9 scripts) $(date)"
$PY -u eval_parseq_anchor.py --scripts all --device cuda:0
echo "[parseq] done $(date) — cat result_anchor_parseq_*.json"
