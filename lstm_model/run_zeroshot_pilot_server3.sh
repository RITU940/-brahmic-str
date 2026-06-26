#!/usr/bin/env bash
# Zero-shot cross-script feasibility pilot (Part ②) — server3.
# Trains Rung A (source-only) and Rung B (+synthetic target) in the shared abugida
# space, evaluates each on the held-out target's REAL test images (pivot WRR).
set -uo pipefail
cd /c/ujjwalb/ritu1/lstm_model
PY=/c/ujjwalb/.conda/envs/ritu_scenetext/bin/python
export HF_HOME=/c/ujjwalb/.cache/huggingface
TGT=${1:-tamil}

wait_gpu(){ while true; do f=$(nvidia-smi --query-gpu=memory.free --format=csv,noheader,nounits | head -1); [ "${f:-0}" -ge 12000 ] && break; echo "[pilot] waiting GPU (${f} MB free) $(date)"; sleep 120; done; }

run_rung(){
  local R=$1
  local splits=splits_zeroshot_rung${R}_${TGT}.json
  local vocab=grapheme_vocab_zeroshot_rung${R}_${TGT}.json
  local ckpt=checkpoints_zeroshot_rung${R}_${TGT}
  echo "==================== RUNG $R ($TGT) $(date) ===================="
  if [ ! -d ${ckpt}/best_model ]; then
    wait_gpu
    $PY -u train_florence2.py --use_graphemes --grapheme_vocab $vocab \
        --splits_file $splits --ckpt_dir $ckpt --filter_missing_images \
        > train_zeroshot_rung${R}_${TGT}.log 2>&1
  else
    echo "[pilot] $ckpt exists, skip train"
  fi
  $PY -u predict_with_conf.py --tag zs_rung${R}_${TGT} --ckpt_dir $ckpt \
      --use_graphemes --grapheme_vocab $vocab --splits_file $splits \
      > conf_zeroshot_rung${R}_${TGT}.log 2>&1
  $PY - "$R" "$TGT" <<'PYEOF'
import json, sys
from metrics import evaluate_corpus
R, TGT = sys.argv[1], sys.argv[2]
d = json.load(open(f'conf_zs_rung{R}_{TGT}.json'))
gts = [r['gt'] for r in d]; preds = [r['pred'] for r in d]
m = evaluate_corpus(gts, preds)
print(f"[RESULT RUNG {R} {TGT}] N={len(d)} WRR={m['WRR']:.2f} CharAcc={m['char_accuracy']:.2f} CER={m['CER']:.2f}")
PYEOF
}

echo "===== ZERO-SHOT PILOT target=$TGT (sources baked into splits) $(date) ====="
run_rung A
run_rung B
echo "----- BASELINES/CONTEXT for $TGT -----"
echo "Florence-2 raw zero-shot = 0.0 WRR (on disk) ; supervised $TGT b1800: std=28.07 grph=36.84"
echo "===== ZEROSHOT PILOT COMPLETE $(date) ====="
