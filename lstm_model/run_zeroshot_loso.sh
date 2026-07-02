#!/usr/bin/env bash
# Full leave-one-SCRIPT-out zero-shot cross-script STR (Part ②) — server3.
# For each held-out Brahmic script: train Rung A (sources only) and Rung B
# (+synthetic target, NO real target images) in the shared abugida pivot space,
# then evaluate pivot WRR on the held-out script's REAL test images.
# Resumable (skips any rung whose checkpoint + result already exist).
#
# After the main 18 rungs, a BPE-BASELINE phase (Rung Bbpe: same splits/recipe,
# stock tokenizer, no grapheme injection — PREREGISTRATION.md §7 + §8 Amendment 3)
# runs for the same tags, tamil/telugu first.
#
# Usage:
#   nohup bash run_zeroshot_loso.sh > zeroshot_loso.log 2>&1 &      # all 9
#   nohup bash run_zeroshot_loso.sh tamil telugu > ... &           # subset (tags)
# Tags = script.lower(): bengali devanagari gujarati gurmukhi kannada
#        malayalam oriya tamil telugu
set -uo pipefail
cd /c/ujjwalb/ritu1/lstm_model
PY=/c/ujjwalb/.conda/envs/ritu_scenetext/bin/python
export HF_HOME=/c/ujjwalb/.cache/huggingface
# Model/tokenizer fully cached locally; offline mode removes network as a failure
# mode (the 2026-06-30 power outage flooded logs with HF DNS retries).
export HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1

ALL_TAGS=(tamil telugu kannada malayalam oriya gujarati bengali devanagari gurmukhi)
TAGS=("$@"); [ ${#TAGS[@]} -eq 0 ] && TAGS=("${ALL_TAGS[@]}")

wait_gpu(){ while true; do f=$(nvidia-smi --query-gpu=memory.free --format=csv,noheader,nounits | head -1); [ "${f:-0}" -ge 12000 ] && break; echo "[loso] waiting GPU (${f} MB free) $(date)"; sleep 120; done; }

run_rung(){
  local R=$1 TAG=$2
  local SR=${R%bpe}   # splits rung: Bbpe reuses Rung B's splits (identical protocol)
  local splits=splits_zeroshot_loso_rung${SR}_${TAG}.json
  local vocab=grapheme_vocab_zeroshot_loso_rung${SR}_${TAG}.json
  local ckpt=checkpoints_zeroshot_loso_rung${R}_${TAG}
  local conf=conf_zs_loso_rung${R}_${TAG}.json
  local result=result_zs_loso_rung${R}_${TAG}.json
  # Grapheme injection everywhere EXCEPT the Bbpe baseline (stock tokenizer).
  local GRAPH_ARGS=(--use_graphemes --grapheme_vocab "$vocab")
  [[ $R == *bpe ]] && GRAPH_ARGS=()
  echo "==================== RUNG $R ($TAG) $(date) ===================="
  if [ -f "$result" ]; then echo "[loso] $result exists, skip rung"; return; fi
  if [ ! -f "$splits" ]; then echo "[loso] MISSING $splits — run prepare_zeroshot_loso.py first"; return; fi
  # Resume on the completion SENTINEL, not best_model: a mid-training crash leaves
  # a partial best_model behind (kannada B, 2026-06-30 outage) which must retrain.
  if [ ! -f ${ckpt}/.train_done ]; then
    wait_gpu
    if ! $PY -u train_florence2.py "${GRAPH_ARGS[@]}" \
        --splits_file $splits --ckpt_dir $ckpt --filter_missing_images \
        > train_zeroshot_loso_rung${R}_${TAG}.log 2>&1; then
      echo "[loso] TRAIN FAILED rung $R $TAG — skipping eval (see train_zeroshot_loso_rung${R}_${TAG}.log)"
      return
    fi
    touch ${ckpt}/.train_done
  else
    echo "[loso] $ckpt training complete, skip train"
  fi
  $PY -u predict_with_conf.py --tag zs_loso_rung${R}_${TAG} --ckpt_dir $ckpt \
      "${GRAPH_ARGS[@]}" --splits_file $splits \
      > conf_zeroshot_loso_rung${R}_${TAG}.log 2>&1
  $PY - "$R" "$TAG" <<'PYEOF'
import json, sys
from metrics import evaluate_corpus
R, TAG = sys.argv[1], sys.argv[2]
d = json.load(open(f'conf_zs_loso_rung{R}_{TAG}.json'))
gts = [r['gt'] for r in d]; preds = [r['pred'] for r in d]
m = evaluate_corpus(gts, preds)
rec = {'rung': R, 'script': TAG, 'N': len(d), 'WRR': round(m['WRR'],2),
       'CharAcc': round(m['char_accuracy'],2), 'CER': round(m['CER'],2)}
print(f"[RESULT LOSO RUNG {R} {TAG}] N={rec['N']} WRR={rec['WRR']} "
      f"CharAcc={rec['CharAcc']} CER={rec['CER']}")
open(f'result_zs_loso_rung{R}_{TAG}.json','w').write(json.dumps(rec))
PYEOF
}

echo "===== ZERO-SHOT LOSO START $(date) — tags: ${TAGS[*]} ====="
for TAG in "${TAGS[@]}"; do
  echo "########## HELD-OUT SCRIPT: $TAG $(date) ##########"
  run_rung A "$TAG"
  run_rung B "$TAG"
done
echo "===== ZERO-SHOT LOSO MAIN PHASE COMPLETE $(date) ====="

# --- BPE-baseline phase (prereg §7 baseline / §8 Amendment 3) ---------------
# Same tags, tamil+telugu first (their grapheme Rung-B numbers already exist,
# so these two complete the key ablation soonest).
BPE_TAGS=()
for t in tamil telugu; do
  for x in "${TAGS[@]}"; do [ "$x" = "$t" ] && BPE_TAGS+=("$t"); done
done
for x in "${TAGS[@]}"; do
  [ "$x" != tamil ] && [ "$x" != telugu ] && BPE_TAGS+=("$x")
done
echo "===== BPE-BASELINE PHASE START $(date) — tags: ${BPE_TAGS[*]} ====="
for TAG in "${BPE_TAGS[@]}"; do
  run_rung Bbpe "$TAG"
done
echo "===== ZERO-SHOT LOSO COMPLETE (incl. BPE baseline) $(date) ====="
echo "Collect results: cat result_zs_loso_rung*_*.json"
