#!/bin/bash
# Wait for GPU to free up, then evaluate both trained models on both test sets.
# Training is already complete; no pgrep dependency (that caused the earlier deadlock).
cd /home/ujjwal/ritu1/lstm_model
PY=/home/ujjwal/miniconda3/envs/ritu_scenetext/bin/python
NEED=4000   # MiB free required before launching

echo "[$(date)] waiting for >= ${NEED} MiB free GPU..."
while true; do
  free=$(nvidia-smi --query-gpu=memory.free --format=csv,noheader,nounits | head -1)
  if [ "${free:-0}" -ge "$NEED" ]; then break; fi
  sleep 120
done
echo "[$(date)] GPU has ${free} MiB free. Running evaluations."

run () {  # tag, splits_file, flags
  tag=$1; splits=$2; flags=$3
  mode=$([[ "$flags" == *use_graphemes* ]] && echo grapheme || echo standard)
  echo "[$(date)] eval ${tag} ..."
  $PY -u train_florence2.py --evaluate --splits_file "$splits" $flags > eval_${tag}.log 2>&1
  if [ -f florence2_results_${mode}.json ]; then
    mv florence2_results_${mode}.json results_${tag}.json
    echo "  saved results_${tag}.json"
  else
    echo "  WARN: no results json for ${tag} (see eval_${tag}.log)"
  fi
}

run std_ours   florence2_splits.json      ""
run grph_ours  florence2_splits.json      "--use_graphemes"
run std_bstd   florence2_splits_bstd.json ""
run grph_bstd  florence2_splits_bstd.json "--use_graphemes"

echo "[$(date)] ALL EVALS DONE"
