#!/bin/bash
cd /c/ujjwalb/ritu1/lstm_model
PY=/c/ujjwalb/.conda/envs/ritu_scenetext/bin/python
LANGS="bengali hindi punjabi tamil telugu kannada malayalam gujarati odia marathi assamese"
NUM=10000
for L in $LANGS; do
  echo "==================== [$L] $(date) ===================="
  if [ ! -f florence2_splits_bstd_${L}_full.json ]; then
    $PY build_full_splits.py --lang $L
  fi
  if [ -f florence2_splits_synth_${L}_full.json ]; then
    echo "[$L] synth split exists, skipping"
  else
    $PY synth_multiscript.py --lang $L --num $NUM
  fi
done
echo "ALL SYNTH DONE $(date)"
