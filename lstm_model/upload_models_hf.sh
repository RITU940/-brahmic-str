#!/usr/bin/env bash
# Upload all trained best_model adapters to a PRIVATE Hugging Face model repo.
# These are the expensive-to-regenerate output (incl. the 9-day LOSO run).
#
# Prereq (one-time):  hf auth login        # paste a WRITE token
# Usage:              bash upload_models_hf.sh <hf-username>/<repo-name>
#   e.g.              bash upload_models_hf.sh ritubaskey/brahmic-str-checkpoints
#
# Re-run anytime to push newly-finished best_model adapters (idempotent).
set -euo pipefail
REPO=${1:?usage: upload_models_hf.sh <user>/<repo>}
HF=/c/ujjwalb/.conda/envs/ritu_scenetext/bin/hf
cd /c/ujjwalb/ritu1/lstm_model

echo "[hf] ensuring private repo $REPO exists"
$HF repo create "$REPO" --repo-type model --private 2>/dev/null || echo "[hf] (repo exists or create skipped)"

shopt -s nullglob
n=0
for d in checkpoints_*/best_model; do
  exp=$(dirname "$d"); exp=${exp#checkpoints_}      # folder name in the HF repo
  echo "[hf] uploading $d  ->  $REPO :/$exp"
  $HF upload "$REPO" "$d" "$exp" --repo-type model
  n=$((n+1))
done
echo "[hf] done — uploaded $n best_model adapters to https://huggingface.co/$REPO"
