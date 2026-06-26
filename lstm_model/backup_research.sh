#!/usr/bin/env bash
# =============================================================================
# backup_research.sh — off-server backup of the big artifacts git can't hold.
#
# Code / results / manuscript also go to GitHub (separate, see
# BACKUP_AND_VERSION_CONTROL.md). This script covers the rest: trained model
# adapters (best_model/) + raw datasets + annotations.
#
# Usage:
#   bash backup_research.sh user@host:/backup/ritu1     # to another machine (ssh)
#   bash backup_research.sh /mnt/external/ritu1         # to a local/mounted drive
#   FULL=1 bash backup_research.sh <dest>               # include epoch_* checkpoints too
#
# Re-run anytime: rsync is incremental, so repeats only copy what changed.
# =============================================================================
set -euo pipefail
SRC=/c/ujjwalb/ritu1/
DEST=${1:?usage: backup_research.sh <dest>  e.g. user@host:/backup/ritu1  or  /mnt/drive/ritu1}

EXCLUDES=(
  --exclude='__pycache__/' --exclude='*.pyc'
  --exclude='.cache/' --exclude='hf_home/'
  --exclude='*/langdata_lstm/.git/'      # vendored tesseract data's own git
)
# Default: skip regenerable intermediate epoch checkpoints, KEEP best_model/.
# FULL=1 includes everything (much larger).
if [ "${FULL:-0}" != "1" ]; then
  EXCLUDES+=(--exclude='checkpoints*/epoch_*')
fi

echo "[backup] $SRC -> $DEST   (FULL=${FULL:-0})   $(date)"
rsync -ah --info=progress2 --partial --mkpath "${EXCLUDES[@]}" "$SRC" "$DEST"
echo "[backup] DONE $(date)"
