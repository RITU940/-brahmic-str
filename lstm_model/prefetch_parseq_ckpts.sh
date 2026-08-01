#!/usr/bin/env bash
# Pre-fetch the remaining PARSeq checkpoints in parallel. eval_parseq_anchor.py's
# ensure_model() skips any checkpoint already on disk, so this just removes the
# serial ~20 min/script download from its critical path.
# Safe against the running job: we download to a private .prefetch name, verify the
# size, then rename() into place -- atomic, and an already-open fd keeps its inode.
set -u
DIR=/c/ujjwalb/ritu1/IndicPhotoOCR/IndicPhotoOCR/recognition/models
BASE=https://github.com/anikde/STocr/releases/download/V2.0.0
MIN=380000000
for m in punjabi hindi gujarati odia malayalam; do
  (
    [ -f "$DIR/$m.ckpt" ] && { echo "[prefetch] $m already present"; exit 0; }
    curl -sSL --retry 3 --retry-delay 5 -o "$DIR/$m.ckpt.prefetch" "$BASE/$m.ckpt"
    sz=$(stat -c %s "$DIR/$m.ckpt.prefetch" 2>/dev/null || echo 0)
    if [ "$sz" -ge "$MIN" ]; then
      if [ -f "$DIR/$m.ckpt" ]; then
        rm -f "$DIR/$m.ckpt.prefetch"; echo "[prefetch] $m: job got there first, discarded"
      else
        mv -f "$DIR/$m.ckpt.prefetch" "$DIR/$m.ckpt"; echo "[prefetch] $m OK ($sz bytes)"
      fi
    else
      rm -f "$DIR/$m.ckpt.prefetch"; echo "[prefetch] $m FAILED (got $sz bytes)"
    fi
  ) &
done
wait
echo "[prefetch] all done"
