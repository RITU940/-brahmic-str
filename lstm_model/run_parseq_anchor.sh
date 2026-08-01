#!/usr/bin/env bash
# Supervised-ceiling anchor (IndicPhotoOCR PARSeq) on the 9 LOSO test sets.
# GPU-gated so it never preempts a labmate's job.
#   nohup bash run_parseq_anchor.sh > parseq_anchor.log 2>&1 &
#
# Hardened after the 2026-07-31 incident: a transient exec failure made nvidia-smi,
# head, wc and sleep all unavailable at once, so the wait loop spun with no delay and
# wrote 24 GB of "command not found" into this log, filling the shared disk. Guards:
#   - PATH pinned; the gate is a python probe (no nvidia-smi/head/wc dependency)
#   - one log line per ~30 min of waiting, never one per iteration
#   - python does the sleeping, so a failed probe can never become a tight loop
#   - hard cap on wait iterations, and a free-disk floor
set -uo pipefail
export PATH=/usr/local/sbin:/usr/local/bin:/usr/sbin:/usr/bin:/sbin:/bin
cd /c/ujjwalb/ritu1/lstm_model || exit 1
PY=/c/ujjwalb/.conda/envs/ritu_scenetext/bin/python
export HF_HOME=/c/ujjwalb/.cache/huggingface

NEED_GPU_MB=6000     # PARSeq is small
NEED_DISK_MB=5000    # refuse to start if the shared disk is critically full
MAX_WAITS=336        # 336 * 120 s = ~11 h, then give up rather than loop forever

# Single gate+sleep in python: no external binaries, and it always sleeps even when
# the probe fails, so this can never become a tight loop.
"$PY" - "$NEED_GPU_MB" "$NEED_DISK_MB" "$MAX_WAITS" <<'PYGATE'
import subprocess, shutil, sys, time
need_gpu, need_disk, max_waits = (int(x) for x in sys.argv[1:4])
for i in range(max_waits):
    free_gpu = -1
    try:
        out = subprocess.run(["nvidia-smi", "--query-gpu=memory.free",
                              "--format=csv,noheader,nounits"],
                             capture_output=True, text=True, timeout=60)
        if out.returncode == 0 and out.stdout.strip():
            free_gpu = int(out.stdout.strip().splitlines()[0])
    except Exception:
        pass
    try:
        free_disk = shutil.disk_usage("/c").free // (1024 * 1024)
    except Exception:
        free_disk = -1
    if free_gpu >= need_gpu and free_disk >= need_disk:
        print(f"[parseq] gate open: {free_gpu} MB GPU, {free_disk} MB disk", flush=True)
        sys.exit(0)
    if i % 15 == 0:  # ~every 30 min, not every iteration
        print(f"[parseq] waiting: GPU {free_gpu} MB (need {need_gpu}), "
              f"disk {free_disk} MB (need {need_disk})", flush=True)
    time.sleep(120)
print("[parseq] gate never opened within the cap; giving up", flush=True)
sys.exit(1)
PYGATE
[ $? -eq 0 ] || { echo "[parseq] aborted at the gate"; exit 1; }

echo "[parseq] running supervised-ceiling anchor (9 scripts) $(date)"
"$PY" -u eval_parseq_anchor.py --scripts all --device cuda:0
echo "[parseq] exit=$? $(date) — cat result_anchor_parseq_*.json"
