#!/usr/bin/env bash
# Amendment-4c frontier-VLM baseline (Qwen2.5-VL-7B) — needs the FULL GPU
# (~17 GB bf16), so it runs only in the window between the LOSO run and the
# scaling sweep (sequenced by .monitor/scale_chain.sh) or manually when free.
set -uo pipefail
cd /c/ujjwalb/ritu1/lstm_model
export HF_HOME=/c/ujjwalb/.cache/huggingface
export HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1   # model pre-downloaded
VPY=/c/ujjwalb/ritu1/.tools/vlm_venv/bin/python
# refuse to start if training holds the GPU
f=$(nvidia-smi --query-gpu=memory.free --format=csv,noheader,nounits | head -1)
if [ "${f:-0}" -lt 20000 ]; then
  echo "[vlm] GPU not free enough (${f} MB) — refusing to start"; exit 1
fi
exec $VPY -u eval_vlm_baseline.py
