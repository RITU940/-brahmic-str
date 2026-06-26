#!/bin/bash
# Resume of run_crossscript.sh after machine reboot (2026-06-10 ~18:39).
# Already done: assamese standard (15 ep, val WRR 50.38) and assamese grapheme
# best_model at epoch 12 (val WRR 53.44; killed by reboot mid-epoch 13 — LR was
# 5.7e-6 in final cosine decay, model converged; using epoch-12 best as-is).
# Remaining: assamese conf predictions + fusion, then full hindi leg.
cd /home/ujjwal/ritu1/lstm_model
PY=/home/ujjwal/miniconda3/envs/ritu_scenetext/bin/python
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
NEED=12000
gpu_wait () { while true; do f=$(nvidia-smi --query-gpu=memory.free --format=csv,noheader,nounits|head -1); [ "${f:-0}" -ge "$NEED" ] && break; sleep 120; done; }

# ---- assamese: predictions + fusion (models already trained) ----
lang=assamese
splits=florence2_splits_bstd_${lang}.json
vocab=grapheme_vocab_${lang}.json

gpu_wait
$PY -u predict_with_conf.py --tag std_${lang} --ckpt_dir checkpoints_bstd_${lang}_standard \
    --splits_file $splits > conf_${lang}_std.log 2>&1
[ -f conf_std_${lang}.json ] || { echo "[$(date)] ${lang} STD CONF FAILED"; exit 1; }

gpu_wait
$PY -u predict_with_conf.py --tag grph_${lang} --ckpt_dir checkpoints_bstd_${lang}_grapheme \
    --use_graphemes --grapheme_vocab $vocab --splits_file $splits > conf_${lang}_grph.log 2>&1
[ -f conf_grph_${lang}.json ] || { echo "[$(date)] ${lang} GRPH CONF FAILED"; exit 1; }

$PY -u fusion_analysis.py --std std_${lang} --grph grph_${lang} --self _none_ > fusion_${lang}.log 2>&1
echo "[$(date)] ${lang} DONE"; cat fusion_${lang}.log

# ---- hindi: full leg ----
lang=hindi
splits=florence2_splits_bstd_${lang}.json
vocab=grapheme_vocab_${lang}.json

gpu_wait
$PY -u train_florence2.py --splits_file $splits \
    --ckpt_dir checkpoints_bstd_${lang}_standard > train_${lang}_std.log 2>&1
[ -d checkpoints_bstd_${lang}_standard/best_model ] || { echo "[$(date)] ${lang} STD TRAIN FAILED"; exit 1; }

gpu_wait
$PY -u train_florence2.py --use_graphemes --grapheme_vocab $vocab --splits_file $splits \
    --ckpt_dir checkpoints_bstd_${lang}_grapheme > train_${lang}_grph.log 2>&1
[ -d checkpoints_bstd_${lang}_grapheme/best_model ] || { echo "[$(date)] ${lang} GRPH TRAIN FAILED"; exit 1; }

gpu_wait
$PY -u predict_with_conf.py --tag std_${lang} --ckpt_dir checkpoints_bstd_${lang}_standard \
    --splits_file $splits > conf_${lang}_std.log 2>&1
gpu_wait
$PY -u predict_with_conf.py --tag grph_${lang} --ckpt_dir checkpoints_bstd_${lang}_grapheme \
    --use_graphemes --grapheme_vocab $vocab --splits_file $splits > conf_${lang}_grph.log 2>&1

$PY -u fusion_analysis.py --std std_${lang} --grph grph_${lang} --self _none_ > fusion_${lang}.log 2>&1
echo "[$(date)] ${lang} DONE"; cat fusion_${lang}.log

echo "[$(date)] CROSS-SCRIPT COMPLETE"
