#!/usr/bin/env bash
# Khmer out-of-benchmark LOSO (Amendment 4b) — Rung A + Rung B ONLY (no BPE phase).
# Mirrors run_zeroshot_loso.sh:run_rung exactly (train_florence2 grapheme-injected,
# predict_with_conf, evaluate_corpus). GPU-GATED: waits for >=12 GB free so it never
# preempts a labmate's job. Resumable (skips a rung whose result JSON already exists;
# resumes on the .train_done sentinel). Splits/vocab built by prepare_zeroshot_loso_khmer.py.
#
#   nohup bash run_khmer_loso.sh > khmer_loso.log 2>&1 &
set -uo pipefail
cd /c/ujjwalb/ritu1/lstm_model
PY=/c/ujjwalb/.conda/envs/ritu_scenetext/bin/python
export HF_HOME=/c/ujjwalb/.cache/huggingface
export HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1
TAG=khmer

wait_gpu(){ while true; do f=$(nvidia-smi --query-gpu=memory.free --format=csv,noheader,nounits | head -1); [ "${f:-0}" -ge 12000 ] && break; echo "[khmer] waiting GPU (${f} MB free) $(date)"; sleep 120; done; }

run_rung(){
  local R=$1
  local splits=splits_zeroshot_loso_rung${R}_${TAG}.json
  local vocab=grapheme_vocab_zeroshot_loso_rung${R}_${TAG}.json
  local ckpt=checkpoints_zeroshot_loso_rung${R}_${TAG}
  local result=result_zs_loso_rung${R}_${TAG}.json
  echo "==================== KHMER RUNG $R $(date) ===================="
  if [ -f "$result" ]; then echo "[khmer] $result exists, skip rung"; return; fi
  if [ ! -f "$splits" ]; then echo "[khmer] MISSING $splits — run prepare_zeroshot_loso_khmer.py first"; return; fi
  if [ ! -f ${ckpt}/.train_done ]; then
    wait_gpu
    if ! $PY -u train_florence2.py --use_graphemes --grapheme_vocab "$vocab" \
        --splits_file $splits --ckpt_dir $ckpt --filter_missing_images \
        > train_zeroshot_loso_rung${R}_${TAG}.log 2>&1; then
      echo "[khmer] TRAIN FAILED rung $R — see train_zeroshot_loso_rung${R}_${TAG}.log"; return
    fi
    touch ${ckpt}/.train_done
  else
    echo "[khmer] $ckpt training complete, skip train"
  fi
  $PY -u predict_with_conf.py --tag zs_loso_rung${R}_${TAG} --ckpt_dir $ckpt \
      --use_graphemes --grapheme_vocab "$vocab" --splits_file $splits \
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
print(f"[RESULT KHMER RUNG {R}] N={rec['N']} WRR={rec['WRR']} CharAcc={rec['CharAcc']} CER={rec['CER']}")
open(f'result_zs_loso_rung{R}_{TAG}.json','w').write(json.dumps(rec))
PYEOF
}

echo "===== KHMER LOSO START $(date) ====="
run_rung A
run_rung B
echo "===== KHMER LOSO COMPLETE $(date) — cat result_zs_loso_rung{A,B}_khmer.json ====="
