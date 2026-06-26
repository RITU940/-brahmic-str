#!/usr/bin/env bash
# Waits for the law batch to finish, THEN launches the zero-shot pilot.
# Safe: only launches the pilot if the law run reached its COMPLETE marker.
cd /c/ujjwalb/ritu1/lstm_model
while ! grep -q "LAW TRAINING COMPLETE" law_run_rest.log 2>/dev/null \
      && pgrep -f "[r]un_law_training_server3.sh" >/dev/null; do
  sleep 300
done
if grep -q "LAW TRAINING COMPLETE" law_run_rest.log 2>/dev/null; then
  sleep 60
  bash run_zeroshot_pilot_server3.sh tamil > zeroshot_pilot.log 2>&1
else
  echo "[chain] law run ended WITHOUT completion marker — pilot NOT launched $(date)" > zeroshot_pilot.log
fi
