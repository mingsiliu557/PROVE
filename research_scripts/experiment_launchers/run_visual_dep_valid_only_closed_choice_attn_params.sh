#!/usr/bin/env bash
set -Eeuo pipefail
QUEUE_ID="${QUEUE_ID:-q-20250901110548-6w2bl}"
FLAVOR_ID="${FLAVOR_ID:-ml.pni2l.7xlarge}"
EXP_ROOT="/path/to/datasets/experiments/visual_dep_reward_valid_only_closed_choice_attn_params_20260705_085958"
TAG="20260705_085958"
RUNS=(ai2d20 realworldqa20_hq seed20 mathvista20_hq mathverse20_promptfix)
echo "[manager] exp_root=$EXP_ROOT"
mkdir -p "$EXP_ROOT"
for name in "${RUNS[@]}"; do
  worker="$EXP_ROOT/worker_${name}.sh"
  session="vdr_${name}_${TAG:9:6}"
  echo "[manager] launch $name session=$session worker=$worker" | tee -a "$EXP_ROOT/manager.log"
  tmux new-session -d -s "$session" "volc ml_devinstance launch --resource_queue_id $QUEUE_ID --flavor_id $FLAVOR_ID zsh"
  launched=0
  for i in $(seq 1 240); do
    pane=$(tmux capture-pane -pt "$session" -S -160 2>/dev/null || true)
    if printf '%s
' "$pane" | grep -q "Connect worker with ip"; then
      sleep 10
      tmux send-keys -t "$session" "bash $worker; exit" C-m
      sleep 3
      tmux send-keys -t "$session" C-m
      launched=1
      echo "[manager] command sent for $name at $(date)" | tee -a "$EXP_ROOT/manager.log"
      break
    fi
    sleep 10
  done
  if [[ "$launched" != 1 ]]; then
    echo "[manager] FAILED to connect worker for $name" | tee -a "$EXP_ROOT/manager.log"
    tmux kill-session -t "$session" || true
    exit 1
  fi
  while tmux has-session -t "$session" 2>/dev/null; do
    sleep 60
  done
  if [[ -f "$EXP_ROOT/$name/SUCCESS" ]]; then
    echo "[manager] SUCCESS $name at $(date)" | tee -a "$EXP_ROOT/manager.log"
  else
    echo "[manager] FAILED $name at $(date)" | tee -a "$EXP_ROOT/manager.log"
    exit 1
  fi
  sleep 20
done
touch "$EXP_ROOT/SUCCESS"
echo "[manager] all done $(date)" | tee -a "$EXP_ROOT/manager.log"
