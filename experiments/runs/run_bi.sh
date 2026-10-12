#!/usr/bin/env bash
# Batch-invariant exactness check (added at the coordinator's request after the
# default-kernel validation 1 showed serving nondeterminism under concurrency).
# Waits for the stage-2 chain to finish, restarts both vLLM servers with
# VLLM_BATCH_INVARIANT=1 (all other flags unchanged, same H100), then runs the
# first 100 B0 episodes (same tasks and seeds) under the baseline gate and again
# under the stopping gate at the middle theta, both at concurrency 32, with full
# request logs. Exactness is judged between these two batch-invariant runs.
#   setsid nohup bash run_bi.sh > bi.out 2>&1 &
set -uo pipefail
L=${LOGROOT:-$HOME/logs/pilot-ab}
H=${HARNESS_DIR:-$(cd "$(dirname "$0")" && pwd)}
ANALYSIS_DIR=${ANALYSIS_DIR:-$H/../analysis}
SERVING_DIR=${SERVING_DIR:-$H/../serving}
CONTROL_ARENA_DIR=${CONTROL_ARENA_DIR:-$HOME/control-arena}
SERVE_SH=${SERVE_SH:-$SERVING_DIR/serve.sh}
[ -f "$SERVE_SH" ] || SERVE_SH="$H/serve.sh"
stamp() { echo "[$(date -u +%FT%TZ)] $*"; }
until grep -q STAGE2_DONE "$L/stage2.out" 2>/dev/null; do sleep 30; done
T2=$(awk '{print $2}' "$L/theta_grid.txt" 2>/dev/null || echo "0.94")
stamp "stage 2 done; restarting servers with VLLM_BATCH_INVARIANT=1 (theta $T2)"
pkill -f "bin/vllm serve" || true
for i in $(seq 1 60); do pgrep -f "bin/vllm serve" >/dev/null || break; sleep 2; done
mkdir -p "$L/vm-default-kernels"
cp "$HOME/logs/vllm-agent.log" "$HOME/logs/vllm-monitor.log" "$L/vm-default-kernels/" 2>/dev/null || true
export VLLM_BATCH_INVARIANT=1
AGENT_UTIL=0.55 MON_UTIL=0.30 AGENT_EXTRA=--no-enable-prefix-caching MON_EXTRA=--no-enable-prefix-caching \
  bash "$SERVE_SH" > "$L/serve-bi.log" 2>&1
if [ -f "$H/capture_host.sh" ]; then
  bash "$H/capture_host.sh" > "$L/serving_host_bi.json" 2>/dev/null || true
fi
stamp "servers up: $(grep -c 'up after' "$L/serve-bi.log" 2>/dev/null || true) of 2"
cd "$CONTROL_ARENA_DIR" || exit 1
export VLLM_API_KEY=EMPTY PYTHONPATH="${PYTHONPATH:-}:$H:$H/.."
PY=${PYTHON:-.venv/bin/python}
for spec in "bi_baseline baseline" "bi_stopping stopping"; do
  set -- $spec
  extra=""; [ "$2" = stopping ] && extra="--theta $T2"
  stamp "start $1"
  $PY "$H/run_pilot.py" --stage "$1" --gate "$2" $extra --plan-file "$L/b0/plan.json" \
    --select 0:100 --concurrency 32 --log-dir "$L/$1" > "$L/$1.out" 2>&1
  if [ -f "$ANALYSIS_DIR/extract_episodes.py" ]; then
    $PY "$ANALYSIS_DIR/extract_episodes.py" --log-dir "$L/$1" --out "$L/$1/episodes.jsonl" >> "$L/$1.out" 2>&1
  elif [ -f "$H/extract_episodes.py" ]; then
    $PY "$H/extract_episodes.py" --log-dir "$L/$1" --out "$L/$1/episodes.jsonl" >> "$L/$1.out" 2>&1
  fi
  stamp "done $1: $(tail -1 "$L/$1.out")"
done
stamp "BI_DONE"
