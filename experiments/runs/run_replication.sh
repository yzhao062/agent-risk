#!/usr/bin/env bash
# D58 confirmatory replication batch (register PLAN-RECONCILE-2026-10-04.md,
# Section 0.4): 20 alternating blocks of 100 episodes from the frozen manifest
# (B2 baseline gate, Val3 stopping gate at theta 0.94), in manifest order, on
# one VM with batch-invariant serving, concurrency 32, full request logging.
# Also used for D64 (paired announcement test) and D65 (long-horizon log).
#
# Resumable (register Section 0.4, missing outcomes and interrupted episodes).
# Each block lives in $LOGROOT/<label>/attempt-<k>/. A block is done once one of
# its attempts finished (run.json written). If an attempt was interrupted (no
# run.json), the records of its partial .eval are recovered with
# extract_episodes.py, and a new attempt starts only the block's manifest
# episodes that never began: every sample that began is listed in the attempt's
# started.jsonl (an Inspect hook) or appears in its log. A started episode is
# never executed again; if its records cannot be recovered, its outcome is
# unknown and analyze_replication.py treats it as such.
#
# The script refuses to start unless the manifest's sha256 equals MANIFEST_SHA256.
#   MANIFEST_SHA256=<registered hash> setsid nohup bash run_replication.sh \
#       > chain.out 2>&1 < /dev/null &
set -uo pipefail
H=${HARNESS_DIR:-$(cd "$(dirname "$0")" && pwd)}
M=${MANIFEST:-$H/../../data/manifests/d58/manifest_d58.json}
L=${LOGROOT:-$HOME/logs/d58}
ANALYSIS_DIR=${ANALYSIS_DIR:-$H/../analysis}
CONTROL_ARENA_DIR=${CONTROL_ARENA_DIR:-$HOME/control-arena}
mkdir -p "$L"
cd "$CONTROL_ARENA_DIR" || exit 1
export VLLM_API_KEY=EMPTY PYTHONPATH="${PYTHONPATH:-}:$H:$H/.."
PY=${PYTHON:-.venv/bin/python}
stamp() { echo "[$(date -u +%FT%TZ)] $*"; }
preempt() { curl -s "http://127.0.0.1:${1}/metrics" | awk '/^vllm:num_preemptions_total/ {print $2}'; }
free_mb() { df --output=avail -BM / | tail -1 | tr -dc 0-9; }

got=$(sha256sum "$M" | cut -d' ' -f1)
if [ "${MANIFEST_SHA256:-}" != "$got" ]; then
  stamp "REFUSING: manifest sha256 $got does not equal MANIFEST_SHA256=${MANIFEST_SHA256:-unset}"
  exit 2
fi
stamp "Replication start: manifest $M sha256 $got | free_mb=$(free_mb) | preemptions agent=$(preempt 8000) monitor=$(preempt 8001)"
if [ -f "$H/capture_host.sh" ]; then
  bash "$H/capture_host.sh" > "$L/serving_host_$(date -u +%Y%m%dT%H%M%SZ).json" 2>/dev/null || true
fi

extract() {
  local ad=$1
  if [ -f "$ANALYSIS_DIR/extract_episodes.py" ]; then
    $PY "$ANALYSIS_DIR/extract_episodes.py" --log-dir "$ad" --out "$ad/episodes.jsonl" >> "$ad.out" 2>&1 || true
  elif [ -f "$H/extract_episodes.py" ]; then
    $PY "$H/extract_episodes.py" --log-dir "$ad" --out "$ad/episodes.jsonl" >> "$ad.out" 2>&1 || true
  fi
}

nblocks=$(python3 -c "import json;print(len(json.load(open('$M'))['block_order']))")
for b in $(seq 1 "$nblocks"); do
  label=$(python3 -c "import json;print([x['label'] for x in json.load(open('$M'))['block_order'] if x['block']==$b][0])")
  bd="$L/$label"
  mkdir -p "$bd"
  # extract any attempt that has a log but no episodes file (interrupted)
  for a in "$bd"/attempt-*; do
    [ -d "$a" ] || continue
    if ls "$a"/*.eval >/dev/null 2>&1 && { [ ! -f "$a/run.json" ] || [ ! -f "$a/episodes.jsonl" ]; }; then
      extract "$a"
    fi
  done
  if ls "$bd"/attempt-*/run.json >/dev/null 2>&1; then
    stamp "skip $label: a finished attempt exists"
    continue
  fi
  k=$(( $(find "$bd" -maxdepth 1 -type d -name 'attempt-*' | wc -l) + 1 ))
  ad="$bd/attempt-$k"
  stamp "start $label attempt $k | free_mb=$(free_mb) | preemptions agent=$(preempt 8000) monitor=$(preempt 8001)"
  $PY "$H/run_pilot.py" --manifest "$M" --manifest-block "$b" --skip-started-under "$bd" \
    --concurrency 32 --log-dir "$ad" > "$ad.out" 2>&1
  if [ -d "$ad" ] && ls "$ad"/*.eval >/dev/null 2>&1; then
    extract "$ad"
  fi
  stamp "done $label attempt $k: $(tail -1 "$ad.out") | free_mb=$(free_mb) | preemptions agent=$(preempt 8000) monitor=$(preempt 8001)"
done
stamp "REPLICATION_DONE"
