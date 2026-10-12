#!/usr/bin/env bash
# E1 retry gate by targeted execution (register PLAN-RECONCILE-2026-10-04.md,
# D72; pre-registration docs/agent-risk-modeling/E1-PREREG-2026-10-05.md): run
# one half of the frozen manifest on this VM. Half "a" holds chunks 1 to 20
# (units 0 to 599), half "b" chunks 21 to 40 (units 600 to 1199); every unit's
# three conditions run on the VM of its half. Block k is one eval: base and live
# of the half's chunk k and cont of chunk k - 1, which start from the snapshots
# block k - 1 committed. Once block k has a finished attempt, chunk k - 1's
# snapshots are deleted. Snapshots are not archived; the episode logs are.
#
# Resumable, as D58 and D66. Each block lives in $LOGROOT/<label>/attempt-<k>/.
# A block is done once one of its attempts finished (run.json). An interrupted
# attempt's partial .eval is extracted, and a new attempt starts only the
# block's episodes that never began. A started episode is never executed again.
#
# Refuses to start unless the manifest's sha256 equals MANIFEST_SHA256 (exit 2).
# Stops when a block is refused because its source block is unfinished (exit 3),
# before a block when free space on the data file systems is below MIN_FREE_MB,
# default 8000 (exit 4), and when an attempt ends without run.json (exit 5).
#
#   HALF=a MANIFEST_SHA256=<registered hash> setsid nohup bash run_e1.sh \
#       > chain-a.out 2>&1 < /dev/null &
set -uo pipefail
H=${HARNESS_DIR:-$(cd "$(dirname "$0")" && pwd)}
M=${MANIFEST:-$H/../../data/manifests/e1/manifest_e1.json}
L=${LOGROOT:-$HOME/logs/e1}
ANALYSIS_DIR=${ANALYSIS_DIR:-$H/../analysis}
CONTROL_ARENA_DIR=${CONTROL_ARENA_DIR:-$HOME/control-arena}
HALF=${HALF:?set HALF=a or HALF=b}
MIN_FREE_MB=${MIN_FREE_MB:-8000}
CONC=${CONCURRENCY:-32}
mkdir -p "$L"
cd "$CONTROL_ARENA_DIR" || exit 1
export VLLM_API_KEY=EMPTY PYTHONPATH="${PYTHONPATH:-}:$H:$H/.."
PY=${PYTHON:-.venv/bin/python}
stamp() { echo "[$(date -u +%FT%TZ)] $*"; }
preempt() { curl -s "http://127.0.0.1:${1}/metrics" | awk '/^vllm:num_preemptions_total/ {print $2}'; }
free_mb() {
  if [ -f "$H/disk_watchdog_d66.py" ]; then
    python3 "$H/disk_watchdog_d66.py" free 2>/dev/null || df --output=avail -BM / | tail -1 | tr -dc 0-9
  else
    df --output=avail -BM / | tail -1 | tr -dc 0-9
  fi
}
root_mb() { df --output=avail -BM / | tail -1 | tr -dc 0-9; }
images() { docker images -q 'arh-e1' | wc -l; }
watchdogs() { pgrep -fc 'disk_watchdog' || echo 0; }

got=$(sha256sum "$M" | cut -d' ' -f1)
if [ "${MANIFEST_SHA256:-}" != "$got" ]; then
  stamp "REFUSING: manifest sha256 $got does not equal MANIFEST_SHA256=${MANIFEST_SHA256:-unset}"
  exit 2
fi
stamp "E1 start half=$HALF: manifest $M sha256 $got | free_mb=$(free_mb) | root_mb=$(root_mb) | images=$(images) | watchdogs=$(watchdogs) | preemptions agent=$(preempt 8000) monitor=$(preempt 8001)"
if [ -f "$H/capture_host.sh" ]; then
  bash "$H/capture_host.sh" > "$L/serving_host_${HALF}_$(date -u +%Y%m%dT%H%M%SZ).json" 2>/dev/null || true
fi

extract() {
  local ad=$1
  if [ -f "$ANALYSIS_DIR/extract_e1.py" ]; then
    $PY "$ANALYSIS_DIR/extract_e1.py" --log-dir "$ad" >> "$ad.out" 2>&1 || true
  elif [ -f "$H/extract_e1.py" ]; then
    $PY "$H/extract_e1.py" --log-dir "$ad" >> "$ad.out" 2>&1 || true
  fi
}

blocks=$(python3 -c "import json;print(' '.join(str(b['block']) for b in json.load(open('$M'))['block_order'] if b['half']=='$HALF'))")
for b in $blocks; do
  label=$(python3 -c "import json;print([o for o in json.load(open('$M'))['block_order'] if o['block']==$b][0]['label'])")
  bd="$L/$label"
  mkdir -p "$bd"
  for a in "$bd"/attempt-*; do
    [ -d "$a" ] || continue
    if ls "$a"/*.eval >/dev/null 2>&1 && { [ ! -f "$a/run.json" ] || [ ! -f "$a/episodes_base.jsonl" ]; }; then
      extract "$a"
    fi
  done
  if ls "$bd"/attempt-*/run.json >/dev/null 2>&1; then
    stamp "skip $label: a finished attempt exists"
  else
    fm=$(free_mb)
    if [ "$fm" -lt "$MIN_FREE_MB" ]; then
      stamp "STOP: free_mb=$fm below MIN_FREE_MB=$MIN_FREE_MB before $label | images=$(images); free disk and resume"
      exit 4
    fi
    k=$(( $(find "$bd" -maxdepth 1 -type d -name 'attempt-*' | wc -l) + 1 ))
    ad="$bd/attempt-$k"
    stamp "start $label attempt $k | free_mb=$fm | root_mb=$(root_mb) | images=$(images) | watchdogs=$(watchdogs) | preemptions agent=$(preempt 8000) monitor=$(preempt 8001)"
    $PY "$H/run_e1.py" --manifest "$M" --manifest-block "$b" --logroot "$L" --skip-started-under "$bd" \
      --concurrency "$CONC" --log-dir "$ad" > "$ad.out" 2>&1
    rc=$?
    if [ -d "$ad" ] && ls "$ad"/*.eval >/dev/null 2>&1; then
      extract "$ad"
    fi
    stamp "done $label attempt $k rc=$rc: $(tail -1 "$ad.out") | free_mb=$(free_mb) | images=$(images)"
    if [ "$rc" = "3" ]; then
      stamp "STOP: $label refused (source block unfinished); fix and resume"
      exit 3
    fi
    if [ ! -f "$ad/run.json" ]; then
      stamp "STOP: $label attempt $k ended without run.json (rc=$rc); see $ad.out, then resume"
      exit 5
    fi
  fi
  # delete the snapshots this block's cont part consumed, once the block has a finished attempt
  if ls "$bd"/attempt-*/run.json >/dev/null 2>&1 && [ ! -f "$bd/cleanup.json" ]; then
    $PY "$H/e1_snapshots.py" cleanup-block --manifest "$M" --block "$b" > "$bd/cleanup.json.tmp" 2>&1 \
      && mv "$bd/cleanup.json.tmp" "$bd/cleanup.json"
    stamp "cleanup $label: $(cat "$bd/cleanup.json" 2>/dev/null || cat "$bd/cleanup.json.tmp") | free_mb=$(free_mb) | images=$(images)"
  fi
done
$PY "$H/e1_snapshots.py" inventory-half --manifest "$M" --half "$HALF" > "$L/inventory_end_${HALF}.json" 2>&1
stamp "E1_DONE half=$HALF | snapshots left of this half: $(cat "$L/inventory_end_${HALF}.json" 2>/dev/null || true) | free_mb=$(free_mb)"
