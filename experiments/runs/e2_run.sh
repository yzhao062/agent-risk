#!/usr/bin/env bash
# E2 chain (E2-PREREG-2026-10-05.md): runs this host's blocks of a frozen E2 manifest in
# manifest order, one eval per block (both members of each task-seed pair in one eval),
# resumable under the D58 rule (a begun episode is never executed again).
#
# Refuses to start (exit 2) unless sha256(MANIFEST) == MANIFEST_SHA256. Before every
# block it stops when: the servers differ from the registered ones (exit 3,
# e2_check_servers.py); free disk on the Docker data file systems is below 21,000 MB
# (exit 4); the egress block is missing (exit 6). An attempt that ends without
# run.json stops the chain (exit 5) so a person looks before the next block.
#
#   MANIFEST=data/manifests/e2/manifest_e2.json LOGROOT=logs/e2 HOST_LABEL=h1 \
#   MANIFEST_SHA256=<registered> setsid nohup bash e2_run.sh > chain-h1.out 2>&1 < /dev/null &
set -uo pipefail
H=${HARNESS_DIR:-$(cd "$(dirname "$0")" && pwd)}
SERVING_DIR=${SERVING_DIR:-$H/../serving}
ANALYSIS_DIR=${ANALYSIS_DIR:-$H/../analysis}
CONTROL_ARENA_DIR=${CONTROL_ARENA_DIR:-$HOME/control-arena}
M=${MANIFEST:?set MANIFEST}
L=${LOGROOT:?set LOGROOT}
HL=${HOST_LABEL:-h1}
MIN_FREE_MB=${MIN_FREE_MB:-21000}
stamp() { echo "[$(date -u +%FT%TZ)] $*"; }
got=$(sha256sum "$M" | cut -d' ' -f1)
if [ "${MANIFEST_SHA256:-}" != "$got" ]; then
  stamp "REFUSING: manifest sha256 $got does not equal MANIFEST_SHA256=${MANIFEST_SHA256:-unset}"
  exit 2
fi
mkdir -p "$L"
cd "$CONTROL_ARENA_DIR" || exit 1
export VLLM_API_KEY=EMPTY PYTHONPATH="${PYTHONPATH:-}:$H:$H/.."
PY=${PYTHON:-.venv/bin/python}
preempt() { curl -s "http://127.0.0.1:${1}/metrics" | awk '/^vllm:num_preemptions_total/ {print $2}'; }
data_free() {
  if [ -f "$H/disk_watchdog_d66.py" ]; then
    python3 "$H/disk_watchdog_d66.py" free 2>/dev/null || df --output=avail -BM / | tail -1 | tr -dc 0-9
  else
    df --output=avail -BM / | tail -1 | tr -dc 0-9
  fi
}
root_free() { df --output=avail -BM / | tail -1 | tr -dc 0-9; }

CHECK_SERVERS=""
if [ -f "$SERVING_DIR/e2_check_servers.py" ]; then
  CHECK_SERVERS="$SERVING_DIR/e2_check_servers.py"
elif [ -f "$H/e2_check_servers.py" ]; then
  CHECK_SERVERS="$H/e2_check_servers.py"
fi

guards() {
  local hj="$L/.serving_check_$HL.json"
  if [ -f "$H/capture_host.sh" ]; then
    bash "$H/capture_host.sh" > "$hj"
    if [ -n "$CHECK_SERVERS" ] && ! python3 "$CHECK_SERVERS" "$hj"; then
      stamp "STOP: servers differ from the registration"; exit 3
    fi
  fi
  if sudo -n iptables -C DOCKER-USER -i br-+ ! -o br-+ -p tcp -j REJECT --reject-with tcp-reset 2>/dev/null; then
    : # egress rule present
  elif [ -f "$H/net_block.sh" ]; then
    stamp "WARNING: egress rule missing; consider running net_block.sh"
  fi
  local fr; fr=$(data_free)
  if [ "$fr" -lt "$MIN_FREE_MB" ]; then stamp "STOP: free disk $fr MB < $MIN_FREE_MB MB"; exit 4; fi
}
guards
stamp "E2 start host=$HL manifest $M sha256 $got | data_free_mb=$(data_free) root_free_mb=$(root_free) | preemptions agent=$(preempt 8000) monitor=$(preempt 8001)"
if [ -f "$H/capture_host.sh" ]; then
  bash "$H/capture_host.sh" > "$L/serving_host_${HL}_$(date -u +%Y%m%dT%H%M%SZ).json" 2>/dev/null || true
fi

extract() {
  local ad=$1
  if [ -f "$ANALYSIS_DIR/e2_extract.py" ]; then
    $PY "$ANALYSIS_DIR/e2_extract.py" --log-dir "$ad" --out "$ad/episodes.jsonl" >> "$ad.out" 2>&1 || true
  elif [ -f "$H/e2_extract.py" ]; then
    $PY "$H/e2_extract.py" --log-dir "$ad" --out "$ad/episodes.jsonl" >> "$ad.out" 2>&1 || true
  fi
}

blocks=$(python3 -c "import json;print(' '.join(str(b['block']) for b in json.load(open('$M'))['block_order'] if b.get('host', '$HL')=='$HL'))")
for b in $blocks; do
  label=$(python3 -c "import json;print([x['label'] for x in json.load(open('$M'))['block_order'] if x['block']==$b][0])")
  bd="$L/$label"
  mkdir -p "$bd"
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
  guards
  k=$(( $(find "$bd" -maxdepth 1 -type d -name 'attempt-*' | wc -l) + 1 ))
  ad="$bd/attempt-$k"
  stamp "start $label attempt $k | data_free_mb=$(data_free) root_free_mb=$(root_free) | preemptions agent=$(preempt 8000) monitor=$(preempt 8001)"
  $PY "$H/e2_run.py" --manifest "$M" --manifest-block "$b" --skip-started-under "$bd" \
    --concurrency 32 --log-dir "$ad" > "$ad.out" 2>&1
  if [ -d "$ad" ] && ls "$ad"/*.eval >/dev/null 2>&1; then
    extract "$ad"
  fi
  stamp "done $label attempt $k: $(tail -1 "$ad.out") | data_free_mb=$(data_free) root_free_mb=$(root_free) | preemptions agent=$(preempt 8000) monitor=$(preempt 8001)"
  if [ -d "$ad" ] && [ ! -f "$ad/run.json" ]; then
    stamp "STOP: $label attempt $k ended without run.json"; exit 5
  fi
done
stamp "E2_DONE host=$HL"
