#!/usr/bin/env bash
# E7 chain (pre-registration docs/agent-risk-modeling/E7-PREREG-2026-10-06.md): run this host's
# blocks of the frozen manifest in order, one eval per block: base and live (gates 1 and 2) of the
# host's chunk k and cont (gates 1 and 2) of its chunk k - 1, which start from the snapshots block
# k - 1 committed. Once block k has a finished attempt, chunk k - 1's snapshots are deleted.
# Snapshots are not archived; the episode logs are.
#
# Resumable, as D58, E1 and E2. Each block lives in $LOGROOT/<label>/attempt-<k>/. A block is done
# once one of its attempts finished (run.json). An interrupted attempt's partial .eval is
# extracted, and a new attempt starts only the block's episodes that never began. A started
# episode is never executed again.
#
# Refuses to start (exit 2) unless sha256(MANIFEST) == MANIFEST_SHA256. Before every block it stops
# when: the servers differ from E2's registered ones (exit 3); free disk is below MIN_FREE_MB (exit 4);
# the egress block is missing (exit 6); the weights directory is missing (exit 3).
#
#   HOST=1 MANIFEST_SHA256=<registered hash> setsid nohup bash run_e7.sh \
#       > chain-h1.out 2>&1 < /dev/null &
set -uo pipefail
H=${HARNESS_DIR:-$(cd "$(dirname "$0")" && pwd)}
SERVING_DIR=${SERVING_DIR:-$H/../serving}
ANALYSIS_DIR=${ANALYSIS_DIR:-$H/../analysis}
CONTROL_ARENA_DIR=${CONTROL_ARENA_DIR:-$HOME/control-arena}
M=${MANIFEST:-$H/../../data/manifests/e7/manifest_e7.json}
L=${LOGROOT:-$HOME/logs/e7}
HOST=${HOST:?set HOST=<host number>}
HL=h$HOST
MIN_FREE_MB=${MIN_FREE_MB:-16000}
CONC=${CONCURRENCY:-32}
E7_HF=${E7_HF:-/mnt/e7hf}
mkdir -p "$L"
stamp() { echo "[$(date -u +%FT%TZ)] $*"; }
got=$(sha256sum "$M" | cut -d' ' -f1)
if [ "${MANIFEST_SHA256:-}" != "$got" ]; then
  stamp "REFUSING: manifest sha256 $got does not equal MANIFEST_SHA256=${MANIFEST_SHA256:-unset}"
  exit 2
fi
SUMS_FILE="${SHA256SUMS_FILE:-$H/../../data/manifests/e7/SHA256SUMS}"
if [ -f "$SUMS_FILE" ]; then
  if ! (cd "$H" && sha256sum -c "$SUMS_FILE" > "$L/sha256sums_check_$HL.txt" 2>&1); then
    stamp "WARNING: SHA256SUMS verification found mismatches (see $L/sha256sums_check_$HL.txt)"
  fi
fi

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
images() { docker images -q 'arh-e7' | wc -l; }

CHECK_SERVERS=""
if [ -f "$SERVING_DIR/e2_check_servers.py" ]; then
  CHECK_SERVERS="$SERVING_DIR/e2_check_servers.py"
elif [ -f "$H/e2_check_servers.py" ]; then
  CHECK_SERVERS="$H/e2_check_servers.py"
fi

guards() {
  local hj="$L/.serving_check_$HL.json"
  [ -f "$E7_HF/hub/models--Qwen--Qwen3.5-27B-FP8/snapshots/97f5941bf617e31c5e237364a8602ce3f03a551a/config.json" ] \
    || { stamp "STOP: agent weights missing under $E7_HF"; exit 3; }
  if [ -f "$H/capture_host.sh" ]; then
    bash "$H/capture_host.sh" > "$hj"
    if [ -n "$CHECK_SERVERS" ] && ! python3 "$CHECK_SERVERS" "$hj"; then
      stamp "STOP: servers differ from the registration"; exit 3
    fi
  fi
  if sudo -n iptables -C DOCKER-USER -i br-+ ! -o br-+ -p tcp -j REJECT --reject-with tcp-reset 2>/dev/null; then
    : # egress rule present
  elif [ -f "$H/net_block.sh" ]; then
    stamp "WARNING: egress block missing (run net_block.sh)"
  fi
  local fr; fr=$(data_free)
  if [ "$fr" -lt "$MIN_FREE_MB" ]; then stamp "STOP: free disk $fr MB < $MIN_FREE_MB MB (images=$(images)); free disk and resume"; exit 4; fi
}
guards
stamp "E7 start host=$HL: manifest $M sha256 $got | data_free_mb=$(data_free) root_free_mb=$(root_free) | images=$(images) | preemptions agent=$(preempt 8000) monitor=$(preempt 8001)"
if [ -f "$H/capture_host.sh" ]; then
  bash "$H/capture_host.sh" > "$L/serving_host_${HL}_$(date -u +%Y%m%dT%H%M%SZ).json" 2>/dev/null || true
fi

extract() {
  local ad=$1
  if [ -f "$ANALYSIS_DIR/extract_e7.py" ]; then
    $PY "$ANALYSIS_DIR/extract_e7.py" --log-dir "$ad" >> "$ad.out" 2>&1 || true
  elif [ -f "$H/extract_e7.py" ]; then
    $PY "$H/extract_e7.py" --log-dir "$ad" >> "$ad.out" 2>&1 || true
  fi
}

blocks=$(python3 -c "import json;print(' '.join(str(b['block']) for b in json.load(open('$M'))['block_order'] if b.get('host')==$HOST))")
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
    guards
    k=$(( $(find "$bd" -maxdepth 1 -type d -name 'attempt-*' | wc -l) + 1 ))
    ad="$bd/attempt-$k"
    stamp "start $label attempt $k | data_free_mb=$(data_free) root_free_mb=$(root_free) | images=$(images) | preemptions agent=$(preempt 8000) monitor=$(preempt 8001)"
    $PY "$H/run_e7.py" --manifest "$M" --manifest-block "$b" --logroot "$L" --skip-started-under "$bd" \
      --concurrency "$CONC" --log-dir "$ad" > "$ad.out" 2>&1
    rc=$?
    if [ -d "$ad" ] && ls "$ad"/*.eval >/dev/null 2>&1; then
      extract "$ad"
    fi
    stamp "done $label attempt $k rc=$rc: $(tail -1 "$ad.out" | cut -c1-200) | data_free_mb=$(data_free) root_free_mb=$(root_free) | images=$(images)"
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
    $PY "$H/e7_snapshots.py" cleanup-block --manifest "$M" --block "$b" > "$bd/cleanup.json.tmp" 2>&1 \
      && mv "$bd/cleanup.json.tmp" "$bd/cleanup.json"
    stamp "cleanup $label: $(cat "$bd/cleanup.json" 2>/dev/null || cat "$bd/cleanup.json.tmp") | data_free_mb=$(data_free) | images=$(images)"
  fi
done
$PY "$H/e7_snapshots.py" inventory --manifest "$M" --host "$HOST" > "$L/inventory_end_${HL}.json" 2>&1
stamp "E7_DONE host=$HL | snapshots left of this host: $(cat "$L/inventory_end_${HL}.json" 2>/dev/null || true) | data_free_mb=$(data_free)"
