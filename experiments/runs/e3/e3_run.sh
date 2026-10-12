#!/usr/bin/env bash
# E3 chain (E3-PREREG-2026-10-05.md): runs this host's blocks of a frozen E3 manifest in
# manifest order, one runner process per block (all members of each unit in one process
# against the same two servers), resumable under the D58 rule (a begun episode is never
# executed again).
#
# Refuses to start (exit 2) unless sha256(MANIFEST) == MANIFEST_SHA256. Before every
# block it stops when: the servers differ from the registration (exit 3); free
# space under ~/logs is below 6,000 MB (exit 4); the e3net egress rule is missing (exit 6).
# An attempt that ends without run.json stops the chain (exit 5).
#
#   MANIFEST=data/manifests/e3/manifest_e3.json LOGROOT=logs/e3 HOST_LABEL=h1 \
#   MANIFEST_SHA256=<registered> CONCURRENCY=32 setsid nohup bash e3/e3_run.sh \
#     > chain-h1.out 2>&1 < /dev/null &
set -uo pipefail
H=${HARNESS_DIR:-$(cd "$(dirname "$0")/.." && pwd)}
E=${E3_DIR:-$(cd "$(dirname "$0")" && pwd)}
SERVING_DIR=${SERVING_DIR:-$H/../serving}
M=${MANIFEST:?set MANIFEST}
L=${LOGROOT:?set LOGROOT}
HL=${HOST_LABEL:-h1}
CONC=${CONCURRENCY:-32}
MIN_FREE_MB=${MIN_FREE_MB:-6000}
PY=${TAU2_PYTHON:-$HOME/venv-tau2/bin/python}
stamp() { echo "[$(date -u +%FT%TZ)] $*"; }
got=$(sha256sum "$M" | cut -d' ' -f1)
if [ "${MANIFEST_SHA256:-}" != "$got" ]; then
  stamp "REFUSING: manifest sha256 $got does not equal MANIFEST_SHA256=${MANIFEST_SHA256:-unset}"
  exit 2
fi
SUMS_FILE="${SHA256SUMS_FILE:-$H/../../data/manifests/e3/SHA256SUMS}"
if [ -f "$SUMS_FILE" ]; then
  if ! (cd "$H" && sha256sum -c "$SUMS_FILE" 2>/dev/null); then
    stamp "WARNING: SHA256SUMS verification had mismatches"
  fi
fi
mkdir -p "$L"
preempt() { curl -s "http://127.0.0.1:${1}/metrics" | awk '/^vllm:num_preemptions_total/ {print $2}'; }
free_mb() { df --output=avail -BM "$L" 2>/dev/null | tail -1 | tr -dc 0-9; }

CHECK_SERVERS=""
if [ -f "$SERVING_DIR/e3_check_servers.py" ]; then
  CHECK_SERVERS="$SERVING_DIR/e3_check_servers.py"
elif [ -f "$E/e3_check_servers.py" ]; then
  CHECK_SERVERS="$E/e3_check_servers.py"
fi

guards() {
  local hj="$L/.serving_check_$HL.json"
  if [ -f "$H/capture_host.sh" ]; then
    bash "$H/capture_host.sh" > "$hj"
    if [ -n "$CHECK_SERVERS" ] && ! python3 "$CHECK_SERVERS" "$hj"; then
      stamp "STOP: servers differ from the registration"; exit 3
    fi
  fi
  if [ -f "$E/e3_net_block.sh" ]; then
    if ! bash "$E/e3_net_block.sh" --check 2>/dev/null; then stamp "STOP: e3net egress rule missing"; exit 6; fi
  fi
  local fr; fr=$(free_mb)
  if [ -n "$fr" ] && [ "$fr" -lt "$MIN_FREE_MB" ]; then stamp "STOP: free disk $fr MB < $MIN_FREE_MB MB"; exit 4; fi
}
guards
stamp "E3 start host=$HL manifest $M sha256 $got | free_mb=$(free_mb) | preemptions agent=$(preempt 8000) monitor=$(preempt 8001)"
if [ -f "$H/capture_host.sh" ]; then
  bash "$H/capture_host.sh" > "$L/serving_host_${HL}_$(date -u +%Y%m%dT%H%M%SZ).json" 2>/dev/null || true
fi

blocks=$(python3 -c "import json;print(' '.join(str(b['block']) for b in json.load(open('$M'))['block_order'] if b.get('host', '$HL')=='$HL'))")
for b in $blocks; do
  label=$(python3 -c "import json;print([x['label'] for x in json.load(open('$M'))['block_order'] if x['block']==$b][0])")
  bd="$L/$label"
  mkdir -p "$bd"
  if ls "$bd"/attempt-*/run.json >/dev/null 2>&1; then
    stamp "skip $label: a finished attempt exists"
    continue
  fi
  guards
  k=$(( $(find "$bd" -maxdepth 1 -type d -name 'attempt-*' | wc -l) + 1 ))
  ad="$bd/attempt-$k"
  stamp "start $label attempt $k | free_mb=$(free_mb) | preemptions agent=$(preempt 8000) monitor=$(preempt 8001)"
  sg e3net -c "$PY $E/e3_run.py --manifest $M --manifest-block $b --skip-started-under $bd --concurrency $CONC --log-dir $ad" \
    > "$ad.out" 2>&1
  stamp "done $label attempt $k: $(grep -o '\"scored\": [0-9]*' "$ad/run.json" 2>/dev/null || true) | free_mb=$(free_mb) | preemptions agent=$(preempt 8000) monitor=$(preempt 8001)"
  if [ ! -f "$ad/run.json" ]; then
    stamp "STOP: $label attempt $k ended without run.json"; exit 5
  fi
done
stamp "E3_DONE host=$HL"
