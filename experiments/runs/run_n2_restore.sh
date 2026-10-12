#!/usr/bin/env bash
# N2 restoration check: plumbing smoke on one B0 episode, then the frozen
# check on the first 10 later-blocked B1 episodes (two restorations each).
set -uo pipefail
H=${HARNESS_DIR:-$(cd "$(dirname "$0")" && pwd)}
L=${LOGROOT:-$HOME/logs/n2/restore}
B0_DIR=${B0_DIR:-$HOME/logs/pilot-ab/b0}
B1_DIR=${B1_DIR:-$HOME/logs/arms-cd/b1}
CONTROL_ARENA_DIR=${CONTROL_ARENA_DIR:-$HOME/control-arena}
mkdir -p "$L"
cd "$CONTROL_ARENA_DIR" || exit 1
export VLLM_API_KEY=EMPTY PYTHONPATH="${PYTHONPATH:-}:$H:$H/.."
PY=${PYTHON:-.venv/bin/python}
stamp() { echo "[$(date -u +%FT%TZ)] $*"; }

if [ -d "$B0_DIR" ]; then
  stamp "smoke (B0, 1 episode x 2)"
  $PY "$H/run_n2_restore.py" prepare --source-log-dir "$B0_DIR" --out-dir "$L/smoke-b0" --n 1
  $PY "$H/run_n2_restore.py" run --units "$L/smoke-b0/restore_units.json" --log-dir "$L/smoke-b0" > "$L/smoke-b0.run.out" 2>&1
  $PY "$H/run_n2_restore.py" check --units "$L/smoke-b0/restore_units.json" --log-dir "$L/smoke-b0"
fi

if [ -d "$B1_DIR" ]; then
  stamp "B1 restoration check (10 x 2)"
  $PY "$H/run_n2_restore.py" prepare --source-log-dir "$B1_DIR" --out-dir "$L/b1"
  $PY "$H/run_n2_restore.py" run --units "$L/b1/restore_units.json" --log-dir "$L/b1" > "$L/b1.run.out" 2>&1
  $PY "$H/run_n2_restore.py" check --units "$L/b1/restore_units.json" --log-dir "$L/b1"
fi
stamp "RESTORE_DONE"
