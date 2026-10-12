#!/usr/bin/env bash
# E3 host setup (an NVIDIA H100 host with vLLM 0.30.0, Qwen3-14B-FP8 and
# Qwen3-4B-Instruct-2507 weights present in the Hugging Face cache; no extra volume is needed).
#
#   bash e3_setup_vm.sh venv      # venv (Python >= 3.12) with tau2 v1.0.1 from tau2-bench
#   bash e3_setup_vm.sh weights   # verify both snapshots are complete in the cache (offline)
#   bash e3_setup_vm.sh guards    # egress rule for group e3net, live egress test
#   bash e3_setup_vm.sh status    # check environment status
#
# tau2-bench tag v1.0.1 (commit fc0055dc); the venv is built while the host still has egress (pip),
# before any episode; episodes then run under group e3net, whose non-loopback packets the host rejects.
set -uo pipefail
TAU2_COMMIT=fc0055dc4e0a316c3f83133267fbd6faaa770992
H=${HARNESS_DIR:-$(cd "$(dirname "$0")/.." && pwd)}
E=${E3_DIR:-$(cd "$(dirname "$0")" && pwd)}
TAU2_DIR=${TAU2_DIR:-$HOME/tau2-bench}
VENV_TAU2=${VENV_TAU2:-$HOME/venv-tau2}
stamp() { echo "[$(date -u +%FT%TZ)] $*"; }

case "${1:-}" in
  venv)
    [ -d "$TAU2_DIR/src/tau2" ] || { echo "missing $TAU2_DIR"; exit 2; }
    PYB=""
    for c in python3.13 python3.12 "${PYTHON:-python3}"; do
      if command -v "$c" >/dev/null 2>&1 && "$c" -c 'import sys; sys.exit(0 if sys.version_info[:2] >= (3, 12) else 1)'; then PYB=$c; break; fi
    done
    if [ -z "$PYB" ]; then
      stamp "no Python >= 3.12 found; please install Python 3.12+"
      exit 2
    fi
    stamp "base interpreter $PYB ($("$PYB" --version 2>&1))"
    "$PYB" -m venv "$VENV_TAU2"
    "$VENV_TAU2/bin/pip" install -q --upgrade pip
    "$VENV_TAU2/bin/pip" install -q -e "$TAU2_DIR" websockets requests
    LITELLM_LOCAL_MODEL_COST_MAP=True "$VENV_TAU2/bin/python" -c "import tau2; from tau2.domains.airline.environment import get_tasks; print('tau2 ok', len(get_tasks('base')))"
    ;;
  weights)
    HF_CACHE="${HF_HOME:-$HOME/.cache/huggingface}/hub"
    for d in models--Qwen--Qwen3-14B-FP8/snapshots/9a283b4a5efbc09ce247e0ae5b02b744739e525a \
             models--Qwen--Qwen3-4B-Instruct-2507/snapshots/cdbee75f17c01a7cc42f958dc650907174af0554; do
      p="$HF_CACHE/$d"
      [ -f "$p/config.json" ] && echo "ok $d ($(du -shL "$p" | cut -f1))" || { echo "MISSING $p"; exit 3; }
    done
    ;;
  guards)
    bash "$E/e3_net_block.sh"
    bash "$E/e3_net_block.sh" --check || { echo "egress rule missing"; exit 6; }
    if sg e3net -c "curl -s -m 5 -o /dev/null https://pypi.org" 2>/dev/null; then echo "EGRESS NOT BLOCKED for e3net"; exit 6; fi
    echo "e3net: external connection refused (expected)"
    df -BM "$HOME/logs" 2>/dev/null || df -BM /
    ;;
  status)
    echo "host $(hostname)"; nvidia-smi --query-gpu=name,memory.total,memory.used --format=csv,noheader
    df -BM --output=avail,target / "$HOME" | sed -n 1,3p
    bash "$E/e3_net_block.sh" --status 2>/dev/null || true
    git -C "$TAU2_DIR" rev-parse HEAD 2>/dev/null || echo "tau2-bench: not a git checkout"
    echo "tau2 commit registered: $TAU2_COMMIT"
    "$VENV_TAU2/bin/python" --version 2>/dev/null || true
    ;;
  *) echo "usage: $0 venv|weights|guards|status"; exit 1;;
esac
