#!/usr/bin/env bash
# E7 host provisioning (pre-registration docs/agent-risk-modeling/E7-PREREG-2026-10-06.md),
# run on an H100 host with vLLM, control-arena, pilot sandbox image, and Qwen3-4B monitor weights.
#
#   bash e7_provision.sh free14b     # new hosts: remove unused 14B weights from HF cache if space is needed
#   bash e7_provision.sh weights     # tmpfs at /mnt/e7hf (40 GB), Qwen3.5-27B-FP8 @ 97f5941b, size/sha256 check
#   bash e7_provision.sh guards      # egress block + disk space check
#   bash e7_provision.sh servers     # E2's registered servers (e2_serve.sh) on tmpfs weights
#   bash e7_provision.sh status      # check host environment
set -uo pipefail
MNT=${E7_HF:-/mnt/e7hf}
REV=97f5941bf617e31c5e237364a8602ce3f03a551a
REPO=Qwen/Qwen3.5-27B-FP8
SNAP=$MNT/hub/models--Qwen--Qwen3.5-27B-FP8/snapshots/$REV
HOSTLABEL=${E7_HOST:-h$(hostname | sed 's/.*-//')}
H=${HARNESS_DIR:-$(cd "$(dirname "$0")" && pwd)}
SERVING_DIR=${SERVING_DIR:-$H/../serving}
stamp() { echo "[$(date -u +%FT%TZ)] $*"; }

case "${1:-}" in
  free14b)
    D=${HF_HOME:-$HOME/.cache/huggingface}/hub/models--Qwen--Qwen3-14B-FP8
    if [ -d "$D/snapshots" ]; then
      for f in "$D"/snapshots/*/*; do b=$(readlink -f "$f" 2>/dev/null); [ -f "$b" ] && rm -f "$b"; done
      rm -rf "$D"
    fi
    df -BM / | tail -1
    ;;
  weights)
    sudo mkdir -p "$MNT"
    mountpoint -q "$MNT" || sudo mount -t tmpfs -o size=40g,mode=0755 tmpfs "$MNT"
    sudo chown "${HOST_USER:-$USER}:${HOST_USER:-$USER}" "$MNT"
    stamp "download $REPO @ $REV"
    HF_HUB_OFFLINE=0 HF_HUB_CACHE=$MNT/hub "${PYTHON:-python3}" - <<PY
from huggingface_hub import snapshot_download
print("snapshot", snapshot_download("$REPO", revision="$REV", cache_dir="$MNT/hub"))
PY
    [ -f "$SNAP/config.json" ] || { echo "snapshot incomplete"; exit 3; }
    stamp "verify sizes and sha256 against the Hub"
    HF_HUB_OFFLINE=0 "${PYTHON:-python3}" - <<PY
import hashlib, os, sys
from concurrent.futures import ThreadPoolExecutor
from huggingface_hub import HfApi
info = HfApi().model_info("$REPO", revision="$REV", files_metadata=True)
snap = "$SNAP"
def check(s):
    p = os.path.join(snap, s.rfilename)
    if not os.path.exists(p):
        return (s.rfilename, "missing")
    if os.path.getsize(p) != s.size:
        return (s.rfilename, f"size {os.path.getsize(p)} != {s.size}")
    lfs = getattr(s, "lfs", None)
    sha = None
    if lfs is not None:
        sha = lfs.get("sha256") if isinstance(lfs, dict) else getattr(lfs, "sha256", None)
    if sha:
        h = hashlib.sha256()
        with open(p, "rb") as f:
            for chunk in iter(lambda: f.read(1 << 24), b""):
                h.update(chunk)
        if h.hexdigest() != sha:
            return (s.rfilename, "sha256 differs")
    return (s.rfilename, "ok")
with ThreadPoolExecutor(8) as ex:
    res = list(ex.map(check, info.siblings))
bad = [r for r in res if r[1] != "ok"]
print(len(res), "files;", len(res) - len(bad), "ok;", "bad:", bad)
sys.exit(1 if bad else 0)
PY
    rc=$?
    stamp "weights verify rc=$rc"
    df -BM "$MNT" | tail -1
    exit $rc
    ;;
  guards)
    if [ -f "$H/net_block.sh" ]; then
      bash "$H/net_block.sh" > /dev/null && echo "egress block installed"
    fi
    sudo iptables -C DOCKER-USER -i br-+ ! -o br-+ -p tcp -j REJECT --reject-with tcp-reset && echo "egress rule present"
    df -BM /
    ;;
  servers)
    SERVE_SCRIPT=""
    if [ -f "$SERVING_DIR/e2_serve.sh" ]; then
      SERVE_SCRIPT="$SERVING_DIR/e2_serve.sh"
    elif [ -f "$H/e2_serve.sh" ]; then
      SERVE_SCRIPT="$H/e2_serve.sh"
    fi
    if [ -n "$SERVE_SCRIPT" ]; then
      E2_HF=$MNT setsid nohup bash "$SERVE_SCRIPT" > "$HOME/logs/e7_serve.log" 2>&1 < /dev/null &
      for i in $(seq 1 200); do grep -q "monitor up\|did not come up" "$HOME/logs/e7_serve.log" && break; sleep 5; done
      tail -6 "$HOME/logs/e7_serve.log" | cut -c1-200
    fi
    ;;
  status)
    hostname; nvidia-smi --query-gpu=name,memory.total,memory.used --format=csv,noheader
    "${CONTROL_ARENA_PYTHON:-python3}" -c "import inspect_ai; print('inspect', inspect_ai.__version__)" 2>/dev/null || true
    "${VLLM_PYTHON:-python3}" -c "import vllm, torch; print('vllm', vllm.__version__, torch.__version__)" 2>/dev/null || true
    findmnt -no SOURCE,FSTYPE,OPTIONS "$MNT" 2>/dev/null || true; df -BM / "$MNT" 2>/dev/null || df -BM /
    docker info --format '{{.Driver}} {{.DockerRootDir}}' 2>/dev/null || true
    docker images --format '{{.Repository}}:{{.Tag}} {{.Size}}' 2>/dev/null | head -5 || true
    if [ -f "$H/net_block.sh" ]; then
      bash "$H/net_block.sh" --status 2>/dev/null | head -8 || true
    fi
    ;;
  *)
    echo "usage: e7_provision.sh free14b|weights|guards|servers|status"; exit 1;;
esac
