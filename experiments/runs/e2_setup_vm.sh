#!/usr/bin/env bash
# E2 host setup on an NVIDIA H100 host (vLLM, control-arena with Inspect,
# pilot sandbox image, monitor weights).
#
#   E2_DEV=/dev/sdb bash e2_setup_vm.sh volume    # format (only if blank) and mount the 50 GB volume at /mnt/e2hf
#   bash e2_setup_vm.sh download                   # Qwen3.5-27B-FP8 @ 97f5941b into /mnt/e2hf/hub, then verify
#   bash e2_setup_vm.sh readonly                   # remount the weights volume read-only
#   bash e2_setup_vm.sh guards                     # egress block + disk space check
#   bash e2_setup_vm.sh status                     # record environment status
#
# Layout: Docker store and sandbox writes stay on the root file system;
# the weights volume holds only the model and is mounted read-only once the
# download is verified, so no sandbox or log can fill it.
set -uo pipefail
MNT=${E2_HF:-/mnt/e2hf}
REV=97f5941bf617e31c5e237364a8602ce3f03a551a
REPO=Qwen/Qwen3.5-27B-FP8
SNAP=$MNT/hub/models--Qwen--Qwen3.5-27B-FP8/snapshots/$REV
H=${HARNESS_DIR:-$(cd "$(dirname "$0")" && pwd)}
stamp() { echo "[$(date -u +%FT%TZ)] $*"; }

case "${1:-}" in
  volume)
    DEV=${E2_DEV:?set E2_DEV to the attached volume device, e.g. /dev/sdb}
    size=$(lsblk -bdno SIZE "$DEV") || { echo "no device $DEV"; exit 2; }
    [ "$size" -ge 50000000000 ] && [ "$size" -le 60000000000 ] || { echo "refusing: $DEV is $size bytes, not the 50 GB volume"; exit 2; }
    if [ -n "$(sudo blkid -o value -s TYPE "$DEV" 2>/dev/null)" ]; then
      echo "$DEV already has a file system; mounting without formatting"
    elif lsblk -no MOUNTPOINT "$DEV" | grep -q .; then
      echo "refusing: $DEV is mounted"; exit 2
    else
      stamp "formatting blank $DEV as ext4 (label e2hf)"
      sudo mkfs.ext4 -q -L e2hf "$DEV" || exit 3
    fi
    sudo mkdir -p "$MNT" && sudo mount "$DEV" "$MNT" && sudo chown "${HOST_USER:-$USER}:${HOST_USER:-$USER}" "$MNT"
    df -BM "$MNT"
    ;;
  download)
    [ -w "$MNT" ] || { echo "$MNT not writable"; exit 2; }
    stamp "download $REPO @ $REV"
    HF_HUB_OFFLINE=0 HF_HUB_CACHE=$MNT/hub "${PYTHON:-python3}" - <<PY
from huggingface_hub import snapshot_download
p = snapshot_download("$REPO", revision="$REV", cache_dir="$MNT/hub")
print("snapshot", p)
PY
    [ -f "$SNAP/config.json" ] || { echo "snapshot incomplete"; exit 3; }
    du -sh --apparent-size "$MNT/hub/models--Qwen--Qwen3.5-27B-FP8/blobs"
    ls "$SNAP" | head -40
    stamp "download done"
    ;;
  readonly)
    sudo mount -o remount,ro "$MNT" && findmnt -no OPTIONS "$MNT"
    ;;
  freeze)
    chmod -R a-w "$MNT/hub" && echo "weights read-only (permissions) under $MNT/hub"
    ;;
  guards)
    if [ -f "$H/net_block.sh" ]; then
      bash "$H/net_block.sh" > /dev/null && echo "egress block installed"
    fi
    sudo iptables -C DOCKER-USER -i br-+ ! -o br-+ -p tcp -j REJECT --reject-with tcp-reset && echo "egress rule present"
    df -BM /
    ;;
  status)
    hostname; nvidia-smi --query-gpu=name,driver_version,memory.total --format=csv,noheader
    "${CONTROL_ARENA_PYTHON:-python3}" -c "import inspect_ai; print('inspect', inspect_ai.__version__)" 2>/dev/null || true
    "${VLLM_PYTHON:-python3}" -c "import vllm, torch; print('vllm', vllm.__version__, torch.__version__)" 2>/dev/null || true
    findmnt -no SOURCE,FSTYPE,OPTIONS "$MNT" 2>/dev/null || true; df -BM / "$MNT" 2>/dev/null || df -BM /
    docker info --format '{{.Driver}} {{.DockerRootDir}}' 2>/dev/null || true
    if [ -f "$H/net_block.sh" ]; then
      bash "$H/net_block.sh" --status 2>/dev/null | head -8 || true
    fi
    ;;
  *)
    echo "usage: e2_setup_vm.sh volume|download|readonly|freeze|guards|status"; exit 1;;
esac
