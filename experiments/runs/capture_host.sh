#!/usr/bin/env bash
# Record the serving host for the run metadata: GPU, driver, CUDA, the full
# vllm serve command lines, and the vLLM/CUDA environment variables of the
# running servers (only VLLM_*, CUDA_*, HF_HUB_OFFLINE, TORCH_* keys; nothing else
# from the process environment is read out). Writes JSON to stdout.
#   bash capture_host.sh > ~/logs/pilot-ab/serving_host.json
python3 - <<'PY'
import json, os, subprocess, datetime
def sh(c):
    return subprocess.run(c, shell=True, capture_output=True, text=True).stdout.strip()
out = {
    "captured_utc": datetime.datetime.utcnow().isoformat() + "Z",
    "hostname": sh("hostname"),
    "gpu": sh("nvidia-smi --query-gpu=name,driver_version,memory.total,pci.bus_id --format=csv,noheader"),
    "cuda_driver_max": sh("nvidia-smi | grep -o 'CUDA Version: [0-9.]*'"),
    "kernel": sh("uname -r"),
    "vllm_version": sh("~/venv-vllm/bin/python -c 'import vllm, torch; print(vllm.__version__, torch.__version__, torch.version.cuda)'"),
    "servers": [],
}
keep = ("VLLM_", "CUDA_", "TORCH_", "HF_HUB_OFFLINE", "NCCL_")
for pid in sh("pgrep -f 'bin/vllm serve'").split():
    try:
        cmd = open(f"/proc/{pid}/cmdline", "rb").read().split(b"\0")
        env = open(f"/proc/{pid}/environ", "rb").read().split(b"\0")
    except Exception:
        continue
    envd = {}
    for kv in env:
        k, _, v = kv.decode(errors="replace").partition("=")
        if k.startswith(keep):
            envd[k] = v
    out["servers"].append({"pid": int(pid), "cmdline": " ".join(x.decode() for x in cmd if x), "env": envd})
print(json.dumps(out, indent=1))
PY
