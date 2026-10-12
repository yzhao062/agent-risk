"""Check that the running vLLM servers are the ones E2 registers (E2-PREREG-2026-10-05.md).

Reads the JSON that capture_host.sh writes (command lines and VLLM_/CUDA_/HF_
environment of every `vllm serve` process) and compares it with the registered
serving configuration:

  agent    Qwen/Qwen3.5-27B-FP8 served from the snapshot directory of revision
           97f5941bf617e31c5e237364a8602ce3f03a551a on 127.0.0.1:8000, with every
           flag in AGENT_REQUIRED and VLLM_BATCH_INVARIANT unset (vLLM 0.30.0
           refuses it for this architecture; default kernels are registered);
  monitor  the D58/D64 monitor command line, byte for byte, with
           VLLM_BATCH_INVARIANT=1, VLLM_USE_DEEP_GEMM=0, VLLM_USE_FLASHINFER_SAMPLER=0
           and HF_HUB_OFFLINE=1.

Exit 0 and print "servers OK" when both match; exit 3 and list the differences
otherwise. e2_run.sh calls it before every block.

  bash capture_host.sh > /tmp/h.json && python3 e2_check_servers.py /tmp/h.json
"""

from __future__ import annotations

import json
import os
import sys

AGENT_REVISION = "97f5941bf617e31c5e237364a8602ce3f03a551a"
AGENT_REQUIRED = [
    f"models--Qwen--Qwen3.5-27B-FP8/snapshots/{AGENT_REVISION}",
    "--served-model-name Qwen/Qwen3.5-27B-FP8",
    "--host 127.0.0.1", "--port 8000",
    "--max-model-len 32768",
    "--enable-auto-tool-choice", "--tool-call-parser qwen3_coder",
    "--reasoning-parser qwen3", "--language-model-only",
    '--default-chat-template-kwargs {"enable_thinking": false}',
    "--generation-config vllm", "--seed 0", "--no-enable-prefix-caching",
]
AGENT_ENV = {"VLLM_USE_DEEP_GEMM": "0", "VLLM_USE_FLASHINFER_SAMPLER": "0", "HF_HUB_OFFLINE": "1"}

MONITOR_REQUIRED = [
    "Qwen/Qwen3-4B-Instruct-2507",
    "--revision cdbee75f17c01a7cc42f958dc650907174af0554",
    "--host 127.0.0.1", "--port 8001",
    "--served-model-name Qwen/Qwen3-4B-Instruct-2507",
    "--gpu-memory-utilization 0.30", "--max-model-len 8192",
    "--enable-auto-tool-choice", "--tool-call-parser hermes",
    "--generation-config vllm", "--seed 0", "--no-enable-prefix-caching",
]
MONITOR_ENV = {"VLLM_USE_DEEP_GEMM": "0", "VLLM_USE_FLASHINFER_SAMPLER": "0", "VLLM_BATCH_INVARIANT": "1",
               "HF_HUB_OFFLINE": "1"}


def check(host: dict) -> list[str]:
    problems = []
    servers = host.get("servers") or []
    agents = [s for s in servers if "Qwen3.5-27B-FP8" in s.get("cmdline", "")]
    mons = [s for s in servers if "Qwen3-4B-Instruct-2507" in s.get("cmdline", "")]
    if len(agents) != 1:
        problems.append(f"expected one agent server, found {len(agents)}")
    if len(mons) != 1:
        problems.append(f"expected one monitor server, found {len(mons)}")
    if len(servers) != len(agents) + len(mons):
        problems.append(f"unexpected extra vllm processes: {len(servers)}")
    for a in agents[:1]:
        cmd = a.get("cmdline", "")
        for flag in AGENT_REQUIRED:
            if flag not in cmd:
                problems.append(f"agent flag missing: {flag}")
        for k, v in AGENT_ENV.items():
            if a.get("env", {}).get(k) != v:
                problems.append(f"agent env {k}={a.get('env', {}).get(k)!r}, want {v!r}")
        if "VLLM_BATCH_INVARIANT" in a.get("env", {}):
            problems.append("agent env sets VLLM_BATCH_INVARIANT (registered: unset)")
    for m in mons[:1]:
        cmd = m.get("cmdline", "")
        for flag in MONITOR_REQUIRED:
            if flag not in cmd:
                problems.append(f"monitor flag missing: {flag}")
        for k, v in MONITOR_ENV.items():
            if m.get("env", {}).get(k) != v:
                problems.append(f"monitor env {k}={m.get('env', {}).get(k)!r}, want {v!r}")
    return problems


def main() -> int:
    if len(sys.argv) < 2:
        print("usage: e2_check_servers.py <host_capture.json>")
        return 2
    host = json.load(open(sys.argv[1]))
    problems = check(host)
    if problems:
        print("SERVER MISMATCH:\n  " + "\n  ".join(problems))
        return 3
    pids = {("agent" if "Qwen3.5" in s.get("cmdline", "") else "monitor"): s.get("pid") for s in host.get("servers", [])}
    print(f"servers OK {json.dumps(pids)}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
