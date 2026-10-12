"""Check that the running vLLM servers are the ones E3 registers (E3-PREREG-2026-10-05.md).

Reads the JSON that ``capture_host.sh`` writes and compares it with e3_serve.sh:

  agent    the D58 agent (Qwen/Qwen3-14B-FP8 @ 9a283b4a) on 127.0.0.1:8000 with the
           exact E3 command line and VLLM_BATCH_INVARIANT unset (default kernels);
  monitor  Qwen/Qwen3-4B-Instruct-2507 @ cdbee75f on 127.0.0.1:8001 with the exact
           E3 command line and VLLM_BATCH_INVARIANT=1.
Both need VLLM_USE_DEEP_GEMM=0, VLLM_USE_FLASHINFER_SAMPLER=0, HF_HUB_OFFLINE=1.
Exit 0 with "servers OK" or exit 3 with the differences. e3_run.sh calls it before
every block.
"""

from __future__ import annotations

import json
import sys

AGENT_REQUIRED = [
    "Qwen/Qwen3-14B-FP8",
    "--revision 9a283b4a5efbc09ce247e0ae5b02b744739e525a",
    "--host 127.0.0.1", "--port 8000",
    "--served-model-name Qwen/Qwen3-14B-FP8",
    "--gpu-memory-utilization 0.55", "--max-model-len 32768",
    "--enable-auto-tool-choice", "--tool-call-parser hermes",
    '--default-chat-template-kwargs {"enable_thinking": false}',
    "--generation-config vllm", "--seed 0", "--no-enable-prefix-caching",
]
MONITOR_REQUIRED = [
    "Qwen/Qwen3-4B-Instruct-2507",
    "--revision cdbee75f17c01a7cc42f958dc650907174af0554",
    "--host 127.0.0.1", "--port 8001",
    "--served-model-name Qwen/Qwen3-4B-Instruct-2507",
    "--gpu-memory-utilization 0.30", "--max-model-len 32768",
    "--enable-auto-tool-choice", "--tool-call-parser hermes",
    "--generation-config vllm", "--seed 0", "--no-enable-prefix-caching",
]
COMMON_ENV = {"VLLM_USE_DEEP_GEMM": "0", "VLLM_USE_FLASHINFER_SAMPLER": "0", "HF_HUB_OFFLINE": "1"}


def check(host: dict) -> list[str]:
    problems = []
    servers = host.get("servers") or []
    agents = [s for s in servers if "Qwen3-14B-FP8" in s.get("cmdline", "")]
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
        for k, v in COMMON_ENV.items():
            if a.get("env", {}).get(k) != v:
                problems.append(f"agent env {k}={a.get('env', {}).get(k)!r}, want {v!r}")
        if "VLLM_BATCH_INVARIANT" in a.get("env", {}):
            problems.append("agent env sets VLLM_BATCH_INVARIANT (registered: unset, default kernels)")
    for m in mons[:1]:
        cmd = m.get("cmdline", "")
        for flag in MONITOR_REQUIRED:
            if flag not in cmd:
                problems.append(f"monitor flag missing: {flag}")
        for k, v in {**COMMON_ENV, "VLLM_BATCH_INVARIANT": "1"}.items():
            if m.get("env", {}).get(k) != v:
                problems.append(f"monitor env {k}={m.get('env', {}).get(k)!r}, want {v!r}")
    return problems


def main() -> int:
    if len(sys.argv) < 2:
        print("usage: e3_check_servers.py <host_capture.json>")
        return 2
    host = json.load(open(sys.argv[1]))
    problems = check(host)
    if problems:
        print("SERVER MISMATCH:\n  " + "\n  ".join(problems))
        return 3
    pids = {("agent" if "Qwen3-14B" in s.get("cmdline", "") else "monitor"): s.get("pid") for s in host.get("servers", [])}
    print(f"servers OK {json.dumps(pids)}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
