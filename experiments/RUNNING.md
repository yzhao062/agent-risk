# Running Experiment and Demonstration Runs

This guide documents the hardware, environment variables, serving configurations, and execution commands for reproducing the demonstration and experimental runs reported in *"agent-risk: Risk Assessment for Changes to AI Agents"* (arXiv).

Commands follow the run registrations and the run codes defined in Section 4 and Appendix D of the paper, as catalogued in `ARTIFACT.md`.

---

## 1. Hardware and Environment Requirements

Per Appendix D.2 of the paper:

### Prerequisites
The experiment execution harness requires Python ≥3.11 and a local checkout of ControlArena (specified via `CONTROL_ARENA_DIR`), whereas offline log analysis and verification checks require Python 3.9+ with NumPy and SciPy (plus Matplotlib for A3 and A38, and Python 3.14 or later for A23; see `ARTIFACT.md`).

### Standard Demonstration Runs (R0–R6, R8–R13)
* **Hosts:** Dedicated cloud/bare-metal compute nodes with **1× NVIDIA H100 80GB GPU**.
* **Disk Space:** High-speed NVMe storage. Free disk space safety thresholds (8,000 to 21,000 MB, configurable via `MIN_FREE_MB`) are checked before each block to prevent container disk exhaustion.
* **Concurrency:** 32 episodes executed concurrently per host.
* **Serving Stack:** vLLM 0.30.0 with PyTorch 2.13.0 and CUDA 13.0. Offline weights loaded from local cache.
* **Network Isolation:** Sandboxes run in fresh Docker containers with unroutable DNS (`192.0.2.1`) and host-level iptables rules blocking non-loopback egress (`net_block.sh` or `e3_net_block.sh`).

### Model-Update Reruns (R7 / Arm E)
* **Hosts:** High-performance computing cluster nodes (DeltaAI) equipped with **NVIDIA GH200 Grace Hopper Superchip (120GB unified memory)**.
* **Serving Stack:** Apptainer / Singularity container running vLLM 0.30.0 (`aarch64`).
* **Memory Utilization:** 85% GPU memory allocation (`--gpu-memory-utilization 0.85`).
* **Concurrency:** 32 requests in flight under batch-invariant kernels; sequential (batch 1) under default kernels for deterministic state replay.

---

## 2. Serving Configuration

Model servers run on localhost only (`127.0.0.1`) and must never be exposed externally.

| Configuration | Agent Model | Monitor Model | Serving Script | Runs Used |
|---|---|---|---|---|
| **Base H100 Stack** | `Qwen/Qwen3-14B-FP8`<br>(rev `9a283b4`, port 8000, max-len 32,768, util 0.55) | `Qwen/Qwen3-4B-Instruct-2507`<br>(rev `cdbee75`, port 8001, max-len 8,192, util 0.30) | `experiments/serving/serve.sh` | R0, R1, R2, R3 (D58), R4 (D64), R5 (D65), R6 (N2), R8 (D66), R9 (E1) |
| **Stronger Agent Stack** | `Qwen/Qwen3.5-27B-FP8`<br>(rev `97f5941`, port 8000, max-len 32,768, util 0.60) | `Qwen/Qwen3-4B-Instruct-2507`<br>(rev `cdbee75`, port 8001, max-len 8,192, util 0.30) | `experiments/serving/e2_serve.sh` | R10 (E2), R13 (E7) |
| **Airline Stack** | `Qwen/Qwen3-14B-FP8`<br>(rev `9a283b4`, port 8000, max-len 32,768, util 0.55) | `Qwen/Qwen3-4B-Instruct-2507`<br>(rev `cdbee75`, port 8001, max-len 32,768, util 0.30) | `experiments/serving/e3_serve.sh` | R11 (E3), R12 (E5) |
| **DeltaAI GH200 Stack** | Rung-dependent (FP8, BF16, Qwen2.5, Qwen3.5) | N/A (replaying logged agent requests) | `experiments/runs/deltaai/serve_lib.sh` | R7 (Arm E) |

### Important vLLM Flags
* `--no-enable-prefix-caching`: Enforced on all servers across runs. Prefix caching causes nondeterministic outputs under concurrency depending on cache state.
* `VLLM_USE_FLASHINFER_SAMPLER=0`: Avoids runtime nvcc JIT compilation on compute images.
* `VLLM_USE_DEEP_GEMM=0`: Uses precompiled CUTLASS block-FP8 GEMM kernels.
* `VLLM_BATCH_INVARIANT=1`: Required for batch-invariant runs (R0, R2, R3, R4, R5, R6, R8, R9, R13 monitor) to ensure seed-addressable replay across varying batch sizes. Unset for default kernel runs (R1, R10 agent, R11 agent, R12 agent).

---

## 3. Environment Variables Reference

All site-specific paths, credentials, and infrastructure parameters are configurable via environment variables.

> [!NOTE]
> Run scripts execute `cd "$CONTROL_ARENA_DIR"` before validating manifest checksums or writing output blocks. Therefore, environment variables passing file or directory paths (`MANIFEST`, `LOGROOT`, and `UNITS`) must always be specified as **absolute paths** (e.g., using `"$PWD/..."` when executing from the repository root).

| Variable | Description | Default / Example |
|---|---|---|
| `HARNESS_DIR` | Path to the experiment harness root | Absolute path to `experiments/runs` |
| `ANALYSIS_DIR` | Path to the analysis scripts directory | `../analysis` relative to harness |
| `SERVING_DIR` | Path to serving scripts | `../serving` relative to harness |
| `CONTROL_ARENA_DIR` | Path to ControlArena checkout | `$HOME/control-arena` |
| `PYTHON` | Python interpreter for ControlArena environment | `.venv/bin/python` |
| `TAU2_PYTHON` | Python interpreter for tau2-bench (airline) environment | `$HOME/venv-tau2/bin/python` |
| `VLLM_BIN` | Path to vLLM executable | `$HOME/venv-vllm/bin/vllm` |
| `LOGROOT` | Output directory for run episode logs (use absolute path) | `$HOME/logs/<run>` (code default; documented commands pass `$PWD/new-logs/<run>`, so a new run stays apart from the released logs in `data/logs/`; to analyze it, pass its folder as `--logroot` or set `AGENT_RISK_LOGS=$PWD/new-logs`) |
| `MANIFEST` | Path to the frozen JSON manifest (use absolute path) | `$H/../../data/manifests/<run>/manifest_<run>.json` |
| `MANIFEST_SHA256` | Registered SHA-256 checksum of the manifest | Hex digest specified per run |
| `CONCURRENCY` | Maximum concurrent episodes / requests in flight | `32` |
| `MIN_FREE_MB` | Disk safety threshold (stops before a block if free disk space is below) | `8000` (E1, relay), `16000` (E7), or `21000` (E2) |
| `VLLM_BATCH_INVARIANT` | Enable vLLM batch-invariance kernel mode | `1` or unset |
| `AGENT_UTIL` | GPU memory utilization fraction for the agent server | `0.56` (code default; `0.55` on H100 for 14B, `0.60` for 27B) |
| `MON_UTIL` | GPU memory utilization fraction for the monitor server | `0.36` (code default; `0.30` on H100) |
| `DELTA_BASE` | Base work directory on cluster (GH200) | Site-specific path (e.g., `/path/to/work/agent-risk-e`) |
| `VLLM_SIF` | Path to vLLM Singularity/Apptainer image on cluster | `$DELTA_BASE/containers/vllm-...sif` |
| `E2_HF` | Hugging Face cache directory mount for E2/E7 serving (`e2_serve.sh`) | `/mnt/e2hf` |
| `E7_HF` | Hugging Face cache directory mount for E7 host checks (`run_e7.sh`) | `/mnt/e7hf` |
| `HALF` | Host half partition identifier (`a` or `b`) for split-host runs (`run_relay.sh`, `run_e1.sh`) | `a` or `b` |
| `HOST` | Host index (`1` to `6`) for E7 multi-host execution (`run_e7.sh`) | `1` through `6` |
| `HOST_LABEL` | Host identifier label for single/multi-host output files (`e2_run.sh`, `e3_run.sh`, `e5_run.sh`) | `h1` |
| `WITH_B1` | Execute B1 held-out baseline alongside Arms C and D (`run_arms_cd.sh`) | `0` (set `1` to run B1) |
| `UNITS` | Path to units JSON manifest for E5 targeted execution (`e5_run.sh`) | `$PWD/data/manifests/e5/e5_units.json` |

---

## 4. Run Execution Guide (R0 to R13)

Run every command from the repository root. The runners add their own folder and its parent to `PYTHONPATH`; in this repository the harness package `agent_risk_harness` sits in `experiments/harness/`, so export that folder first:

```bash
export PYTHONPATH="$PWD/experiments/harness${PYTHONPATH:+:$PYTHONPATH}"
```

### R0: Start Baseline vLLM Servers (H100)
Used for all ControlArena runs (R1 through R9).
```bash
# For batch-invariant runs (R2, R3, R4, R5, R6, R8, R9):
VLLM_BATCH_INVARIANT=1 AGENT_UTIL=0.55 MON_UTIL=0.30 \
  AGENT_EXTRA=--no-enable-prefix-caching MON_EXTRA=--no-enable-prefix-caching \
  bash experiments/serving/serve.sh

# For R1 (exploratory pilot), omit VLLM_BATCH_INVARIANT:
AGENT_UTIL=0.55 MON_UTIL=0.30 \
  AGENT_EXTRA=--no-enable-prefix-caching MON_EXTRA=--no-enable-prefix-caching \
  bash experiments/serving/serve.sh
```

### R1: The Exploratory Log and Live Gates (B0, Arm A)
Runs B0 (300 baseline episodes), extracts theta grid, runs validation live gates (val2_t1, val2_t2, val2_t3), and concurrency-1 diagnostics.
```bash
mkdir -p "$PWD/new-logs/pilot-ab"
LOGROOT="$PWD/new-logs/pilot-ab" bash experiments/runs/run_stage2.sh
```

### R2: Held-Out Log and Live Stopping/Retry Gates (B1, Arm C, Arm D)
Sequential chain on batch-invariant servers: Arm C (retry gate, $\theta=0.94$, cap 2), Arm D (announced stopping gate, $\theta=0.94$), and B1 (held-out baseline log).
```bash
WITH_B1=1 LOGROOT="$PWD/new-logs/arms-cd" bash experiments/runs/run_arms_cd.sh
```

### R3: Registered Replication (D58: B2 and Val3)
20 alternating blocks of 100 episodes from the frozen replication manifest. Resumable under the begun-episode rule.
```bash
MANIFEST="$PWD/data/manifests/d58/manifest_d58.json" \
LOGROOT="$PWD/new-logs/d58" \
MANIFEST_SHA256=59fab8e1e1059fcfc39cbeff631ee993c326b978b5c080a76e1a076f19d564b2 \
bash experiments/runs/run_replication.sh
```

### R4: Paired Announcement Test (D64)
12 alternating blocks of 100 episodes evaluating announced vs. unannounced stopping gates.
```bash
MANIFEST="$PWD/data/manifests/d64/manifest_d64.json" \
LOGROOT="$PWD/new-logs/d64" \
MANIFEST_SHA256=c6911fb3d495e9ca07a2ac694c1c8dd5ff5f1c01fcd5c90d797b8b6f3e23c413 \
bash experiments/runs/run_replication.sh
```

### R5: Long-Horizon Log (D65)
Long-horizon evaluation up to 40 steps per episode across two hosts.
```bash
# First host (VM 2):
MANIFEST="$PWD/data/manifests/d65/manifest_d65_vm2.json" \
LOGROOT="$PWD/new-logs/d65" \
MANIFEST_SHA256=ebb3dd57748c0847d787fb5477b490484266e09266a091ddbd72863b76008593 \
bash experiments/runs/run_replication.sh

# Second host (VM 3):
MANIFEST="$PWD/data/manifests/d65/manifest_d65_vm3.json" \
LOGROOT="$PWD/new-logs/d65" \
MANIFEST_SHA256=b3a16ce48129e3d035c0f51ce5a278ba7d4a032ccf4d983655f70412912f2056 \
bash experiments/runs/run_replication.sh
```

### R6: Step-0 Continuations (N2 Stage 1)
Targeted continuation of the retry gate ($\theta=0.94$, cap 2) for B1 episodes first blocked at step 0.
```bash
N2_LOGS="$PWD/new-logs/n2" SRC="$PWD/data/logs/arms-cd/b1" SMOKE_SRC="$PWD/data/logs/pilot-ab/b0" \
  REF="$PWD/data/logs/arms-cd/serving_host_bi.json" BOUNDS="$PWD/data/logs/arms-cd/retry_bounds_b1.json" \
  bash experiments/runs/run_n2.sh

# Optional: N2 restoration check on the held-out log (the B0 smoke step needs the B0 Inspect archive,
# which is not released; the script skips it when B0_DIR does not exist)
LOGROOT="$PWD/new-logs/n2/restore" B0_DIR="$PWD/new-logs/no-b0" B1_DIR="$PWD/data/logs/arms-cd/b1" \
  bash experiments/runs/run_n2_restore.sh
```

### R7: Model-Update Reruns (Arm E)
The jobs read per-step request files (`steps_bi_baseline.jsonl`, `steps_c1_baseline.jsonl`) that `arm_e_prepare.py steps` builds from the Inspect logs of the two logged runs; those Inspect logs are not in the release, so R7 cannot be rerun from the released data alone. A6 analyzes the released rerun results in `data/arm-e/results/`. The commands below record how the reruns were launched.
Replays logged requests from `bi_baseline` (99 episodes) and `c1_baseline` (30 episodes), located in `arm-e/data/`, across updated model checkpoints on NVIDIA GH200 (Slurm cluster).
```bash
cd experiments/runs/deltaai/arm-e

# Rung 1: Same model reference (Qwen3-14B-FP8)
sbatch --account=<ACCOUNT> arm_e_r1_same.sbatch

# Rung 2: Precision update (Qwen3-14B BF16)
sbatch --account=<ACCOUNT> \
  --export=ALL,RUNG=r2_bf16,RUNG_MODEL=Qwen/Qwen3-14B,RUNG_REV=40c069824f4251a91eefaf281ebe4c544efd3e18 \
  arm_e_hermes.sbatch

# Rung 4: Rollback control (Qwen2.5-14B-Instruct)
sbatch --account=<ACCOUNT> \
  --export=ALL,RUNG=r4_qwen25,RUNG_MODEL=Qwen/Qwen2.5-14B-Instruct,RUNG_REV=cf98f3b3bbb457ad9e2bb7baf9a0125b6b88caa8 \
  arm_e_hermes.sbatch

# Rung 3: Real release update (Qwen3.5-27B-FP8, 6-GPU array)
sbatch --account=<ACCOUNT> arm_e_r3_qwen35.sbatch
```

### R8: Two-Agent Relay (D66)
Two-shift relay using live Docker container snapshot handoff between agents across two host halves.
```bash
# First host (half a):
HALF=a \
MANIFEST="$PWD/data/manifests/d66/manifest_d66.json" \
LOGROOT="$PWD/new-logs/d66" \
MANIFEST_SHA256=90d3573a8462fe276f6210e52dee97fce702f8d54be742cf086d63f32708e39c \
bash experiments/runs/run_relay.sh

# Second host (half b):
HALF=b \
MANIFEST="$PWD/data/manifests/d66/manifest_d66.json" \
LOGROOT="$PWD/new-logs/d66" \
MANIFEST_SHA256=90d3573a8462fe276f6210e52dee97fce702f8d54be742cf086d63f32708e39c \
bash experiments/runs/run_relay.sh
```

### R9: Targeted-Execution Run (E1)
Targeted continuation of retry gate with snapshot branching across two host halves.
```bash
# First host (half a):
HALF=a \
MANIFEST="$PWD/data/manifests/e1/manifest_e1.json" \
LOGROOT="$PWD/new-logs/e1" \
MANIFEST_SHA256=6fb2a1a654cefa9a971190bcfcc35a3773a413aa0148b925372300a6f8e935fd \
bash experiments/runs/run_e1.sh

# Second host (half b):
HALF=b \
MANIFEST="$PWD/data/manifests/e1/manifest_e1.json" \
LOGROOT="$PWD/new-logs/e1" \
MANIFEST_SHA256=6fb2a1a654cefa9a971190bcfcc35a3773a413aa0148b925372300a6f8e935fd \
bash experiments/runs/run_e1.sh
```

### R10: Stronger-Agent Run (E2)
Evaluates Qwen3.5-27B-FP8 agent with default kernels and Qwen3-4B-Instruct monitor under batch-invariance.
```bash
# 1. Start servers:
bash experiments/serving/e2_serve.sh

# 2. Run chain:
MANIFEST="$PWD/data/manifests/e2/manifest_e2.json" \
LOGROOT="$PWD/new-logs/e2" \
HOST_LABEL=h1 \
MANIFEST_SHA256=82b1a67867c5059be578f6415bec8fdab92e080f3dfa759c8afde7f6418e0b08 \
bash experiments/runs/e2_run.sh
```

### R11: Airline Run (E3)
Evaluates $\tau^2$-bench airline domain in Python environment with process-group egress isolation (`e3net`).
```bash
# 1. Start servers:
bash experiments/serving/e3_serve.sh

# 2. Run chain:
MANIFEST="$PWD/data/manifests/e3/manifest_e3.json" \
LOGROOT="$PWD/new-logs/e3" \
HOST_LABEL=h1 \
MANIFEST_SHA256=aedf40cb1583093611bff2b4be04977a5345efa287cf5c619f51925b6e09fe07 \
CONCURRENCY=32 \
bash experiments/runs/e3/e3_run.sh
```

### R12: Airline Retry Gate by Targeted Execution (E5)
Targeted execution continuing airline episodes from restored in-memory state at the first block.
```bash
# 1. Ensure servers are running (e3_serve.sh).

# 2. Run chain:
MANIFEST="$PWD/data/manifests/e5/manifest_e5.json" \
UNITS="$PWD/data/manifests/e5/e5_units.json" \
LOGROOT="$PWD/new-logs/e5" \
HOST_LABEL=h1 \
MANIFEST_SHA256=71b5008f311e7eafd6cf16e7692a05e2b16e73ff53f694383a705dd658496377 \
CONCURRENCY=32 \
bash experiments/runs/e5/e5_run.sh
```

### R13: Two-Gate Run (E7)
Evaluates two retry gates by targeted execution using intermediate snapshot branching across 6 hosts.
```bash
# 1. Start servers on each host (K = 1 to 6):
E2_HF=/mnt/e7hf bash experiments/serving/e2_serve.sh

# 2. Run chain on each host:
HOST=<K> \
MANIFEST="$PWD/data/manifests/e7/manifest_e7.json" \
LOGROOT="$PWD/new-logs/e7" \
MANIFEST_SHA256=290f45e73cc9fc50bb995806c211fe667287ebdd657dbb76a91da39a45f61e6c \
CONCURRENCY=32 \
bash experiments/runs/run_e7.sh
```

---

## 5. Verification Checklist

All shell and batch scripts have been verified for syntax validity:
```bash
find experiments/runs experiments/serving -name "*.sh" -o -name "*.sbatch" | xargs -n1 bash -n
```
All Python scripts have been verified for compilation:
```bash
find experiments/runs experiments/serving -name "*.py" | xargs -n1 python -m py_compile
```
