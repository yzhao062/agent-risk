#!/usr/bin/env bash
# Start the two vLLM servers on localhost only (do NOT expose these ports).
#   agent (untrusted) : Qwen3-14B-FP8          on 127.0.0.1:8000
#   monitor (trusted) : Qwen3-4B-Instruct-2507 on 127.0.0.1:8001
# The two share one 40 GB A100: agent 0.56 util, monitor 0.36 util.
# On NVIDIA H100 80GB hosts (Appendix D.2): AGENT_UTIL=0.55 MON_UTIL=0.30.
set -uo pipefail
V="${VLLM_BIN:-$HOME/venv-vllm/bin/vllm}"
AGENT_MODEL=Qwen/Qwen3-14B-FP8
AGENT_REV=9a283b4a5efbc09ce247e0ae5b02b744739e525a
MON_MODEL=Qwen/Qwen3-4B-Instruct-2507
MON_REV=cdbee75f17c01a7cc42f958dc650907174af0554
AGENT_UTIL=${AGENT_UTIL:-0.56}
MON_UTIL=${MON_UTIL:-0.36}
AGENT_EXTRA=${AGENT_EXTRA:-}
MON_EXTRA=${MON_EXTRA:-}
LOGDIR="${LOGDIR:-$HOME/logs}"
mkdir -p "$LOGDIR"
export HF_HUB_OFFLINE=1
# FlashInfer top-k/top-p sampler JIT-compiles with nvcc; fall back to PyTorch sampler
export VLLM_USE_FLASHINFER_SAMPLER=0
# Precompiled CUTLASS block-FP8 GEMMs, no DeepGEMM JIT
export VLLM_USE_DEEP_GEMM=0

nohup "$V" serve "$AGENT_MODEL" --revision "$AGENT_REV" \
  --host 127.0.0.1 --port 8000 --served-model-name "$AGENT_MODEL" \
  --gpu-memory-utilization "$AGENT_UTIL" --max-model-len 32768 \
  --enable-auto-tool-choice --tool-call-parser hermes \
  --default-chat-template-kwargs '{"enable_thinking": false}' \
  --generation-config vllm --seed 0 $AGENT_EXTRA \
  > "$LOGDIR/vllm-agent.log" 2>&1 < /dev/null &
echo "agent pid $!"
for i in $(seq 1 120); do
  curl -sf http://127.0.0.1:8000/v1/models >/dev/null && { echo "agent up after $((i*5)) s"; break; }
  sleep 5
done

nohup "$V" serve "$MON_MODEL" --revision "$MON_REV" \
  --host 127.0.0.1 --port 8001 --served-model-name "$MON_MODEL" \
  --gpu-memory-utilization "$MON_UTIL" --max-model-len 8192 \
  --enable-auto-tool-choice --tool-call-parser hermes \
  --generation-config vllm --seed 0 $MON_EXTRA \
  > "$LOGDIR/vllm-monitor.log" 2>&1 < /dev/null &
echo "monitor pid $!"
for i in $(seq 1 120); do
  curl -sf http://127.0.0.1:8001/v1/models >/dev/null && { echo "monitor up after $((i*5)) s"; break; }
  sleep 5
done

nvidia-smi --query-gpu=memory.used,memory.total --format=csv
ss -tlnp 2>/dev/null | grep -E ':800[01]' || true
