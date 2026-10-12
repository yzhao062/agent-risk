#!/usr/bin/env bash
# E3 serving (E3-PREREG-2026-10-05.md): two vLLM servers on one H100 80 GB, loopback only.
#   agent+user  Qwen/Qwen3-14B-FP8 @ 9a283b4a, 127.0.0.1:8000, DEFAULT kernels
#               (VLLM_BATCH_INVARIANT unset), memory fraction 0.55, the D58 agent flags
#               otherwise (hermes tool parser, thinking off, seed 0, prefix caching off)
#   monitor     Qwen/Qwen3-4B-Instruct-2507 @ cdbee75f, 127.0.0.1:8001, VLLM_BATCH_INVARIANT=1,
#               memory fraction 0.30, prefix caching off; the D58 monitor command line except
#               --max-model-len 32768 (the monitor reads the whole conversation)
# The agent starts first so the monitor's startup check sees the agent's memory as used.
#   bash e3_serve.sh
set -uo pipefail
V="${VLLM_BIN:-$HOME/venv-vllm/bin/vllm}"
LOGDIR="${LOGDIR:-$HOME/logs}"
mkdir -p "$LOGDIR"
export HF_HUB_OFFLINE=1 VLLM_USE_FLASHINFER_SAMPLER=0 VLLM_USE_DEEP_GEMM=0

# MONITOR_ONLY=1 restarts only the monitor (same command line), with the agent server left running.
if [ "${MONITOR_ONLY:-0}" != "1" ]; then
env -u VLLM_BATCH_INVARIANT nohup "$V" serve Qwen/Qwen3-14B-FP8 --revision 9a283b4a5efbc09ce247e0ae5b02b744739e525a \
  --host 127.0.0.1 --port 8000 --served-model-name Qwen/Qwen3-14B-FP8 \
  --gpu-memory-utilization 0.55 --max-model-len 32768 \
  --enable-auto-tool-choice --tool-call-parser hermes \
  --default-chat-template-kwargs '{"enable_thinking": false}' \
  --generation-config vllm --seed 0 --no-enable-prefix-caching \
  > "$LOGDIR/vllm-agent-e3.log" 2>&1 < /dev/null &
echo "agent pid $!"
for i in $(seq 1 180); do
  curl -sf http://127.0.0.1:8000/v1/models >/dev/null && { echo "agent up after $((i*5)) s"; break; }
  sleep 5
done
curl -sf http://127.0.0.1:8000/v1/models >/dev/null || { echo "agent did not come up"; tail -30 "$LOGDIR/vllm-agent-e3.log"; exit 3; }
fi

VLLM_BATCH_INVARIANT=1 nohup "$V" serve Qwen/Qwen3-4B-Instruct-2507 --revision cdbee75f17c01a7cc42f958dc650907174af0554 \
  --host 127.0.0.1 --port 8001 --served-model-name Qwen/Qwen3-4B-Instruct-2507 \
  --gpu-memory-utilization 0.30 --max-model-len 32768 \
  --enable-auto-tool-choice --tool-call-parser hermes \
  --generation-config vllm --seed 0 --no-enable-prefix-caching \
  > "$LOGDIR/vllm-monitor-e3.log" 2>&1 < /dev/null &
echo "monitor pid $!"
for i in $(seq 1 120); do
  curl -sf http://127.0.0.1:8001/v1/models >/dev/null && { echo "monitor up after $((i*5)) s"; break; }
  sleep 5
done
curl -sf http://127.0.0.1:8001/v1/models >/dev/null || { echo "monitor did not come up"; tail -30 "$LOGDIR/vllm-monitor-e3.log"; exit 3; }

grep -h "GPU KV cache size\|Model loading took\|Available KV cache memory\|batch.invariant\|Maximum concurrency" \
  "$LOGDIR/vllm-agent-e3.log" "$LOGDIR/vllm-monitor-e3.log" | tail -12
nvidia-smi --query-gpu=memory.used,memory.total --format=csv
ss -tlnp 2>/dev/null | grep -E ':800[01]' || true
