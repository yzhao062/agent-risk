#!/usr/bin/env bash
# E2 serving (E2-PREREG-2026-10-05.md): two vLLM servers on one H100 80 GB, localhost only.
#   agent   Qwen/Qwen3.5-27B-FP8 @ 97f5941b, served from its snapshot directory on the
#           weights volume, 127.0.0.1:8000, default kernels (vLLM 0.30.0 refuses
#           VLLM_BATCH_INVARIANT for this architecture), memory fraction AGENT_UTIL (0.60)
#   monitor Qwen/Qwen3-4B-Instruct-2507 @ cdbee75f, 127.0.0.1:8001, the D58/D64 command
#           line byte for byte, VLLM_BATCH_INVARIANT=1, memory fraction 0.30
# The agent starts first so that the monitor sees the agent's memory as used.
# GDN_BACKEND (default: vLLM's default, flashinfer) sets --gdn-prefill-backend if the
# default cannot start on this host (the smoke records which one runs).
#   E2_HF=/mnt/e2hf bash e2_serve.sh
set -uo pipefail
V="${VLLM_BIN:-$HOME/venv-vllm/bin/vllm}"
E2_HF=${E2_HF:-/mnt/e2hf}
AGENT_REV=97f5941bf617e31c5e237364a8602ce3f03a551a
AGENT_PATH="$E2_HF/hub/models--Qwen--Qwen3.5-27B-FP8/snapshots/$AGENT_REV"
AGENT_UTIL=${AGENT_UTIL:-0.60}
GDN_BACKEND=${GDN_BACKEND:-}
LOGDIR="${LOGDIR:-$HOME/logs}"
mkdir -p "$LOGDIR"
[ -f "$AGENT_PATH/config.json" ] || { echo "agent snapshot missing: $AGENT_PATH"; exit 2; }
export HF_HUB_OFFLINE=1 VLLM_USE_FLASHINFER_SAMPLER=0 VLLM_USE_DEEP_GEMM=0
GDN_ARGS=()
[ -n "$GDN_BACKEND" ] && GDN_ARGS=(--gdn-prefill-backend "$GDN_BACKEND")

env -u VLLM_BATCH_INVARIANT nohup "$V" serve "$AGENT_PATH" \
  --host 127.0.0.1 --port 8000 --served-model-name Qwen/Qwen3.5-27B-FP8 \
  --gpu-memory-utilization "$AGENT_UTIL" --max-model-len 32768 --max-num-seqs 256 \
  --enable-auto-tool-choice --tool-call-parser qwen3_coder --reasoning-parser qwen3 --language-model-only \
  --default-chat-template-kwargs '{"enable_thinking": false}' \
  --generation-config vllm --seed 0 --no-enable-prefix-caching "${GDN_ARGS[@]}" \
  > "$LOGDIR/vllm-agent-e2.log" 2>&1 < /dev/null &
echo "agent pid $!"
for i in $(seq 1 180); do
  curl -sf http://127.0.0.1:8000/v1/models >/dev/null && { echo "agent up after $((i*5)) s"; break; }
  sleep 5
done
curl -sf http://127.0.0.1:8000/v1/models >/dev/null || { echo "agent did not come up"; tail -30 "$LOGDIR/vllm-agent-e2.log"; exit 3; }

VLLM_BATCH_INVARIANT=1 nohup "$V" serve Qwen/Qwen3-4B-Instruct-2507 --revision cdbee75f17c01a7cc42f958dc650907174af0554 \
  --host 127.0.0.1 --port 8001 --served-model-name Qwen/Qwen3-4B-Instruct-2507 \
  --gpu-memory-utilization 0.30 --max-model-len 8192 \
  --enable-auto-tool-choice --tool-call-parser hermes \
  --generation-config vllm --seed 0 --no-enable-prefix-caching \
  > "$LOGDIR/vllm-monitor-e2.log" 2>&1 < /dev/null &
echo "monitor pid $!"
for i in $(seq 1 120); do
  curl -sf http://127.0.0.1:8001/v1/models >/dev/null && { echo "monitor up after $((i*5)) s"; break; }
  sleep 5
done
curl -sf http://127.0.0.1:8001/v1/models >/dev/null || { echo "monitor did not come up"; tail -30 "$LOGDIR/vllm-monitor-e2.log"; exit 3; }

grep -h "GPU KV cache size\|Model loading took\|Available KV cache memory\|attention block size\|gdn\|GDN" \
  "$LOGDIR/vllm-agent-e2.log" "$LOGDIR/vllm-monitor-e2.log" | tail -12
nvidia-smi --query-gpu=memory.used,memory.total --format=csv
ss -tlnp 2>/dev/null | grep -E ':800[01]' || true
