# Shared helpers for the DeltaAI arm E jobs (sourced by the sbatch scripts).
# Serves one model at a time from the vLLM 0.30.0 aarch64 container on
# 127.0.0.1:$PORT. Engine flags follow the Jetstream2 agent server
# (serve.sh): --max-model-len 32768, --generation-config vllm, --seed 0,
# enable_thinking false, plus the per-config flags passed by the caller.
#
# Several jobs can share one GH200 node (1 GPU each), so everything here is
# scoped to this job: a port derived from the job id, a server started in its
# own process session and stopped by process group (never a node-wide pkill),
# and a readiness check that confirms the served model name.

B=${DELTA_BASE:-/work/nvme/<ACCOUNT>/<USER>/agent-risk-e}
H=${HARNESS_DIR:-$B/harness}
SIF=${VLLM_SIF:-$B/containers/vllm-openai_v0.30.0-aarch64.sif}
PY=${CLIENT_PYTHON:-$B/venv-client/bin/python}
PORT=${PORT:-$((20000 + ${SLURM_JOB_ID:-0} % 20000))}
BASE_URL=http://127.0.0.1:$PORT/v1

export HF_HOME=${HF_HOME:-$B/hf} HF_HUB_OFFLINE=1
export TIKTOKEN_ENCODINGS_BASE=${TIKTOKEN_ENCODINGS_BASE:-$B/tiktoken} TIKTOKEN_RS_CACHE_DIR=${TIKTOKEN_RS_CACHE_DIR:-$B/tiktoken}
export VLLM_CACHE_ROOT=${VLLM_CACHE_ROOT:-$B/vllm-cache} TRITON_CACHE_DIR=${TRITON_CACHE_DIR:-$B/triton-cache}
# Same sampler path as H100 (PyTorch sampler, no FlashInfer top-k/top-p JIT).
export VLLM_USE_FLASHINFER_SAMPLER=0
# Precompiled CUTLASS block-FP8 GEMMs, no DeepGEMM JIT.
export VLLM_USE_DEEP_GEMM=0

stamp() { echo "[$(date -u +%FT%TZ)] $*"; }

gpu_used_mib() { nvidia-smi --query-gpu=memory.used --format=csv,noheader,nounits 2>/dev/null | head -1 | tr -d ' ' || echo 0; }

# start_server TAG MODEL REVISION [vllm flags...]; extra env vars are taken from
# the caller's environment (apptainer passes the host environment through).
start_server() {
  local tag=$1 model=$2 rev=$3
  shift 3
  if curl -sf "$BASE_URL/models" > /dev/null 2>&1; then
    stamp "port $PORT already answers; refusing to start $tag"
    return 1
  fi
  stamp "start server $tag on port $PORT (GPU used $(gpu_used_mib) MiB): $model@$rev $*"
  setsid apptainer exec --nv --bind "${DELTA_BIND:-$B}" "$SIF" \
    vllm serve "$model" --revision "$rev" --host 127.0.0.1 --port "$PORT" \
    --served-model-name "$model" --max-model-len 32768 --generation-config vllm --seed 0 \
    --gpu-memory-utilization 0.85 "$@" > "$OUT/vllm-$tag.log" 2>&1 < /dev/null &
  SERVER_PID=$!
  local i
  for i in $(seq 1 240); do
    if curl -sf "$BASE_URL/models" 2> /dev/null | grep -q "\"id\":\"$model\""; then
      stamp "server $tag up after $((i * 5)) s (pid $SERVER_PID, pgid $(ps -o pgid= -p "$SERVER_PID" | tr -d ' '))"
      return 0
    fi
    if ! kill -0 "$SERVER_PID" 2> /dev/null; then
      stamp "server $tag exited; last error lines:"
      grep -v "Capturing CUDA graphs" "$OUT/vllm-$tag.log" | grep -E "Error|error" | tail -5
      return 1
    fi
    sleep 5
  done
  stamp "server $tag did not come up in 20 min"
  return 1
}

stop_server() {
  if [ -n "${SERVER_PID:-}" ]; then
    kill -TERM -- "-$SERVER_PID" 2> /dev/null
    for _ in $(seq 1 30); do kill -0 "$SERVER_PID" 2> /dev/null || break; sleep 2; done
    kill -KILL -- "-$SERVER_PID" 2> /dev/null
    SERVER_PID=
  fi
  # wait for this job's GPU memory to be released before the next server
  for _ in $(seq 1 30); do
    [ "$(gpu_used_mib)" -lt 1024 ] && break
    sleep 2
  done
  stamp "server stopped; GPU used $(gpu_used_mib) MiB"
}

# vrun TAG MODEL BATCH [extra validate_repro args]
vrun() {
  local tag=$1 model=$2 batch=$3
  shift 3
  $PY "$H/validate_repro.py" run --base-url "$BASE_URL" --model "$model" \
    --requests "$H/fixed_requests.json" --batch "$batch" --max-tokens 4096 --tag "$tag" --out "$OUT/$tag.json" "$@"
}

trap 'stop_server' EXIT
