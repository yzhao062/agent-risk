# Arm E job helpers (sourced by the arm E sbatch scripts after OUT is set).
# Server start and stop come from ../serve_lib.sh: one model per GPU, a per-job
# port, process-group shutdown, and the engine flags (--max-model-len
# 32768, --generation-config vllm, --seed 0, enable_thinking false through
# --default-chat-template-kwargs, VLLM_USE_FLASHINFER_SAMPLER=0,
# VLLM_USE_DEEP_GEMM=0). Every replay server also runs with prefix caching off.

SERVE_LIB="${DELTA_SERVE_LIB:-$(dirname "$0")/../serve_lib.sh}"
source "$SERVE_LIB"
E=${ARM_E_DIR:-$(cd "$(dirname "$0")" && pwd)}
STEPS_BI=${STEPS_BI:-$E/data/steps_bi_baseline.jsonl}
STEPS_C1=${STEPS_C1:-$E/data/steps_c1_baseline.jsonl}
PROBE=${PROBE:-$E/data/probe.json}
NOTHINK='{"enable_thinking": false}'
HERMES=(--enable-auto-tool-choice --tool-call-parser hermes --default-chat-template-kwargs "$NOTHINK" --no-enable-prefix-caching)
QWEN35=(--enable-auto-tool-choice --tool-call-parser qwen3_coder --reasoning-parser qwen3 --language-model-only
        --max-num-seqs 256 --default-chat-template-kwargs "$NOTHINK" --no-enable-prefix-caching)

job_meta() {  # job_meta RUNG
  {
    echo "{\"rung\": \"$1\", \"job_id\": \"${SLURM_JOB_ID:-}\", \"array_job_id\": \"${SLURM_ARRAY_JOB_ID:-}\","
    echo " \"array_task\": \"${SLURM_ARRAY_TASK_ID:-}\", \"account\": \"${SLURM_JOB_ACCOUNT:-}\","
    echo " \"partition\": \"${SLURM_JOB_PARTITION:-}\", \"host\": \"$(hostname)\", \"start_utc\": \"$(date -u +%FT%TZ)\","
    echo " \"gpu\": \"$(nvidia-smi --query-gpu=name,driver_version,memory.total --format=csv,noheader 2>/dev/null | head -1 || echo unknown)\"}"
  } > "$OUT/job_meta.json"
  cat "$OUT/job_meta.json"
}

# rep TAG MODEL STEPS_FILE(S)... -- [arm_e_replay args]
rep() {
  local tag=$1 model=$2
  shift 2
  local steps=()
  while [ "$1" != "--" ]; do steps+=("$1"); shift; done
  shift
  stamp "replay $tag start"
  $PY "$E/arm_e_replay.py" --steps "${steps[@]}" --base-url "$BASE_URL" --model "$model" \
    --out "$OUT/$tag.jsonl" --tag "$tag" "$@" 2>&1 | tail -3
  stamp "replay $tag end"
}

# server_metrics TAG: preemption and token counters of the running server
server_metrics() {
  curl -sf "http://127.0.0.1:$PORT/metrics" | grep -E "^vllm:(num_preemptions|prompt_tokens|generation_tokens|request_success)" \
    > "$OUT/metrics_$1.txt" 2> /dev/null || true
  stamp "metrics $1: $(grep -h num_preemptions "$OUT/metrics_$1.txt" 2>/dev/null | tr '\n' ' ' || true)"
}
