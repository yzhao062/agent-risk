#!/bin/bash -l
#SBATCH --job-name=arh-e-hermes
#SBATCH --partition=ghx4
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --gpus-per-node=1
#SBATCH --cpus-per-task=16
#SBATCH --mem=100G
#SBATCH --time=01:30:00
#SBATCH --output=%x-%j.out
#
# Arm E rungs served with the Hermes tool parser and batch-invariant kernels:
#   rung 2  RUNG=r2_bf16    Qwen/Qwen3-14B @ 40c069824f4251a91eefaf281ebe4c544efd3e18 (precision-only update)
#   rung 4  RUNG=r4_qwen25  Qwen/Qwen2.5-14B-Instruct @ cf98f3b3bbb457ad9e2bb7baf9a0125b6b88caa8 (rollback control)
# Submit with --account=... --export=ALL,RUNG=...,RUNG_MODEL=...,RUNG_REV=...
# bi_baseline and c1_baseline, every step, 32 requests in flight (batch-invariant
# serving makes the output independent of the batch), then the probe set one
# request at a time as the batch-invariance check.
set -uo pipefail
module load python/3.11.9 > /dev/null 2>&1 || true
DELTA_BASE=${DELTA_BASE:-/path/to/work/agent-risk-e}
OUT=${DELTA_OUT:-$DELTA_BASE/results/arm-e/${RUNG}-${SLURM_JOB_ID:-0}}
mkdir -p "$OUT"
source "$(dirname "$0")/arm_e_lib.sh"
job_meta "$RUNG"

export VLLM_BATCH_INVARIANT=1
if start_server "$RUNG" "$RUNG_MODEL" "$RUNG_REV" "${HERMES[@]}"; then
  rep bi_all "$RUNG_MODEL" "$STEPS_BI" -- --mode all --concurrency 32
  rep c1_all "$RUNG_MODEL" "$STEPS_C1" -- --mode all --concurrency 32
  rep probe "$RUNG_MODEL" "$STEPS_BI" "$STEPS_C1" -- --select "$PROBE" --concurrency 1
  server_metrics "$RUNG"
fi
stop_server
stamp "ARM_E_${RUNG}_DONE"
