#!/bin/bash -l
#SBATCH --job-name=arh-e-r1-same
#SBATCH --partition=ghx4
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --gpus-per-node=1
#SBATCH --cpus-per-task=16
#SBATCH --mem=100G
#SBATCH --time=01:45:00
#SBATCH --output=%x-%j.out
#
# Arm E rung 1: the logged model itself (Qwen3-14B-FP8 at the logged revision),
# the same-model reproducibility reference for assumption (R). Pass --account on
# the sbatch command line.
#   bi session      : VLLM_BATCH_INVARIANT=1, as the bi_baseline log was served.
#                     bi_baseline, every step, 32 requests in flight; c1_baseline,
#                     every step (a kernel change for that log: it was served with
#                     default kernels); the probe set one request at a time.
#   default session : default kernels, as the c1_baseline log was served.
#                     c1_baseline, every step, one request at a time (the log's
#                     own batch size); bi_baseline, stopping each episode at its
#                     first action-only disagreement, one at a time (a kernel
#                     change for that log); the probe set.
# Also checks the seed coupling of the sampling noise across logits widths.
set -uo pipefail
module load python/3.11.9 > /dev/null 2>&1 || true
DELTA_BASE=${DELTA_BASE:-/path/to/work/agent-risk-e}
OUT=${DELTA_OUT:-$DELTA_BASE/results/arm-e/r1_same-${SLURM_JOB_ID:-0}}
mkdir -p "$OUT"
source "$(dirname "$0")/arm_e_lib.sh"
job_meta r1_same
apptainer exec --nv --bind "${DELTA_BIND:-$DELTA_BASE}" "$SIF" python3 "$E/noise_check.py" > "$OUT/noise_check.json" 2> "$OUT/noise_check.err"
cat "$OUT/noise_check.json"

M=Qwen/Qwen3-14B-FP8
R=9a283b4a5efbc09ce247e0ae5b02b744739e525a

export VLLM_BATCH_INVARIANT=1
if start_server bi "$M" "$R" "${HERMES[@]}"; then
  rep bi_all "$M" "$STEPS_BI" -- --mode all --concurrency 32
  rep c1_all_bikernels "$M" "$STEPS_C1" -- --mode all --concurrency 32
  rep probe_bi "$M" "$STEPS_BI" "$STEPS_C1" -- --select "$PROBE" --concurrency 1
  server_metrics bi
fi
stop_server
unset VLLM_BATCH_INVARIANT

if start_server default "$M" "$R" "${HERMES[@]}"; then
  rep c1_all_default "$M" "$STEPS_C1" -- --mode all --concurrency 1
  rep bi_early_default "$M" "$STEPS_BI" -- --mode early --concurrency 1
  rep probe_default "$M" "$STEPS_BI" "$STEPS_C1" -- --select "$PROBE" --concurrency 1
  server_metrics default
fi
stop_server
stamp "ARM_E_R1_DONE"
