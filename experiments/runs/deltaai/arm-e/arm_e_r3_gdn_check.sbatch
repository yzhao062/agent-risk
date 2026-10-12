#!/bin/bash -l
#SBATCH --job-name=arh-e-r3-gdncheck
#SBATCH --partition=ghx4
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --gpus-per-node=1
#SBATCH --cpus-per-task=16
#SBATCH --mem=100G
#SBATCH --time=01:00:00
#SBATCH --array=0-1
#SBATCH --output=%x-%A_%a.out
#
# Arm E rung 3 diagnostic: is Qwen3.5-27B-FP8 seed-addressable on vLLM 0.30.0?
# The rung 3 probe gave identical tokens in all six task processes for only 24 of
# 40 steps. Each array task (one GPU) starts two server processes for each Gated
# DeltaNet prefill backend (flashinfer, the default; triton; cutedsl) and sends the
# 16 divergent probe steps plus 4 stable ones, one request at a time. Same flags
# as rung 3 otherwise. Pass --account on the sbatch command line.
set -uo pipefail
module load python/3.11.9 > /dev/null 2>&1 || true
T=${SLURM_ARRAY_TASK_ID:-0}
DELTA_BASE=${DELTA_BASE:-/path/to/work/agent-risk-e}
OUT=${DELTA_OUT:-$DELTA_BASE/results/arm-e/r3_gdncheck-${SLURM_ARRAY_JOB_ID:-${SLURM_JOB_ID:-0}}/task$T}
mkdir -p "$OUT"
source "$(dirname "$0")/arm_e_lib.sh"
job_meta r3_gdncheck
M=Qwen/Qwen3.5-27B-FP8
R=97f5941bf617e31c5e237364a8602ce3f03a551a
for be in flashinfer triton cutedsl; do
  for proc in a b; do
    if start_server "${be}_$proc" "$M" "$R" "${QWEN35[@]}" --gdn-prefill-backend "$be"; then
      rep "probe_${be}_$proc" "$M" "$STEPS_BI" "$STEPS_C1" -- --select "$E/data/probe_q35_div.json" --concurrency 1
    else
      stamp "backend $be failed to start"
    fi
    stop_server
  done
done
stamp "ARM_E_R3_GDNCHECK_TASK${T}_DONE"
