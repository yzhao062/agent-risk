#!/bin/bash -l
#SBATCH --job-name=arh-e-r3-qwen35
#SBATCH --partition=ghx4
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --gpus-per-node=1
#SBATCH --cpus-per-task=16
#SBATCH --mem=100G
#SBATCH --time=01:45:00
#SBATCH --array=0-5
#SBATCH --output=%x-%A_%a.out
#
# Arm E rung 3: Qwen/Qwen3.5-27B-FP8 @ 97f5941bf617e31c5e237364a8602ce3f03a551a, a
# real release (Qwen3-Coder XML tool parser, qwen3 reasoning parser, thinking off,
# --language-model-only, --max-num-seqs 256 for its Gated DeltaNet cache).
# Six 1-GPU array tasks, each replaying every sixth episode of bi_baseline and
# c1_baseline, every step, one request at a time: batch-invariant mode does not
# cover the Gated DeltaNet state, and a lone request is processed in the same
# prefill chunks every time. Batch-invariant mode is still requested so that the
# dense layers match the other rungs; if the server refuses it, the task falls
# back to default kernels and says so. Every task also sends the probe set, so
# the probe compares six GPUs. Pass --account on the sbatch command line.
set -uo pipefail
module load python/3.11.9 > /dev/null 2>&1 || true
T=${SLURM_ARRAY_TASK_ID:-0}
DELTA_BASE=${DELTA_BASE:-/path/to/work/agent-risk-e}
OUT=${DELTA_OUT:-$DELTA_BASE/results/arm-e/r3_qwen35-${SLURM_ARRAY_JOB_ID:-${SLURM_JOB_ID:-0}}/task$T}
mkdir -p "$OUT"
source "$(dirname "$0")/arm_e_lib.sh"
job_meta r3_qwen35

M=Qwen/Qwen3.5-27B-FP8
R=97f5941bf617e31c5e237364a8602ce3f03a551a
export VLLM_BATCH_INVARIANT=1
if ! start_server q35_bi "$M" "$R" "${QWEN35[@]}"; then
  stop_server
  unset VLLM_BATCH_INVARIANT
  stamp "FALLBACK: batch-invariant start failed; default kernels"
  echo default > "$OUT/kernel_mode.txt"
  start_server q35_default "$M" "$R" "${QWEN35[@]}" || { stamp "server failed"; exit 1; }
else
  echo batch_invariant > "$OUT/kernel_mode.txt"
fi
rep bi_shard "$M" "$STEPS_BI" -- --mode all --concurrency 1 --shard "$T/6"
rep c1_shard "$M" "$STEPS_C1" -- --mode all --concurrency 1 --shard "$T/6"
rep probe "$M" "$STEPS_BI" "$STEPS_C1" -- --select "$PROBE" --concurrency 1
server_metrics q35
stop_server
stamp "ARM_E_R3_TASK${T}_DONE"
