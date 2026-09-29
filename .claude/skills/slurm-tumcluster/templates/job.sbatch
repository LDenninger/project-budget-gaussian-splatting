#!/bin/bash
#
# Canonical sbatch template for the CVAI/I9 TUM cluster.
#
# HARD RULES (violating these fails silently, not loudly):
#   1. No whitespace inside a directive value. `--gres=gpu:1, VRAM:48G` makes SLURM
#      stop parsing at the space and discard the rest of the line.
#   2. No `$VAR` or `~` in --output/--error/--chdir. SLURM does not run a shell over
#      these; the literal text becomes part of the path.
#   3. Every #SBATCH line must sit above the first command. Directives below it are
#      ignored without warning.
#   4. The --output/--error directory must already exist. SLURM will not create it,
#      and the job dies with no log to explain why.
#
# Replace every <PLACEHOLDER>, then validate before submitting:
#   python .claude/skills/slurm-tumcluster/scripts/validate_sbatch.py <this file>

#--- Identity ---
#SBATCH --job-name=<JOB_NAME>

#--- Resources -------------------------------------------------------------------
# cpus-per-task should match the dataloader worker count (pytorch num_workers).
# 3 CPUs per GPU is a sane default for a normal DL job; 32G RAM is the usual ask.
# Request honestly: over-requesting lengthens your queue wait and starves others,
# under-requesting gets the job killed mid-run.
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=3
#SBATCH --mem=32G
#SBATCH --gres=gpu:1,VRAM:24G
#SBATCH --time=<D-HH:MM:SS>

#--- Node selection --------------------------------------------------------------
# Prefer a VRAM floor or a broad --constraint over pinning to specific nodes: the
# more nodes your job can land on, the sooner it starts. Uncomment what you need.
#
##SBATCH --constraint="GPU_GEN:AMPERE|GPU_GEN:ADA|GPU_GEN:HOPPER"
##SBATCH --exclude=<NODE>
##SBATCH --nodelist=<NODE>

#--- Deadline mode ---------------------------------------------------------------
# DEADLINE preempts other people's running jobs, so it needs a written
# justification. Add --constraint=NOTBIG unless you specifically need a 48G GPU,
# otherwise DEADLINEBIG can preempt you in turn.
#
##SBATCH --partition=DEADLINE
##SBATCH --comment="<actual deadline name, e.g. CVPR 2027 rebuttal>"
##SBATCH --constraint=NOTBIG

#--- Logging ---------------------------------------------------------------------
# Absolute path, %j expands to the job id. mkdir -p this directory first.
#SBATCH --output=/storage/slurm/<USER>/logs/%x-%j.out
#SBATCH --error=/storage/slurm/<USER>/logs/%x-%j.err

#--- Notification ----------------------------------------------------------------
#SBATCH --mail-type=END,FAIL
#SBATCH --mail-user=<EMAIL>

# ================================================================================
# Everything below this line is the job body. No #SBATCH directives past here.
# ================================================================================

# Fail loudly: without this, a failed `conda activate` is ignored and the job
# reports COMPLETED after doing nothing.
set -euo pipefail

echo "job      : ${SLURM_JOB_ID} (${SLURM_JOB_NAME})"
echo "node     : $(hostname)"
echo "gpus     : ${CUDA_VISIBLE_DEVICES:-none}"
echo "restarts : ${SLURM_RESTART_COUNT:-0}"
echo "started  : $(date --iso-8601=seconds)"
nvidia-smi --query-gpu=name,memory.total --format=csv,noheader || true

#--- Environment ---
# The batch shell is non-interactive, so ~/.bashrc conda hooks do not run.
# Source conda explicitly rather than relying on `conda` being on PATH.
source "$(conda info --base)/etc/profile.d/conda.sh"
conda activate <CONDA_ENV>

#--- Preemption-aware resume ---
# If SLURM requeued this job after preemption, resume instead of starting over.
RESUME_FLAG=""
if [ "${SLURM_RESTART_COUNT:-0}" -gt 0 ]; then
    echo "requeued after preemption — resuming from latest checkpoint"
    RESUME_FLAG="--resume"
fi

#--- Run ---
# --unbuffered keeps stdout flowing in real time; without it glibc buffers the
# output and a crashed job loses its last and most informative lines.
cd <PROJECT_DIR>
srun --unbuffered python <SCRIPT>.py ${RESUME_FLAG} <ARGS>

echo "finished : $(date --iso-8601=seconds)"
