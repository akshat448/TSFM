#!/usr/bin/env bash
#SBATCH --job-name=chisco-tsfm
#SBATCH --partition=gpu                 # CLUSTER-SPECIFIC: rename to your site's GPU partition name
#SBATCH --gres=gpu:rtx6000:1            # CLUSTER-SPECIFIC: rename if your scheduler uses --gpus= or a
                                         # different GPU type string (e.g. --gpus=rtx_6000:1) -- check
                                         # `sinfo -o "%P %G"` on your cluster to confirm the right name
#SBATCH --cpus-per-task=8
#SBATCH --mem=64G
#SBATCH --time=24:00:00
#SBATCH --output=logs/%x-%j.out
#SBATCH --error=logs/%x-%j.err

# End-to-end: download CHISCO (if not already present) -> install deps ->
# train with the real CUDA mamba_ssm kernel -> log everything to wandb.
#
# Usage:
#   sbatch slurm/train_rtx6000.sbatch
#   sbatch slurm/train_rtx6000.sbatch --subjects 01,02 --ssl-epochs 50   # extra args forwarded to train_cluster.py
#
# Assumes:
#   - .env exists at repo root with WANDB_API_KEY set (see .env.sample) --
#     OR WANDB_API_KEY is already exported in your shell before `sbatch` is run
#     (sbatch does NOT inherit your interactive env by default unless you add
#     --export=ALL or your site's equivalent -- safest is a real .env file).
#   - AWS CLI is available on compute nodes for the S3 download (module load
#     awscli, or pip install awscli --user, depending on your site).

set -euo pipefail

# Pin to physical GPU device index 7 on the node SLURM assigns this job to.
# IMPORTANT CAVEAT: this only does what you want if your site's SLURM is
# NOT using cgroup-based GPU isolation for the `gpu` GRES type -- on most
# properly-configured SLURM clusters, `--gres=gpu:rtx6000:1` already
# restricts the job to exactly one (SLURM-chosen) GPU and remaps it to
# device index 0 inside the job's cgroup, in which case setting
# CUDA_VISIBLE_DEVICES=7 here would be wrong (there IS no device 7 visible
# inside the job) and would make CUDA fail to find a GPU at all. Verify
# with `nvidia-smi -L` in an interactive job on this cluster first -- if it
# lists all 8 GPUs (not just 1), cgroup isolation is off and this line does
# what's intended; if it lists only 1 GPU, delete this line and instead ask
# your cluster admin how to request a specific physical GPU index (usually
# a site-specific --gres or --constraint flag, not a generic SLURM feature).
export CUDA_VISIBLE_DEVICES=7

REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$REPO_ROOT"
mkdir -p logs

# ---- 1. Environment setup ----
# CLUSTER-SPECIFIC: replace with your site's module system if it uses one,
# e.g. `module load cuda/12.1 python/3.11`. Left generic here since this
# varies a lot between clusters.
if [ ! -d ".venv-cluster" ]; then
  python3 -m venv .venv-cluster
fi
source .venv-cluster/bin/activate
python -m pip install --quiet --upgrade pip

# Pin to the CUDA build matching your cluster's driver -- cu121 is a common
# default as of this writing; check `nvidia-smi` for your actual CUDA
# version and adjust the index URL below if needed.
python -m pip install --quiet torch --index-url https://download.pytorch.org/whl/cu121
python -m pip install --quiet -r requirements-cluster.txt

# Real CUDA mamba-ssm kernel -- this is the whole point of running on the
# RTX 6000 instead of locally (see chisco_pipeline/encoder.py backend note).
# Building these from source can take several minutes; that's expected.
python -m pip install --quiet "causal-conv1d>=1.4.0"
python -m pip install --quiet "mamba-ssm>=2.2.0"

# ---- 2. Data ----
if [ -z "$(find data/derivatives/preprocessed_pkl -name '*.pkl' 2>/dev/null)" ]; then
  echo "No local CHISCO data found -- downloading (this can take a while for all subjects)..."
  ./scripts/download_chisco_full.sh
else
  echo "CHISCO data already present locally, skipping download."
fi

# ---- 3. Train ----
python scripts/train_cluster.py \
  --mamba-backend mamba_ssm \
  --device cuda \
  --wandb-project "${WANDB_PROJECT:-chisco-tsfm}" \
  "$@"
