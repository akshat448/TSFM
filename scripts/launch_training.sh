#!/usr/bin/env bash
# Launches training on this shared, bare (no SLURM/scheduler) multi-GPU box,
# pinned to one specific physical GPU so it doesn't collide with other
# users' jobs on the other 7.
#
# This REPLACES an earlier slurm/train_rtx6000.sbatch script -- that assumed
# a SLURM-managed cluster, but `squeue` on this machine fails with a DNS/
# config error, meaning there is no SLURM here: it's a bare server (8x RTX
# PRO 6000 Blackwell, confirmed via `nvidia-smi -L`) reached over SSH/VSCode
# tunnel, shared with other users whose jobs are already running on it. A
# plain background launcher is the correct tool here, not sbatch.
#
# Runs inside tmux (falls back to nohup if tmux isn't installed) so it
# survives your AnyDesk/VSCode-tunnel session disconnecting.
#
# Usage:
#   ./scripts/launch_training.sh [--gpu N] [-- <extra args forwarded to train_cluster.py>]
#   ./scripts/launch_training.sh                                  # defaults to GPU 7
#   ./scripts/launch_training.sh --gpu 3 -- --subjects 01,02
#
# Then:
#   tmux attach -t chisco-train      # watch live output
#   tail -f logs/train_$(date +%Y%m%d)*.log   # or just tail the log file
#   Ctrl-b d                         # detach (job keeps running) -- do NOT Ctrl-C

set -euo pipefail

REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$REPO_ROOT"
mkdir -p logs

GPU=7
EXTRA_ARGS=()
while [[ $# -gt 0 ]]; do
  case "$1" in
    --gpu) GPU="$2"; shift 2 ;;
    --) shift; EXTRA_ARGS=("$@"); break ;;
    *) echo "Unknown arg: $1 (extra train_cluster.py args must come after --)" >&2; exit 1 ;;
  esac
done

echo "Checking GPU ${GPU} before committing a long job to it..."
nvidia-smi --query-gpu=index,utilization.gpu,memory.used,memory.total --format=csv -i "$GPU"
echo "(shared box -- if this GPU is already near-saturated, consider --gpu <idx> for a freer one)"

LOG_FILE="logs/train_$(date +%Y%m%d_%H%M%S)_gpu${GPU}.log"

run_training() {
  # CUDA_DEVICE_ORDER=PCI_BUS_ID is required here: without it, CUDA's device
  # enumeration can differ from nvidia-smi's index ordering on multi-GPU
  # boxes, and CUDA_VISIBLE_DEVICES=N would silently pin to the wrong
  # physical card. See the GPU-confirmation commands used to validate this
  # before running for real.
  export CUDA_DEVICE_ORDER=PCI_BUS_ID
  export CUDA_VISIBLE_DEVICES="$GPU"

  if [ ! -d ".venv-cluster" ]; then
    python3 -m venv .venv-cluster
  fi
  source .venv-cluster/bin/activate
  python -m pip install --quiet --upgrade pip

  # RTX PRO 6000 Blackwell needs a recent CUDA build -- cu121/cu124 wheels
  # predate Blackwell (compute capability 12.0) kernel support and will
  # either fail to find a usable kernel or refuse to run on this card.
  # cu128 is the earliest PyTorch wheel index with Blackwell support as of
  # this writing; if this errors on your exact torch/driver combo, check
  # https://pytorch.org/get-started/locally/ for the current recommended
  # index URL and swap it in.
  python -m pip install --quiet torch --index-url https://download.pytorch.org/whl/cu128
  python -m pip install --quiet -r requirements-cluster.txt
  python -m pip install --quiet "causal-conv1d>=1.4.0"
  python -m pip install --quiet "mamba-ssm>=2.2.0"

  echo "Verifying torch sees exactly 1 GPU (the pinned one) and Blackwell kernels work:"
  python -c "
import torch
assert torch.cuda.device_count() == 1, f'expected 1 visible GPU, got {torch.cuda.device_count()}'
print('device:', torch.cuda.get_device_name(0))
print('compute capability:', torch.cuda.get_device_capability(0))
x = torch.randn(1024, 1024, device='cuda')
(x @ x).sum().item()  # forces a real kernel launch, not just device query
print('CUDA matmul kernel launch OK')
"

  if [ -z "$(find data/derivatives/preprocessed_pkl -name '*.pkl' 2>/dev/null)" ]; then
    echo "No local CHISCO data found -- downloading..."
    ./scripts/download_chisco_full.sh
  else
    echo "CHISCO data already present locally, skipping download."
  fi

  python scripts/train_cluster.py \
    --mamba-backend mamba_ssm \
    --device cuda \
    --wandb-project "${WANDB_PROJECT:-chisco-tsfm}" \
    "${EXTRA_ARGS[@]}"
}

if command -v tmux &>/dev/null; then
  SESSION="chisco-train"
  if tmux has-session -t "$SESSION" 2>/dev/null; then
    echo "tmux session '$SESSION' already exists -- attach with: tmux attach -t $SESSION"
    exit 1
  fi
  export -f run_training
  export GPU EXTRA_ARGS LOG_FILE REPO_ROOT
  tmux new-session -d -s "$SESSION" "bash -c 'run_training 2>&1 | tee $LOG_FILE'"
  echo "Launched in tmux session '$SESSION'. Attach with: tmux attach -t $SESSION"
  echo "Log file: $LOG_FILE"
else
  echo "tmux not found -- falling back to nohup (less robust to tunnel drops than tmux)."
  nohup bash -c "$(declare -f run_training); run_training" > "$LOG_FILE" 2>&1 &
  echo "Launched in background, PID $!. Log file: $LOG_FILE"
fi
