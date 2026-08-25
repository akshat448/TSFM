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
#   ./scripts/launch_training.sh [--gpu N|auto] [--min-free-mib N] [-- <extra args forwarded to train_cluster.py>]
#   ./scripts/launch_training.sh                                  # auto-picks a GPU (default)
#   ./scripts/launch_training.sh --gpu 3 -- --subjects 01,02
#
# --gpu auto (the default) picks the physical GPU with the LOWEST reported
# utilization among those with at least --min-free-mib (default 10000) MiB
# free, using `nvidia-smi --query-gpu`. This is a snapshot heuristic on a
# box shared with other users, not a reservation -- someone else's job can
# still start on the same GPU a second later. It's meant to avoid the
# obviously-bad choice (a GPU already near 100% util or nearly out of
# memory), not to guarantee exclusivity the way a real scheduler would.
#
# Then:
#   tmux attach -t chisco-train      # watch live output
#   tail -f logs/train_$(date +%Y%m%d)*.log   # or just tail the log file
#   Ctrl-b d                         # detach (job keeps running) -- do NOT Ctrl-C

set -euo pipefail

REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$REPO_ROOT"
mkdir -p logs

GPU="auto"
MIN_FREE_MIB=10000
EXTRA_ARGS=()
while [[ $# -gt 0 ]]; do
  case "$1" in
    --gpu) GPU="$2"; shift 2 ;;
    --min-free-mib) MIN_FREE_MIB="$2"; shift 2 ;;
    --) shift; EXTRA_ARGS=("$@"); break ;;
    *) echo "Unknown arg: $1 (extra train_cluster.py args must come after --)" >&2; exit 1 ;;
  esac
done

echo "Current GPU state on this box:"
nvidia-smi --query-gpu=index,utilization.gpu,memory.used,memory.total --format=csv

if [[ "$GPU" == "auto" ]]; then
  # index,util%,used_mib,total_mib -> free_mib = total-used; among GPUs with
  # free_mib >= MIN_FREE_MIB, pick lowest utilization (util is the more
  # volatile/representative-of-contention signal; free memory is filtered
  # as a hard floor rather than optimized, since we don't know this job's
  # exact memory footprint yet).
  GPU=$(nvidia-smi --query-gpu=index,utilization.gpu,memory.used,memory.total \
          --format=csv,noheader,nounits \
        | awk -F', *' -v min_free="$MIN_FREE_MIB" '
            { free = $4 - $3; if (free >= min_free) print $2, $1, free }
          ' \
        | sort -n \
        | head -1 \
        | awk '{print $2}')
  if [[ -z "$GPU" ]]; then
    echo "No GPU has >= ${MIN_FREE_MIB} MiB free right now -- lower --min-free-mib or wait and retry." >&2
    exit 1
  fi
  echo "Auto-selected GPU ${GPU} (lowest utilization among GPUs with >= ${MIN_FREE_MIB} MiB free)"
fi

echo "Checking GPU ${GPU} before committing a long job to it..."
nvidia-smi --query-gpu=index,utilization.gpu,memory.used,memory.total --format=csv -i "$GPU"

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

launch_with_nohup() {
  echo "Falling back to nohup (less robust to tunnel drops than tmux -- if the SSH/VSCode"
  echo "tunnel itself dies, not just your terminal, nohup survives but you lose the ability"
  echo "to re-attach and watch live; tail -f the log file instead)."
  nohup bash -c "$(declare -f run_training); run_training" > "$LOG_FILE" 2>&1 &
  disown
  echo "Launched in background, PID $!. Log file: $LOG_FILE"
}

if command -v tmux &>/dev/null; then
  # On some shared/containerized boxes, tmux's default socket dir
  # (/tmp/tmux-$UID) is broken -- e.g. something else already created a FILE
  # at that path instead of a directory ("/tmp/tmux-1000 is not a directory"),
  # which isn't something this script can fix (would need root to remove a
  # stray file in shared /tmp), so point tmux at a private socket dir instead.
  export TMUX_TMPDIR="$HOME/.tmux-sockets"
  mkdir -p "$TMUX_TMPDIR"

  SESSION="chisco-train"
  if tmux has-session -t "$SESSION" 2>/dev/null; then
    echo "tmux session '$SESSION' already exists -- attach with: TMUX_TMPDIR=$TMUX_TMPDIR tmux attach -t $SESSION"
    exit 1
  fi
  export -f run_training
  export GPU EXTRA_ARGS LOG_FILE REPO_ROOT
  if tmux new-session -d -s "$SESSION" "bash -c 'run_training 2>&1 | tee $LOG_FILE'" 2>/tmp/tmux_launch_err_$$; then
    echo "Launched in tmux session '$SESSION'."
    echo "Attach with: TMUX_TMPDIR=$TMUX_TMPDIR tmux attach -t $SESSION"
    echo "(add 'export TMUX_TMPDIR=$TMUX_TMPDIR' to your shell rc so plain 'tmux attach' works too)"
    echo "Log file: $LOG_FILE"
  else
    echo "tmux still failed even with a private TMUX_TMPDIR:"
    cat /tmp/tmux_launch_err_$$
    rm -f /tmp/tmux_launch_err_$$
    launch_with_nohup
  fi
else
  echo "tmux not found."
  launch_with_nohup
fi
