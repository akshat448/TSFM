#!/usr/bin/env bash
# The actual training work, factored out of launch_training.sh into a real
# file (not a bash function passed via `export -f` across a tmux/nohup shell
# boundary) -- that pattern was tested and found to silently fail
# ("command not found", exit 127) once nested through tmux's own shell
# invocation on the real target machine, discarding the crash it should have
# reported. A plain script file has no such cross-shell export requirement.
#
# Usage: _train_worker.sh <gpu-index> [extra train_cluster.py args...]

set -euo pipefail

GPU="$1"
shift
EXTRA_ARGS=("$@")

REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$REPO_ROOT"

# CUDA_DEVICE_ORDER=PCI_BUS_ID is required here: without it, CUDA's device
# enumeration can differ from nvidia-smi's index ordering on multi-GPU boxes,
# and CUDA_VISIBLE_DEVICES=N would silently pin to the wrong physical card.
export CUDA_DEVICE_ORDER=PCI_BUS_ID
export CUDA_VISIBLE_DEVICES="$GPU"

# python3 -m venv failed on the real target box: "ensurepip is not
# available" -- python3.12-venv isn't installed system-wide, and we don't
# want to require sudo. Prefer conda (confirmed present on that box, prompt
# showed "(base) mbz-imran@...") since it doesn't depend on ensurepip at
# all; only fall back to venv if conda genuinely isn't available.
if command -v conda &>/dev/null; then
  ENV_NAME=chisco-cluster
  # shellcheck disable=SC1091
  source "$(conda info --base)/etc/profile.d/conda.sh"
  if ! conda env list | awk '{print $1}' | grep -qx "$ENV_NAME"; then
    conda create -y -n "$ENV_NAME" python=3.11
  fi
  conda activate "$ENV_NAME"
elif [ -d ".venv-cluster" ] && [ -f ".venv-cluster/bin/activate" ]; then
  source .venv-cluster/bin/activate
elif python3 -m venv .venv-cluster 2>/tmp/venv_err_$$; then
  source .venv-cluster/bin/activate
else
  rm -rf .venv-cluster  # clean up whatever partial dir the failed attempt left behind
  echo "Neither conda nor a working 'python3 -m venv' is available:" >&2
  cat /tmp/venv_err_$$ >&2
  rm -f /tmp/venv_err_$$
  echo "Fix: install/activate conda (no sudo needed), or 'sudo apt install python3.12-venv'." >&2
  exit 1
fi
python -m pip install --quiet --upgrade pip

# RTX PRO 6000 Blackwell needs a recent CUDA build -- cu121/cu124 wheels
# predate Blackwell (compute capability 12.0) kernel support and will either
# fail to find a usable kernel or refuse to run on this card. cu128 is the
# earliest PyTorch wheel index with Blackwell support as of this writing; if
# this errors on your exact torch/driver combo, check
# https://pytorch.org/get-started/locally/ for the current recommended index
# URL and swap it in.
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
