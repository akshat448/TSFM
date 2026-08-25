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
# showed "(base) mbz-imran@...") since it doesn't depend on ensurepip at all.
#
# `command -v conda` alone is NOT enough to detect it here: `conda init`
# normally wires `conda` onto PATH via ~/.bashrc, which only gets sourced in
# an interactive login shell -- this script runs as a plain non-interactive
# script under tmux/nohup, so PATH lookup can miss a conda that works fine
# in your normal terminal. Search common install locations for
# etc/profile.d/conda.sh directly instead of trusting PATH.
CONDA_SH=""
for base in "$HOME/miniconda3" "$HOME/anaconda3" "$HOME/miniforge3" "/opt/conda" "/opt/miniconda3" "/usr/local/miniconda3"; do
  if [ -f "$base/etc/profile.d/conda.sh" ]; then
    CONDA_SH="$base/etc/profile.d/conda.sh"
    break
  fi
done
if [ -z "$CONDA_SH" ] && command -v conda &>/dev/null; then
  CONDA_SH="$(conda info --base)/etc/profile.d/conda.sh"
fi

if [ -n "$CONDA_SH" ]; then
  echo "Using conda ($CONDA_SH)"
  # shellcheck disable=SC1090
  source "$CONDA_SH"
  ENV_NAME=chisco-cluster
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

# RTX PRO 6000 Blackwell needs a recent CUDA build. Pinned to cu130, not
# cu128: this box's ONLY CUDA toolkits are the default nvcc 12.0 (no
# versioned /usr/local/cuda-12.* dir at all, it's just whatever's on PATH by
# default) and a real, separately-installed /usr/local/cuda-13[.2] -- and
# CUDA 12.0's nvcc rejects the system's g++ 13.3.0 as a host compiler
# ("must be <13.0"), confirmed by an actual build failure on this box. cu130
# lets the CUDA_HOME auto-detection below correctly redirect to the newer,
# working toolkit (major version 13 matches torch's 13, avoiding the earlier
# 12-vs-13 hard mismatch), and CUDA 13.x's nvcc accepts much newer host
# compilers than 12.0 does. If this errors on a different torch/driver combo
# later, check https://pytorch.org/get-started/locally/ for the current
# index URL.
python -m pip install --quiet torch --index-url https://download.pytorch.org/whl/cu130
python -m pip install --quiet -r requirements-cluster.txt

# mamba-ssm/causal-conv1d need a CUDA extension built (or a matching
# prebuilt wheel) for your EXACT torch+CUDA+Python combo. Confirmed on this
# box: no prebuilt wheel exists yet for cu13/torch2.13/cp311 (404 on the
# release URL), and source-building failed because the DEFAULT nvcc on PATH
# is CUDA 12.0 while torch itself was built against CUDA 13.0 -- but a
# matching /usr/local/cuda-13.2 toolkit genuinely exists on this box, it's
# just not what nvcc resolves to by default. Point CUDA_HOME at it directly
# before building, matched to whatever CUDA version torch actually reports
# (don't hardcode 13.2 -- would silently go stale if torch's index serves a
# different CUDA version later).
if command -v python &>/dev/null; then
  TORCH_CUDA_VERSION="$(python -c 'import torch; print(torch.version.cuda or "")' 2>/dev/null || true)"
  if [ -n "$TORCH_CUDA_VERSION" ]; then
    TORCH_CUDA_MAJOR="${TORCH_CUDA_VERSION%%.*}"
    CUDA_CANDIDATE=""
    for c in "/usr/local/cuda-${TORCH_CUDA_VERSION}" "/usr/local/cuda-${TORCH_CUDA_MAJOR}" /usr/local/cuda-"${TORCH_CUDA_MAJOR}".*; do
      if [ -x "$c/bin/nvcc" ]; then
        CUDA_CANDIDATE="$c"
        break
      fi
    done
    if [ -n "$CUDA_CANDIDATE" ]; then
      echo "Pointing CUDA_HOME at $CUDA_CANDIDATE to match torch's CUDA $TORCH_CUDA_VERSION (default nvcc on PATH was a different version)"
      export CUDA_HOME="$CUDA_CANDIDATE"
      export PATH="$CUDA_CANDIDATE/bin:$PATH"
      export LD_LIBRARY_PATH="$CUDA_CANDIDATE/lib64:${LD_LIBRARY_PATH:-}"
    else
      echo "WARNING: no /usr/local/cuda-${TORCH_CUDA_MAJOR}.* toolkit found matching torch's CUDA $TORCH_CUDA_VERSION -- build will likely fail the same way it did before." >&2
    fi
  fi
fi

# ninja speeds up the build; packaging/psutil/setuptools/wheel are build-time
# deps mamba-ssm's/causal-conv1d's setup.py import directly -- normally pip's
# build isolation would fetch these automatically into its sandbox, but
# --no-build-isolation below means they must already be in THIS env instead.
python -m pip install --quiet ninja packaging psutil setuptools wheel || true

# ACTUAL root cause of the repeated "torch was built with CUDA 13.0" error,
# confirmed by comparing our env's real torch.version.cuda (12.8, matching
# the --index-url cu128 install above) against what the FAILED BUILD itself
# printed (torch.__version__ = ...+cu130): pip's default BUILD ISOLATION
# creates a separate, temporary environment just to run causal-conv1d's
# setup.py, and installs its OWN torch into that sandbox (whatever the
# newest default index resolves to, which is cu130) -- completely ignoring
# the cu128 torch already installed in our real env. --no-build-isolation
# makes it use our actual installed torch instead of fetching a mismatched
# one. With that fixed, the real comparison becomes our env's torch (CUDA
# 12.8) against whatever nvcc CUDA_HOME points at above (12.0, if no exact
# /usr/local/cuda-12.* toolkit was found) -- same MAJOR version (12), which
# PyTorch's version check only warns about, not a hard failure, unlike the
# major-version mismatch (12 vs 13) that broke every previous attempt.
MAMBA_BACKEND="mamba_ssm"
if ! python -m pip install --quiet --no-build-isolation "causal-conv1d>=1.4.0" \
   || ! python -m pip install --quiet --no-build-isolation "mamba-ssm>=2.2.0"; then
  echo "WARNING: mamba-ssm/causal-conv1d failed to install (see output above for the real error --" >&2
  echo "commonly a local CUDA toolkit / torch CUDA version mismatch on brand-new hardware)." >&2
  echo "Falling back to --mamba-backend pure_pytorch for this run. Retry with the real kernel" >&2
  echo "once the toolkit versions are reconciled (check nvcc --version vs torch.version.cuda)." >&2
  MAMBA_BACKEND="pure_pytorch"
fi

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

echo "Training with --mamba-backend $MAMBA_BACKEND"
python scripts/train_cluster.py \
  --mamba-backend "$MAMBA_BACKEND" \
  --device cuda \
  --wandb-project "${WANDB_PROJECT:-chisco-tsfm}" \
  "${EXTRA_ARGS[@]}"
