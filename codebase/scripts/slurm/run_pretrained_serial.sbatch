#!/usr/bin/env bash
# One shared-GPU request. Do not add an array or raise --gpus above one.
# Fill partition/account/QoS through `run_pretrained_models.py submit` on the server.
# This checked-in template is the portable fallback.
#SBATCH --job-name=tsfm-pretrained
#SBATCH --gpus=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=64G
#SBATCH --time=24:00:00
#SBATCH --output=logs/pretrained-%j.out
#SBATCH --error=logs/pretrained-%j.err

set -euo pipefail
PROJECT_ROOT="${PROJECT_ROOT:?set PROJECT_ROOT before submitting}"
MODEL_ENV_ROOT="${MODEL_ENV_ROOT:-$PROJECT_ROOT/models/envs}"
export HF_HOME="${HF_HOME:-$PROJECT_ROOT/models/cache/huggingface}"
mkdir -p "$PROJECT_ROOT/codebase/logs"
cd "$PROJECT_ROOT/codebase"
nvidia-smi

for model in chronos_bolt timesfm moirai_1_1_r moirai_moe moment_1_large ttm_r2 chronos_2; do
  source "$MODEL_ENV_ROOT/$model/bin/activate"
  python run_pretrained_models.py run --model "$model" --device cuda --resume
  deactivate
done
