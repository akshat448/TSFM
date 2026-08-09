#!/usr/bin/env bash
# Build one Python 3.11 environment per model family. This is intentional:
# upstream MOMENT and TimesFM currently pin incompatible NumPy versions.
set -euo pipefail

PROJECT_ROOT="${1:?usage: scripts/setup_model_envs.sh /absolute/TSFM/root [model]}"
ONLY_MODEL="${2:-}"
ENV_ROOT="$PROJECT_ROOT/models/envs"
PYTHON_VERSION="3.11"

models=(chronos_bolt timesfm moirai_1_1_r moirai_moe moment_1_large ttm_r2 chronos_2)

for model in "${models[@]}"; do
  if [[ -n "$ONLY_MODEL" && "$ONLY_MODEL" != "$model" ]]; then
    continue
  fi
  env_dir="$ENV_ROOT/$model"
  uv venv "$env_dir" --python "$PYTHON_VERSION"
  common=(numpy pandas pyyaml torch huggingface_hub)
  case "$model" in
    chronos_bolt|chronos_2) package=("chronos-forecasting>=2.0") ;;
    timesfm) package=(timesfm) ;;
    moirai_1_1_r|moirai_moe) package=(uni2ts) ;;
    moment_1_large) package=(momentfm) ;;
    ttm_r2) package=(granite-tsfm) ;;
  esac
  uv pip install --python "$env_dir/bin/python" "${common[@]}" "${package[@]}"
  "$env_dir/bin/python" -c "import numpy, pandas, torch, yaml; print('$model environment ready')"
done
