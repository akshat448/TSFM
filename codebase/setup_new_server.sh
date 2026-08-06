#!/usr/bin/env bash
# setup_new_server.sh
# Run once on the new server (/mnt/hdd1/TSFM). Scaffolds the directory tree.
# Does NOT copy data from the old server -- see PATHS_AND_MIGRATION.md step 2
# for the rsync commands, run those separately (they need the old server's
# hostname, which isn't hardcoded here on purpose).

set -euo pipefail

ROOT="/mnt/hdd1/TSFM"

echo "[1/5] codebase/ subtree"
mkdir -p "$ROOT/codebase"/{config,scripts/download,scripts/standardize,scripts/harness/models,notebooks/eda,literature/notes,documents,results/phase1}

echo "[2/5] dataset/ subtree"
mkdir -p "$ROOT/dataset/raw"/{energy,traffic,weather,ecg,multi_domain,wearables}
mkdir -p "$ROOT/dataset/processed"/{energy,traffic,weather,ecg,multi_domain,wearables}
mkdir -p "$ROOT/dataset/manifests"

echo "[3/5] models/ subtree (empty until the first checkpoint download task)"
mkdir -p "$ROOT/models/checkpoints"
mkdir -p "$ROOT/models/cache/huggingface"

echo "[4/5] venv"
if [ ! -d "$ROOT/venv" ]; then
  command -v uv >/dev/null 2>&1 || { echo "uv not found on PATH, install it first"; exit 1; }
  uv venv "$ROOT/venv" --python 3.11
  echo "  created $ROOT/venv -- activate with: source $ROOT/venv/bin/activate"
else
  echo "  $ROOT/venv already exists, skipping"
fi

echo "[5/5] disk space check"
df -h "$ROOT"

cat <<EOF

Scaffolding done. Next steps (see PATHS_AND_MIGRATION.md for full detail):
  1. rsync the old server's codebase + dataset/raw + dataset/processed over.
  2. source $ROOT/venv/bin/activate
  3. uv pip install -r <requirements list in AGENTS.md / PATHS_AND_MIGRATION.md step 3>
  4. cp config_paths.yaml $ROOT/codebase/config/paths.yaml
  5. Run the validation block in PATHS_AND_MIGRATION.md step 5.
EOF