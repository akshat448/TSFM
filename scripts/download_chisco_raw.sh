#!/usr/bin/env bash
# Downloads RAW (unprocessed) CHISCO .edf files, plus the textdataset word
# lists needed to attach text labels to trials -- for use with
# chisco_pipeline.raw_dataloader.CHISCORawDataloader, which epochs directly
# from these .edf files without the authors' PREP/ICA/autoreject/1Hz-highpass
# pipeline (see raw_dataloader.py for exactly what is and isn't done).
#
# SIZE WARNING: raw .edf files are much bigger than the preprocessed
# derivative pkls used elsewhere in this repo -- 400-600 MB EACH, and each
# subject has 40+ runs. A full subject is commonly 15-25 GB; all 5 subjects
# is on the order of 100+ GB. This script does NOT default to "all runs" for
# that reason -- pass --runs explicitly or accept the small default.
#
# Usage:
#   ./scripts/download_chisco_raw.sh --subjects 01 --runs 1,2,3
#   ./scripts/download_chisco_raw.sh --subjects 01,02 --max-runs 5   # first N runs per subject
#   ./scripts/download_chisco_raw.sh --subjects 01 --runs all        # explicit opt-in to everything

set -euo pipefail

REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
DEST_DIR="${REPO_ROOT}/data/raw"
SUBJECTS="01"
RUNS=""
MAX_RUNS=5

while [[ $# -gt 0 ]]; do
  case "$1" in
    --subjects) SUBJECTS="$2"; shift 2 ;;
    --runs) RUNS="$2"; shift 2 ;;
    --max-runs) MAX_RUNS="$2"; shift 2 ;;
    --dest) DEST_DIR="$2"; shift 2 ;;
    *) echo "Unknown arg: $1" >&2; exit 1 ;;
  esac
done

if ! command -v aws &>/dev/null; then
  echo "aws CLI not found -- installing via pip..."
  if [ -n "${VIRTUAL_ENV:-}" ] || [ -n "${CONDA_DEFAULT_ENV:-}" ]; then
    python3 -m pip install --quiet awscli
  else
    python3 -m pip install --quiet --user awscli
    export PATH="$HOME/.local/bin:$PATH"
  fi
fi

IFS=',' read -ra SUBJECT_LIST <<< "$SUBJECTS"

mkdir -p "$DEST_DIR"

# textdataset (word lists) -- small (~10KB/file, ~45 files), always grab all of it.
echo "== textdataset (word labels) =="
mkdir -p "$DEST_DIR/textdataset"
aws s3 sync --no-sign-request \
  "s3://openneuro.org/ds005170/textdataset/" \
  "$DEST_DIR/textdataset/"

for sub in "${SUBJECT_LIST[@]}"; do
  echo "== Listing runs for sub-${sub} =="
  ALL_KEYS=$(aws s3 ls --no-sign-request "s3://openneuro.org/ds005170/sub-${sub}/" --recursive \
    | awk '{print $4}' | grep '_eeg\.edf$')

  if [[ "$RUNS" == "all" ]]; then
    SELECTED_KEYS="$ALL_KEYS"
  elif [[ -n "$RUNS" ]]; then
    SELECTED_KEYS=""
    IFS=',' read -ra RUN_LIST <<< "$RUNS"
    for r in "${RUN_LIST[@]}"; do
      match=$(echo "$ALL_KEYS" | grep -E "run-0*${r}_eeg\.edf$" || true)
      SELECTED_KEYS="${SELECTED_KEYS}${match}"$'\n'
    done
  else
    SELECTED_KEYS=$(echo "$ALL_KEYS" | sort | head -n "$MAX_RUNS")
  fi

  N=$(echo "$SELECTED_KEYS" | grep -c '.edf$' || true)
  echo "sub-${sub}: downloading ${N} run(s) (~$((N * 500))MB estimated)"

  while IFS= read -r key; do
    [[ -z "$key" ]] && continue
    local_path="$DEST_DIR/${key#ds005170/}"
    mkdir -p "$(dirname "$local_path")"
    if [ -f "$local_path" ]; then
      echo "already have: $key"
      continue
    fi
    echo "downloading: $key"
    aws s3 cp --no-sign-request "s3://openneuro.org/${key}" "$local_path"
  done <<< "$SELECTED_KEYS"
done

echo "Done. Raw files: $(find "$DEST_DIR" -name '*.edf' | wc -l), textdataset files: $(find "$DEST_DIR/textdataset" -name '*.xlsx' | wc -l)"
