#!/usr/bin/env bash
# Downloads CHISCO derivative pickles (OpenNeuro ds005170) at whatever scale
# you ask for -- from a single subject/task up to the full released dataset.
# Same public, unauthenticated S3 mirror used by download_chisco_chunk.sh, so
# no OpenNeuro account/token is needed, including on a compute cluster with
# no interactive login.
#
# Usage:
#   ./scripts/download_chisco_full.sh [--subjects 01,02,03,04,05] [--tasks imagine,read] [--dest DIR]
#
# Defaults to ALL 5 subjects, BOTH tasks -- this is the full derivative
# dataset (multiple GB; each subject's imagine+read pkls run into the tens
# of GB combined). Narrow with --subjects/--tasks for a partial pull.
#
# Uses `aws s3 sync` (not `cp --recursive`) so re-running after a partial/
# interrupted download only fetches what's missing -- safe to re-run on a
# flaky cluster network without re-downloading everything.

set -euo pipefail

REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
DEST_DIR="${REPO_ROOT}/data/derivatives/preprocessed_pkl"
SUBJECTS="01,02,03,04,05"
TASKS="imagine,read"

while [[ $# -gt 0 ]]; do
  case "$1" in
    --subjects) SUBJECTS="$2"; shift 2 ;;
    --tasks) TASKS="$2"; shift 2 ;;
    --dest) DEST_DIR="$2"; shift 2 ;;
    *) echo "Unknown arg: $1" >&2; exit 1 ;;
  esac
done

IFS=',' read -ra SUBJECT_LIST <<< "$SUBJECTS"
IFS=',' read -ra TASK_LIST <<< "$TASKS"

echo "Downloading CHISCO (ds005170) derivatives -> $DEST_DIR"
echo "Subjects: ${SUBJECT_LIST[*]}"
echo "Tasks: ${TASK_LIST[*]}"

for sub in "${SUBJECT_LIST[@]}"; do
  sub_dir="${DEST_DIR}/sub-${sub}/eeg"
  mkdir -p "$sub_dir"
  for task in "${TASK_LIST[@]}"; do
    echo "== sub-${sub}, task-${task} =="
    aws s3 sync --no-sign-request \
      "s3://openneuro.org/ds005170/derivatives/preprocessed_pkl/sub-${sub}/eeg/" \
      "$sub_dir/" \
      --exclude "*" \
      --include "sub-${sub}_task-${task}_run-*_eeg.pkl"
  done
done

echo "Done. Downloaded files:"
find "$DEST_DIR" -name '*.pkl' | wc -l
