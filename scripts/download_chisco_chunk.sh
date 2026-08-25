#!/usr/bin/env bash
# Downloads a single small run (~44 MB, 28 trials, subject 01, imagined-speech
# task) from the real CHISCO dataset on OpenNeuro (ds005170), for local
# smoke-testing of chisco_pipeline without pulling the full ~multi-GB corpus.
#
# OpenNeuro datasets are mirrored on a public, unauthenticated S3 bucket, so
# no account/token is needed -- just the AWS CLI with --no-sign-request.
#
# Picked run-045 specifically because `aws s3 ls --recursive` on
# derivatives/preprocessed_pkl/sub-01/eeg/ showed it as the smallest file in
# that subject's run list (46,230,524 bytes vs. ~200-300MB for most others).
#
# To grab more data later (bigger local runs, other subjects, or the "read"
# task variant instead of "imagine"), list the bucket first:
#   aws s3 ls --no-sign-request s3://openneuro.org/ds005170/derivatives/preprocessed_pkl/sub-02/eeg/
# and cp additional files the same way.

set -euo pipefail

REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
DEST_DIR="${REPO_ROOT}/data/derivatives/preprocessed_pkl/sub-01/eeg"
mkdir -p "$DEST_DIR"

aws s3 cp --no-sign-request \
  s3://openneuro.org/ds005170/derivatives/preprocessed_pkl/sub-01/eeg/sub-01_task-imagine_run-045_eeg.pkl \
  "$DEST_DIR/"

echo "Downloaded to: $DEST_DIR/sub-01_task-imagine_run-045_eeg.pkl"
