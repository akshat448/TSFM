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
WORKER="$REPO_ROOT/scripts/_train_worker.sh"

# Build the exact command as a single string via a real launcher SCRIPT FILE,
# not an exported bash function or inline nested-quoted string. Both of
# those were tried and broke on the real target machine (tmux/nohup add
# extra shell layers that don't reliably propagate exported functions, and
# nested quoting across bash -c boundaries is its own hazard) -- a plain
# file that tmux/nohup just execute has neither problem.
LAUNCHER_SCRIPT="logs/_launcher_$$.sh"
{
  echo "#!/usr/bin/env bash"
  printf '%q ' bash "$WORKER" "$GPU" "${EXTRA_ARGS[@]}"
  printf '> >(tee %q) 2>&1\n' "$LOG_FILE"
  echo 'ec=${PIPESTATUS[0]}'
  echo 'echo "=== run_training exited with code $ec (session stays open -- Ctrl-b & to kill it) ==="'
} > "$LAUNCHER_SCRIPT"
chmod +x "$LAUNCHER_SCRIPT"

launch_with_nohup() {
  echo "Falling back to nohup (less robust to tunnel drops than tmux -- if the SSH/VSCode"
  echo "tunnel itself dies, not just your terminal, nohup survives but you lose the ability"
  echo "to re-attach and watch live; tail -f the log file instead)."
  nohup bash "$LAUNCHER_SCRIPT" > /dev/null 2>&1 &
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
    echo "tmux session '$SESSION' already exists -- killing it (and whatever is running in it,"
    echo "including an in-progress training run) before starting a new one. To re-attach to an"
    echo "existing run instead of replacing it, use: TMUX_TMPDIR=$TMUX_TMPDIR tmux attach -t $SESSION"
    tmux kill-session -t "$SESSION"
  fi
  if tmux new-session -d -s "$SESSION" "bash '$LAUNCHER_SCRIPT'" 2>/tmp/tmux_launch_err_$$; then
    # Only reachable once new-session has actually started the server, so
    # set-option now has something to target. Doing this BEFORE new-session
    # (as an earlier version of this script did) fails outright on some tmux
    # builds: unlike new-session, set-option does not auto-start a server,
    # so it errors "no server running" and -- combined with `set -e` --
    # kills the whole script before ever reaching new-session or the nohup
    # fallback. Confirmed as the actual failure mode on the real target box.
    #
    # Without remain-on-exit, tmux kills the session the INSTANT
    # run_training exits -- success or crash. A fast crash (bad venv, pip
    # failure, CUDA check failure) then means `tmux attach` immediately
    # after launch shows "no sessions", discarding the one place you'd see
    # why. This keeps the pane open after exit so you can actually attach
    # and read it; the exit-code echo baked into $LAUNCHER_SCRIPT is a
    # second belt-and-suspenders signal that also lands in the log file for
    # cases you never attach at all.
    tmux set-option -t "$SESSION" remain-on-exit on
    echo "Launched in tmux session '$SESSION'."
    echo "Attach with: TMUX_TMPDIR=$TMUX_TMPDIR tmux attach -t $SESSION"
    echo "(add 'export TMUX_TMPDIR=$TMUX_TMPDIR' to your shell rc so plain 'tmux attach' works too)"
    echo "Log file: $LOG_FILE"
    echo "If the session is gone/empty when you attach, it crashed instantly -- read the log file."
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
