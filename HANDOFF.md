# CHISCO EEG-to-text pipeline — handoff / continue-later notes

Repo: `git@github.com:akshat448/TSFM.git`, branch **`chisco-pipeline`** (not `main` — `main` untouched).
Target machine: shared bare GPU box (no SLURM, confirmed — `squeue`/`scontrol` fail with DNS SRV errors), reached via AnyDesk → VSCode tunnel → SSH. 8x RTX PRO 6000 Blackwell (97GB each), shared with other users' jobs. Repo lives at `/mnt/hdd1/TSFM/chisco-tsfm` on that box.

## What this is

A from-scratch PyTorch pipeline decoding CHISCO (OpenNeuro `ds005170`, imagined-speech EEG, NOT RSVP — corrected from an earlier wrong assumption) into semantic text embeddings: SincNet front-end → dynamic spatial whitening (EOG-based, no EMG channel exists in real CHISCO) → causal Mamba SSM encoder → DINO-style momentum-teacher self-supervised pretraining with a heteroscedastic-NLL loss → linear-probe sanity gate → cross-attention text-alignment/retrieval head.

Every architectural claim in the original spec was checked against real data (not assumed) — see docstrings in `chisco_pipeline/*.py` for the corrections made (schema, channel counts, sample rate, etc.).

## Key files

- `chisco_pipeline/` — the model/pipeline code (frontend, encoder, losses, eval_probe, text_align, dataloader, raw_dataloader).
- `scripts/download_chisco_chunk.sh` — one small derivative run, for local smoke tests.
- `scripts/download_chisco_full.sh` — full derivative dataset (preprocessed pkl, PREP/ICA/autoreject already applied by CHISCO authors).
- `scripts/download_chisco_raw.sh` — raw `.edf` files (400-600MB each, defaults to 5 runs/subject, not all — pass `--runs all`/`--max-runs N` for more).
- `chisco_pipeline/raw_dataloader.py` (`CHISCORawDataloader`) — loads raw `.edf` directly, skips PREP/ICA/autoreject/highpass entirely per explicit request, but still does event-detection + epoching (unavoidable — that's how trials get extracted from a continuous recording at all, not "preprocessing" in the artifact-cleaning sense).
- `scripts/train_local.py`, `scripts/train_read_imagine_probe.py` — small CPU smoke tests, already validated against real data (see below).
- `scripts/train_cluster.py` — the real training run: proper minibatching, wandb logging, real CUDA `mamba_ssm` kernel (optional), real Chinese T5 text encoder. Supports `--eeg-source {derivative,raw}`.
- `scripts/launch_training.sh` + `scripts/_train_worker.sh` — the actual launcher for the target box (tmux-based, no SLURM). Auto-picks the least-busy GPU (`--gpu auto`, override with `--gpu N`), auto-kills a stale `chisco-train` tmux session before relaunching.
- `.env.sample` → copy to `.env`, set `WANDB_API_KEY`.

## Validated results (real data, not synthetic)

- **Read-vs-imagine binary probe**, subj-01 run-045: 68.8–81.2% held-out accuracy across 4 seeds vs. 50% chance (pooled 75.0%, p≈3.9e-5). **Confirmed real via label-shuffle permutation control**: shuffled labels collapsed to ~39% (chance/below), proving it wasn't a leakage bug. Caveat: likely substantially explained by a large low-level EOG/amplitude artifact difference between overt reading and imagined speech (naive single-feature threshold alone gets 82.8%), not deep semantic decoding — this was always meant as an easy sanity gate, not a claim about sentence-level decoding.
- **Sentence-identity probe** (28-trial single run): statistically underpowered (chance 3.6%, 5 held-out samples) — inconclusive by design, not a negative result.
- `train_cluster.py --eeg-source raw` validated end-to-end against one real downloaded 568MB run (200 real trials, correct shapes, no crashes) before pushing.

## Cluster setup — hard-won fixes already in place

Long back-and-forth getting `launch_training.sh` to actually run on the real box. In order, what broke and what fixed it (all already pushed, don't re-diagnose these from scratch):

1. **tmux function-export bug**: `export -f` across tmux's shell invocation silently failed. Fixed by moving worker logic into a real file (`_train_worker.sh`) invoked via a generated launcher script instead of an exported bash function.
2. **tmux socket dir broken** (`/tmp/tmux-1000 is not a directory` on the shared box): fixed via private `TMUX_TMPDIR=$HOME/.tmux-sockets`.
3. **`tmux set-option -g` before any session existed**: doesn't auto-start a server on this tmux build, killed the script under `set -e`. Fixed by creating the session first, then setting `remain-on-exit` on it.
4. **`python3 -m venv` broken** (`ensurepip` not available, no sudo): switched to conda, but `command -v conda` fails in a non-interactive shell (conda init only wires PATH via `~/.bashrc`, not sourced by tmux/nohup) — fixed by searching common install paths (`~/miniconda3` etc.) for `conda.sh` directly.
5. **`aws` CLI missing**: auto-installed via pip on demand.
6. **`mamba-ssm`/`causal-conv1d` build failures** (this took several rounds):
   - pip's build isolation was silently building against a *different* torch (cu130) than the one actually installed (cu128) — fixed with `--no-build-isolation`.
   - g++ 13.3.0 too new for CUDA 12.0's nvcc (needs `<13.0`) — switched torch to `cu130` so `CUDA_HOME` auto-detection redirects to the box's separately-installed `/usr/local/cuda-13.x`, which accepts newer host compilers.
   - **conda env persists across runs**, and `pip install torch` (no version pin) is a no-op if any torch is already installed — kept reinstalling stale cu128 even after the cu130 fix was pushed. Fixed with `--upgrade --force-reinstall` (latest commit, `9d3d3f3` — **not yet confirmed working on the real box**, this is where things left off).
   - If `mamba-ssm` build still fails after all that: it's non-fatal by design, falls back to `--mamba-backend pure_pytorch` automatically (confirmed working — base torch+Blackwell verified via a real CUDA matmul kernel launch).

## Where things left off / next steps

- Last pushed commit: `9d3d3f3` ("Fix stale torch: --upgrade --force-reinstall"). **Not yet confirmed** whether this actually gets `mamba-ssm` building on the real box — that was the last thing in flight.
- To resume: `git pull origin chisco-pipeline`, confirm `git log --oneline -1` shows `9d3d3f3` (or later), rerun `./scripts/launch_training.sh`.
- If it still fails on mamba-ssm: that's OK, training still works via the `pure_pytorch` fallback — not a hard blocker, just slower (no fused CUDA kernel). Worth deciding whether it's worth continuing to chase the CUDA-kernel build vs. just training on `pure_pytorch` for now.
- Raw-EDF path (`--eeg-source raw`) is built and validated on real data but has NOT yet been run at real scale on the target box (only tested locally on this dev machine against one downloaded run).
- No queueing system was found on the target box (checked `qsub`/`condor_q`/`kubectl`/`bsub`, none present; SLURM confirmed absent) — `launch_training.sh`'s GPU auto-pick is a snapshot heuristic, not a real reservation, since there's no scheduler to reserve through. Worth asking the colleague again what they actually use, if anything.
- No probe/retrieval result yet reflects a real, adequately-powered training run — everything so far is either a toy smoke test or the deliberately-easy read-vs-imagine sanity gate.
