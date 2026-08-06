# AGENTS.md (read this first — Codex reads this file automatically)

This is the sibling of `CLAUDE.md` from the bridge phase, updated for the new
server and the new phase. If you (the coding agent) are Codex, this file is
your entry point. If a `CLAUDE.md` also exists in this repo, it should be
identical or symlinked to this one — don't let them drift.

## What this project is

One line: we are building a contamination-controlled, cross-domain benchmark
for Time Series Foundation Models (TSFMs), and later using what it teaches us
to design a new TSFM architecture. Full context: `PROJECT.md`,
`BENCHMARK_SPEC.md`, `PHASE1_PLAN.md`, and `documents/` (full originals).

## Where we are right now (updated 2026-07-30)

The bridge phase (data-only, single borrowed A100) is **closed out**. All 14
Minimal + light-Recommended datasets are downloaded, standardized, EDA'd, and
literature-confirmed — see `DATASETS.md` and the final `STATUS.md` entry
dated 2026-07-08 for the full findings (three-way solar duplicate, PEMS
split correction, Meyer et al. 2025 corroboration, etc). Read that entry
before touching anything, it's load-bearing context.

We have now moved to a new server with real (if currently oversubscribed)
compute: **8x NVIDIA RTX PRO 6000 Blackwell, 97GB VRAM each**, CUDA 13.2.
This is a large upgrade from the single A100 80GB the bridge phase used.
**GPU time is scheduled, not standing.** Check `NEXT_PHASE_TASKS.md` for what
is in-scope right now (no GPU needed) versus what waits for the next
scheduled GPU slot. Do not run training or inference jobs against a GPU
without confirming a slot is active — other users' jobs are visible on this
box (`nvidia-smi`) and are not ours to preempt.

**Scope right now:** dataset expansion to full Recommended tier + harness
hardening + model checkpoint pre-staging. This is genuinely more than the
bridge phase's "data only" rule — see `NEXT_PHASE_TASKS.md` bucket A for the
exact list of what's newly in scope (MIMIC-IV-ECG credentialing, PPG-DaLiA,
full GIFT-Eval, expanded Monash, all model checkpoint downloads). Nothing in
bucket B (actual GPU runs) starts without an explicit go-ahead in the
conversation, same discipline as the bridge phase's rule 1.

## Directory layout (new server, do not deviate)

```
/mnt/hdd1/TSFM/
├── codebase/            <- this repo. scripts, config, harness, notebooks, literature notes.
├── dataset/             <- ALL data lives here now (raw + processed). Renamed/moved from
│                           the bridge phase's `datasets/` subfolder-of-repo layout — data is
│                           no longer nested inside the code repo, it's a sibling directory.
└── models/              <- not created yet. Will hold model checkpoints/weights once
                             the first checkpoint download task runs. Keep it a sibling of
                             codebase/ and dataset/, same reasoning: large binary artifacts
                             don't belong inside a git-tracked code repo.
```

This is a deliberate change from the bridge phase, where everything sat under
one `/workspace/TSFM/` tree with `datasets/` nested inside the repo. See
`PATHS_AND_MIGRATION.md` for the migration steps and the new
`config/paths.yaml`.

## Environment, do not deviate from this

- Do **not** assume the bridge phase's venv path (`/workspace/Venv/TSFM`)
  exists here — it doesn't, this is a different machine. Create a fresh `uv`
  managed venv on this server; see `PATHS_AND_MIGRATION.md` step 3 for the
  exact command and the path convention to use (`/mnt/hdd1/TSFM/venv`, a
  sibling of `codebase/`, `dataset/`, `models/`).
- Install packages with `uv pip install`. Do not use plain `pip`, do not use
  conda.
- Re-verify Python version and CUDA/torch compatibility on this box before
  assuming anything — CUDA 13.2 / driver 595.58.03 is newer than what the
  bridge phase's torch build (cu126) targeted. Check whether a newer torch
  wheel is needed once GPU work actually starts; CPU-only work (data,
  checkpoint downloads, adapter smoke tests) doesn't need this resolved yet.
- Packages needed, cumulative from both phases: `datasets`, `huggingface_hub`,
  `transformers`, `numpy`, `pandas`, `scikit-learn`, `torch`/`torchvision`/
  `torchaudio`, `wfdb`, `pyarrow`, `tqdm`, `requests`, `openpyxl`,
  `matplotlib`, `seaborn`, `jupyter`/`ipykernel`, `gluonts`, `gdown`,
  `pyyaml`. New this phase, only once actually wiring in each model:
  `chronos-forecasting`, `timesfm`, `granite-tsfm` (TTM-r2), and whatever
  Wave 3/4 need (`neuralforecast` or similar for PatchTST/iTransformer/
  S-Mamba if not hand-rolled, `statsforecast` for AutoARIMA). Note what you
  add and why in `STATUS.md`, same discipline as before.

## Ground rules for this phase

1. **No GPU runs without an explicit go-ahead.** Downloading data, downloading
   model checkpoints, writing/testing adapter code on CPU, and expanding the
   harness are all in scope right now. Actually running inference/training
   is not, until a scheduled GPU slot is confirmed active in the
   conversation. This is the direct analog of the bridge phase's "data only"
   rule, updated for what's actually available now.
2. **Check `nvidia-smi` before assuming a GPU is free**, even during a
   scheduled slot — this is a shared box, other users' jobs show up in the
   output. If your slot's GPU still shows another process using most of its
   memory, stop and flag it rather than launching alongside it.
3. **No hardcoded paths.** Every script reads `data_root` /
   `checkpoint_root` from `config/paths.yaml`, same rule as before, now with
   a `checkpoint_root` key added for `models/`. Never write
   `/mnt/hdd1/TSFM/...` directly inside a script.
4. **Raw data is immutable, and now raw data isn't even in the code repo.**
   Nothing under `dataset/raw/` gets edited in place. Standardization writes
   new files under `dataset/processed/`. This is unchanged from the bridge
   phase, just re-stated because the path moved.
5. **Every new dataset gets a dataset card before it counts as "done."** Same
   template as before, `DATASETS.md`'s bottom section. Applies to every
   Recommended-tier addition this phase.
6. **Every new model adapter gets a smoke test before it's wired into
   `phase1_eval.yaml`.** The harness's Wave 2 adapters
   (`chronos_bolt.py`, `timesfm.py`, `ttm.py`) are scaffolding, written
   against documented APIs but never run against real weights. Before
   trusting one: run its `__main__` smoke test, on CPU if the model is small
   enough, otherwise flag it for the first GPU slot rather than guessing.
7. **Log every session.** Append to `STATUS.md`, same convention as before —
   what got done, what's blocked, what's next, what new packages/scripts got
   added.
8. **When genuinely unsure, ask.** Same as before. In particular: don't
   silently decide a GPU slot is "close enough" to launch early, don't
   silently expand which model families get downloaded beyond what
   `NEXT_PHASE_TASKS.md` lists, don't silently start MIMIC-IV-ECG
   credentialing paperwork that needs the user's identity/institution info.

## Reference documents, read in this order when you need more depth

1. `PROJECT.md` — condensed project charter.
2. `BENCHMARK_SPEC.md` — condensed domain/task/dataset/model selection and
   evaluation protocol. This is what every run has to be compatible with.
3. `DATASETS.md` — the dataset tracker. All 14 bridge-phase datasets are
   `[x]`. New Recommended/Full-tier additions get added here as new rows.
4. `PHASE1_PLAN.md` — the harness architecture and model wave sequencing.
   Written for single-GPU; re-read the "given single-GPU reality" framing in
   light of the new 8x97GB box, but don't unilaterally reorder the waves —
   flag the opportunity (e.g. running Wave 2+3 in parallel across GPUs) and
   confirm before changing the plan.
5. `NEXT_PHASE_TASKS.md` — the ordered checklist for *this* phase specifically.
6. `PATHS_AND_MIGRATION.md` — the one-time migration steps, read once, then
   ignore unless something about paths seems broken.
7. `STATUS.md` — read the most recent entries first, especially the
   2026-07-08 phase-wrap-up entry, before doing anything else. It contains
   findings (contamination, data quality issues, portability audit results)
   that materially affect what you should and shouldn't trust going forward.
8. `documents/` — full original planning documents, source of truth if
   anything above seems inconsistent or underspecified.

## Known open items, do not resolve these silently

Carried over from the bridge phase, still unresolved:

- **MIMIC-IV-ECG vs MIMIC-IV clinical.** Clinical tables (hosp/icu) are
  confirmed present (bridge phase, `EXISTING_ASSETS.md`). MIMIC-IV-ECG
  (waveform DB) is a separate PhysioNet product needing its own
  credentialing, not started. This phase's task list includes *starting*
  that credentialing process (paperwork/registration), not downloading the
  data itself yet — see `NEXT_PHASE_TASKS.md`.
- **"engauge."** Searched explicitly across five lab directories in the
  bridge phase, not found. Still unresolved. Do not guess at it again; if it
  matters, it needs the user's direct input.
- **New this phase: does the new server (`/mnt/hdd1`) have any existing
  TSFM-relevant assets already on it**, the way the old DGX server had
  MIMIC-IV clinical and WESAD? Nobody has inventoried this new box yet. This
  is the first thing to check before any fresh download — see
  `NEXT_PHASE_TASKS.md` task A0.