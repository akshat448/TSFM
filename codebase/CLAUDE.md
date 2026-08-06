# CLAUDE.md (read this first)

## What this project is
One line version: we are building a contamination controlled, cross domain benchmark for Time Series Foundation Models (TSFMs), and later using what it teaches us to design a new TSFM architecture. Full context lives in `PROJECT.md` and `BENCHMARK_SPEC.md` (condensed) and in `documents/` (full originals).

## Where we are right now
This is a bridge phase. The collaborator who is providing the real compute (multi GPU cluster) and storage (~5TB) has not delivered it yet, that is still days to weeks out. In the meantime we are on a temporary allocation: one A100 80GB on the college DGX server. The plan for this phase is deliberately narrow:

**Scope right now: data only.** Download, structure, understand, and document a subset of datasets. No model downloads, no inference, no training, no fine tuning, unless the user explicitly says otherwise in a session. This keeps the eventual Phase 1/2 timeline honest and gives us something concrete to show while waiting on compute.

Update this section (in your own working notes / STATUS.md) as the phase changes. Do not assume the scope silently expands just because the GPU is idle.

## Environment, do not deviate from this
- A uv managed venv already exists at `/workspace/Venv/TSFM`, python 3.11.15.
- Activate it with `source /workspace/Venv/TSFM/bin/activate` before doing anything.
- Install packages with `uv pip install <package>`. Do not use plain pip, do not create a new venv, do not use conda.
- Already installed: `datasets`, `huggingface_hub` (via `huggingface`), `transformers`, `numpy`, `pandas`, `scikit-learn`, and `torch`/`torchvision`/`torchaudio` (cu126 build). Torch is installed but is not needed for this phase since we are not running models yet, leave it alone.
- Packages you will likely need to add for this phase: `wfdb` (reading PTB-XL waveform records), `pyarrow` (parquet output for the standardized datasets), `tqdm`, `requests`, `openpyxl`, `matplotlib`, `seaborn`, `jupyter` or `ipykernel` (for the EDA notebooks), `gluonts` (Monash/GIFT-Eval style loaders), `gdown` (fallback for Google Drive hosted files that have no other mirror). Add others only as a specific dataset needs them, and note what you added and why in `STATUS.md`.
- Repo root is `/workspace/TSFM`. This file lives at that root.

## Ground rules for this phase
1. **Data only.** Do not pull model weights or run inference/training this phase unless explicitly told to in the conversation.
2. **Check before you download.** Some datasets already exist elsewhere on this server, for example MIMIC-IV (hosp/icu tables) and WESAD are already present under `/workspace/TimeSeriesBenchMark/`, and there are more the user is still compiling a full list of. Before downloading anything, check `EXISTING_ASSETS.md` and actually look at `/workspace/TimeSeriesBenchMark/` (and ask the user if unsure). If something is already there, symlink it into `datasets/raw/` rather than re-downloading it.
3. **Headless only.** Every download must work without a browser and without interactive OAuth (this is a headless server). If the usual source for a dataset needs a browser (Google Drive share links are the classic offender), look for a HuggingFace mirror, a GitHub raw file, a direct S3/HTTP zip, or a package that wraps the download (e.g. `datasetsforecast`, `gluonts.dataset.repository`, `huggingface_hub`). If no headless path exists, stop and flag it in `STATUS.md` rather than improvising something fragile.
4. **No hardcoded paths.** Every script reads its data root from `config/paths.yaml`. This project is moving to a different, bigger server soon. The only thing that should need to change when that happens is that one YAML file. Never write `/workspace/TSFM/...` directly inside a script.
5. **Raw data is immutable.** Nothing under `datasets/raw/` gets edited, renamed, or reformatted in place. Standardization always writes new files under `datasets/processed/`, so we can always regenerate processed data from raw if the pipeline logic changes.
6. **Every dataset gets a dataset card before it counts as "done".** Source, license, exact headless download command, raw file schema, standard train/val/test split convention from the literature, standard normalization convention from the literature, 2 to 3 papers that use it and how they use it, and any gotchas. Template and current status are in `DATASETS.md`, that file is the actual place this work happens.
7. **Respect the tier boundary.** This phase covers Minimal tier plus a specific set of light Recommended tier additions, both are listed in `DATASETS.md`. Do not go pull MIMIC-IV-ECG, full GIFT-Eval, or anything from the wearables/EEG/CGM tiers "while we're at it". If it seems worth expanding scope, ask first.
8. **Log every session.** At the end of each work session, append an entry to `STATUS.md`: what got done, what's blocked, what's next, what new packages or scripts got added. This is how continuity survives across sessions and across the eventual server migration.
9. **When genuinely unsure, ask.** Which split convention to trust when two papers disagree, whether a dataset already counts as covered by an existing asset, whether a preprocessing choice matters, these are worth a quick question rather than a guess baked silently into a script.

## Reference documents, read in this order when you need more depth
1. `PROJECT.md`, condensed project charter: why this project exists, the three phase structure, what the two headline findings are supposed to be.
2. `BENCHMARK_SPEC.md`, condensed domain/task/dataset/model selection and evaluation protocol, this is what Phase 2 will hold everything to, so today's data structuring should already be compatible with it.
3. `DATASETS.md`, the actual working tracker for this phase. Start here for anything dataset specific.
4. `STEPS.md`, the ordered checklist for this phase.
5. `documents/`, the full original planning documents. The condensed files above simplify things for quick reference, if something seems inconsistent or underspecified, this folder is the source of truth.

## Known open items, do not resolve these silently
- **MIMIC-IV-ECG vs MIMIC-IV clinical.** The benchmark's ECG domain needs MIMIC-IV-ECG, the waveform database (12 lead ECGs at 500Hz, ~800k records). What is already on the server (`/workspace/TimeSeriesBenchMark/.../mimiciv/3.1/hosp/`) is MIMIC-IV clinical, the structured EHR tables (admissions, diagnoses, labs, prescriptions, etc). These are two different PhysioNet products that happen to share a parent name. The clinical tables are a genuinely useful asset for the project's Phase 3 "EHR" future direction, but they do not substitute for MIMIC-IV-ECG. If MIMIC-IV-ECG is needed for the ECG domain FM comparison, that is a separate, not yet started, credentialing process.
- **Incomplete asset inventory.** WESAD confirmed. The `/workspace/` lab directories one level up from `TimeSeriesBenchMark/` (`Emotiv`, `Brain`, `SensorLLM`, `GlycoLM`, `TSZechen`) were inventoried 2026-07-08 — all EEG/HAR/CGM research, none overlap this project's scope, see `EXISTING_ASSETS.md`. The "engauge" name mentioned early on was searched for explicitly across those five and not found; it remains genuinely unresolved, not a search-thoroughness gap — if it refers to something else, that needs the user's direct input. Treat anything in `EXISTING_ASSETS.md` marked "unconfirmed" as provisional until the user confirms it.
