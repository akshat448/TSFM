# STEPS.md (ordered checklist for this phase)

Work through these roughly in order. Each step should end with something committed to disk (a file, a note, an entry in `DATASETS.md` or `STATUS.md`), not just something run once and discarded. Check off as you go, and append a summary to `STATUS.md` at the end of every session regardless of how far you got.

## Step 0: environment and repo scaffolding
- [x] Confirm the venv works: `source /workspace/Venv/TSFM/bin/activate`, `python --version` should show 3.11.15. (re-verified 2026-07-07)
- [x] Add the packages listed in `CLAUDE.md` under Environment that aren't already installed (`wfdb`, `pyarrow`, `tqdm`, `requests`, `openpyxl`, `matplotlib`, `seaborn`, `jupyter` or `ipykernel`, `gluonts`, `gdown`), using `uv pip install`.
- [x] Create the working directory structure under `/workspace/TSFM/`:
  ```
  config/
  datasets/raw/{energy,traffic,weather,ecg,multi_domain}/
  datasets/processed/{energy,traffic,weather,ecg,multi_domain}/
  datasets/manifests/
  notebooks/eda/
  literature/notes/
  scripts/download/
  scripts/standardize/
  ```
- [x] Write `config/paths.yaml` with at minimum: `data_root`, `raw_dir`, `processed_dir`, `manifest_dir`, `venv_path`. Every download/standardize script reads from this file, nothing hardcodes `/workspace/TSFM/...` directly. (re-verified 2026-07-07: parses cleanly, all required keys present)

## Step 1: inventory existing assets before downloading anything
- [x] Explore `/workspace/TimeSeriesBenchMark/` (and any other location the user points to) and write down what's actually there: exact paths, file types, approximate sizes, what dataset each thing corresponds to.
- [x] Fill in `EXISTING_ASSETS.md` with this inventory. Cross-reference against the "Deferred / already present" table in `DATASETS.md`.
- [x] Explicitly confirm whether the MIMIC-IV present is clinical tables (hosp/icu) or the ECG waveform database, don't assume, check the actual file listing (`.csv.gz` tabular files means clinical, WFDB `.dat`/`.hea` pairs would mean the waveform ECG product).
- [x] For WESAD and anything else confirmed present, symlink it into `datasets/raw/<domain>/<name>/` rather than copying or re-downloading, and note in the dataset's status that it was inventoried, not downloaded fresh. (re-verified 2026-07-07: `datasets/raw/wearables/wesad` is a real symlink to `/workspace/TimeSeriesBenchMark/datasets/WESAD`)
- [ ] If the user has since shared the fuller list of existing assets, revisit this step before continuing, it directly changes what Step 2 needs to actually download. **Not checked: still open** — the "one more existing dataset, name unconfirmed" item in `CLAUDE.md`/`EXISTING_ASSETS.md` has not been resolved by the user as of 2026-07-07, so this step's precondition hasn't been met yet, not something we can mark done.

## Step 2: download the Phase-scope datasets
Work through `DATASETS.md` sections 1 through 14 in whatever order is easiest, but energy and weather datasets first (they're the smallest and most reliable, good for validating the download/standardize pipeline before tackling PTB-XL and the multi-domain corpora).
- [x] For each dataset: run the headless download command from `DATASETS.md`, verify the file count/size roughly matches what's expected, write a short download log entry to `datasets/manifests/<name>.log` (command used, timestamp, size, any warnings). (re-verified 2026-07-07: 9 manifest logs on disk covering all 14 datasets — `ETT.log` covers 4, `pems.log` covers 3)
- [x] If a dataset's primary headless route fails (this is most likely for PEMS03/04/08), try the fallback listed, and if that also fails, mark it `[!]` in `DATASETS.md`, write what was tried in `STATUS.md`, and move on rather than burning the session on one dataset. (PEMS03/04/08 hit exactly this case; `gdown` fallback against the official Drive bundle succeeded, no `[!]` needed)
- [x] Update the status column in `DATASETS.md` as each dataset lands.

## Step 3: standardize into a unified format
For each downloaded dataset:
- [x] Write a small standardize script under `scripts/standardize/` that reads from `datasets/raw/...` and writes to `datasets/processed/...` in a consistent shape: long or wide format (pick one convention project-wide and note it in the script), explicit timestamp column, explicit split column (`train`/`val`/`test`) following the standard convention documented for that dataset in `DATASETS.md`, and parquet as the output format (not CSV, for the multivariate ones especially, `pyarrow`). (re-verified 2026-07-07: 14 parquet outputs on disk, schema spot-checked on ETTh1/electricity/ptbxl/gift_eval/monash)
- [x] For PTB-XL specifically: carry `strat_fold` through untouched as the split indicator, do not re-derive folds. (re-verified: `strat_fold` column present untouched, plus derived `split` convenience column)
- [x] For anything physiological: double check no patient/subject ID appears in more than one split after standardization, this is worth an explicit assertion in the script, not just a visual check. (re-verified 2026-07-07: real `assert` statements in `standardize_ptbxl.py`, not just a visual/eyeballed check)
- [x] Do not compute or bake in normalization statistics at this stage, that's a modeling-time decision for Phase 2 (train-only stats, per `BENCHMARK_SPEC.md`), just get clean, split-labeled, unnormalized data into `datasets/processed/`. (re-verified 2026-07-07: grepped all `scripts/standardize/*.py` for normalization/mean/std logic, none found)

## Step 4: exploratory understanding of each dataset
For each dataset, in `notebooks/eda/<name>.ipynb` (or a plain script if a notebook feels like overkill for something this small):
- [x] Basic shape: number of series/variables, length, frequency, missing value rate.
- [x] A couple of representative plots: a few individual series over time, and for multivariate sets, a correlation or seasonality glance. (re-verified 2026-07-07: correlation plots present for ETT/Electricity/Traffic/Weather/Solar/PEMS; GIFT-Eval's missing solar representative plot fixed this session — all three of its source configs now plotted)
- [x] For PTB-XL: label distribution across diagnostic superclasses/subclasses, per split, to see how imbalanced it is. (re-verified: `eda_ptbxl.py` computes per-split label prevalence and plots it, not just a single global count)
- [x] Write a short paragraph of observations into the dataset's card (see template at the bottom of `DATASETS.md`), things like "strong daily and weekly seasonality," "heavily imbalanced classes," "several long stretches of missing data," whatever's actually true of that dataset. This is meant to build real understanding, not just tick a box.

## Step 5: literature notes per dataset
For each dataset, fill out `literature/notes/<name>.md` using the dataset card template in `DATASETS.md`:
- [x] Confirm (don't just copy from this doc) the split convention and normalization convention by actually checking 1 to 2 of the papers listed, since conventions sometimes drift between papers even for the "standard" split. (2026-07-08: done for all 14 via dedicated research agents fetching primary sources — paper text and/or reference-implementation data loaders, not memory. One real correction found and applied: PEMS03/04/08's split convention was mischaracterized, see below.)
- [x] Note anything project-relevant that the literature flags about the dataset: known contamination concerns (particularly relevant given this whole project is about contamination auditing), known criticisms (e.g. near-random-walk behavior, short series, cherry-picked horizons), anything that would matter when this dataset shows up in the Phase 2 leakage matrix. (2026-07-08: done; see STATUS.md's final phase-summary entry for the headline findings, most notably that Meyer et al. 2025 (arXiv:2510.13654) independently confirms and extends this project's own ETT/solar contamination findings.)

## Step 6: wrap-up and handoff prep
- [x] Update the phase-scope summary table at the top of `DATASETS.md`, everything should be `[x]` or `[!]` with a reason. (2026-07-08: all 14 confirmed `[x]`, none blocked)
- [x] Write a final `STATUS.md` entry summarizing the whole phase: what's downloaded, what's standardized, what's blocked, total disk used, and a short note on what Phase 1 proper (once real compute/storage lands) should pick up first given what was learned here.
- [x] Since this is on borrowed, temporary compute, make sure everything under `datasets/` is something that can be `rsync`ed cleanly to the new server later, no absolute paths baked into any processed file, no dependency on anything outside `/workspace/TSFM/` and `config/paths.yaml`. (2026-07-08: audited; found and fixed one real violation — `standardize_monash.py` had a hardcoded `/workspace/Venv/TSFM/...` sys.path insert, removed since it was unnecessary under an activated venv anyway. `existing_assets_root` in `paths.yaml` pointing outside the repo is intentional, by design, for the WESAD symlink.)

## Explicitly out of scope for this phase, do not start these without being asked
- Loading or running any model, including the tiny/CPU-class ones (DLinear, Seasonal Naive, TTM-r2), that was a deliberate call for this phase, see `CLAUDE.md`.
- MIMIC-IV-ECG credentialing or download.
- Full GIFT-Eval or GiftEvalPretrain download.
- Anything from the EEG, CGM, or Finance domains.
- Building the actual inference harness (that's Phase 1 proper, once real compute is confirmed).
