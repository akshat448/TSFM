# DATASETS.md (working tracker for this phase)

This is where the actual work happens. Update the status column as you go, and fill in a full dataset card (template at the bottom) for each dataset before marking it "done". Keep raw data under `datasets/raw/<domain>/<n>/` and standardized output under `datasets/processed/<domain>/<n>/`, both paths resolved from `config/paths.yaml`, never hardcoded.

**NOTE (2026-07-30): path convention changed on the new server.** Everything below refers to the bridge-phase layout (`datasets/raw/...`, `datasets/processed/...` nested inside the repo). On the new server these are `dataset/raw/...` and `dataset/processed/...` as SIBLING directories to `codebase/`, not nested inside it — see `PATHS_AND_MIGRATION.md`. The dataset content/status below is otherwise still accurate and should be treated as the source of truth once the actual bytes are migrated over.

Everything below assumes headless download (no browser, no interactive OAuth), since this is a DGX server with no display. Where the standard source needs a browser (Google Drive links are the usual offender), an alternate headless source is given, flagged clearly.

## Status legend
`[ ]` not started, `[~]` downloading/in progress, `[x]` downloaded and dataset-carded, `[!]` blocked, see notes.

---

## Phase-scope summary (Minimal + light Recommended, target 6 to 8GB)

| # | Dataset | Domain | Tier | Est. size | Status |
|---|---|---|---|---|---|
| 1 | ETTh1 | Energy | Minimal | <5MB | [x] |
| 2 | ETTh2 | Energy | Minimal | <5MB | [x] |
| 3 | ETTm1 | Energy | Minimal | <15MB | [x] |
| 4 | ETTm2 | Energy | Minimal | <15MB | [x] |
| 5 | Electricity (ECL) | Energy | Minimal | ~130MB | [x] |
| 6 | Solar-Energy | Energy | Recommended | ~30MB | [x] |
| 7 | Traffic | Traffic | Minimal | ~200MB | [x] |
| 8 | PEMS03 | Traffic | Recommended | ~50MB | [x] |
| 9 | PEMS04 | Traffic | Recommended | ~40MB | [x] |
| 10 | PEMS08 | Traffic | Recommended | ~40MB | [x] |
| 11 | Weather (Jena) | Weather | Minimal | ~15MB | [x] |
| 12 | PTB-XL | ECG | Minimal | ~2GB (WFDB) / ~11GB if you grab both 500Hz and 100Hz plus raw, see notes | [x] |
| 13 | GIFT-Eval (subset) | Multi-domain | Recommended | <1GB if filtered to 2 to 3 domains | [x] |
| 14 | Monash Archive (subset) | Multi-domain | Recommended | ~500MB for ~15 to 20 smaller series | [x] |

Deferred this phase (not a download task, tracked separately): MIMIC-IV-ECG (needs credentialing), full GIFT-Eval, PPG-DaLiA, WESAD (already present, see `EXISTING_ASSETS.md`), EEG, CGM, ILI, Exchange Rate.

**Status as of the new-server move (2026-07-30): all 14 above are marked `[x]` from the bridge phase. This means "the bytes existed and were correct on the OLD server" — it does NOT yet mean the bytes have landed on the new server. Re-verify after migration (`PATHS_AND_MIGRATION.md` step 5) before trusting this table again.**

---

## 1 to 4. ETT family (ETTh1, ETTh2, ETTm1, ETTm2)

- **Domain / task:** Energy, long-term forecasting (also used for imputation/anomaly in some papers, out of scope here).
- **Source:** github.com/zhouhaoyi/ETDataset (original authors, Informer paper). Mirror: HuggingFace `thuml/Time-Series-Library`, config names `ETTh1`/`ETTh2`/`ETTm1`/`ETTm2`.
- **Headless download:**
  ```
  git clone https://github.com/zhouhaoyi/ETDataset.git
  # CSVs are at ETDataset/ETT-small/ETTh1.csv, ETTh2.csv, ETTm1.csv, ETTm2.csv
  ```
  or via HuggingFace (equivalent content, useful if GitHub is flaky):
  ```python
  from huggingface_hub import hf_hub_download
  hf_hub_download("thuml/Time-Series-Library", "ETT-small/ETTh1.csv", repo_type="dataset")
  ```
- **Raw format:** CSV, columns `date, HUFL, HULL, MUFL, MULL, LUFL, LULL, OT`. OT (oil temperature) is the canonical forecast target in univariate setups; all 7 columns are used in multivariate setups.
- **Size:** ETTh1/h2 ~17,420 hourly steps each. ETTm1/m2 ~69,680 steps each (15 min).
- **Standard split:** 12/4/4 months train/val/test (the "6:2:2" convention cited across papers). Chronological, no shuffling.
- **Standard normalization:** per-channel z-score (standardization), statistics computed from the training split only, then applied to val/test.
- **Standard horizons:** {96, 192, 336, 720} for long-term forecasting.
- **Key papers:** Informer (Zhou et al., AAAI 2021, introduced the dataset), Autoformer, PatchTST, DLinear/LTSF-Linear, iTransformer, TimesNet, virtually every TSFM zero-shot eval (Chronos, TimesFM, Moirai, Sundial).
- **Gotchas:** ETTh1/ETTh2 are the same two transformers as ETTm1/ETTm2 at different sampling frequency, they are correlated, do not treat all four as fully independent domains when reporting aggregate stats. Some sources license this CC BY, others CC BY-NC, note whichever the actual repo states at download time. **Actual license found at download (2026-07-06): CC BY-ND 4.0**, per the repo's own `LICENSE` file — update whichever assumption this project was carrying.
- **Status:** [x] standardized to `datasets/processed/energy/ETT{h1,h2,m1,m2}.parquet`, EDA done, dataset cards written to `literature/notes/ETT{h1,h2,m1,m2}.md`. Confirmed real train/val/test distribution shift in all four (see cards).

---

## 5. Electricity (ECL)

- **Domain / task:** Energy, long-term forecasting.
- **Source:** UCI ElectricityLoadDiagrams20112014, commonly redistributed pre-cleaned by the Autoformer/Informer authors. Headless mirror: HuggingFace `thuml/Time-Series-Library`, config `electricity`.
  ```python
  from huggingface_hub import hf_hub_download
  hf_hub_download("thuml/Time-Series-Library", "electricity/electricity.csv", repo_type="dataset")
  ```
  Alternative: `pip install datasetsforecast` then `from datasetsforecast.long_horizon import LongHorizon2; LongHorizon2.load(directory, group='ECL')`, this pulls from a direct S3 zip (`https://nhits-experiments.s3.amazonaws.com/datasets.zip`), fully headless. Note this wrapper applies its own train-mean/std normalization when you call it, prefer the raw CSV route above so we control normalization ourselves per `BENCHMARK_SPEC.md`.
- **Raw format:** CSV, 321 client columns, hourly consumption (kWh).
- **Size:** ~26,304 hourly steps x 321 variables, roughly 130 to 150MB as CSV.
- **Standard split:** 7:1:2, chronological.
- **Standard normalization:** per-channel z-score from train stats.
- **Key papers:** Informer, Moirai, PatchTST, virtually every multivariate LTSF paper as the "high dimensional" stress test.
- **Status:** [x] standardized to `datasets/processed/energy/electricity.parquet`, EDA done (found one client with 100%-zero test split, see card), card at `literature/notes/electricity.md`

---

## 6. Solar-Energy

- **Domain / task:** Energy, long-term forecasting, strong daily seasonality.
- **Source:** LSTNet lineage, `laiguokun/multivariate-time-series-data` on GitHub.
- **Headless download:**
  ```
  git clone https://github.com/laiguokun/multivariate-time-series-data.git
  # solar-energy/solar_AL.txt.gz
  ```
- **Raw format:** comma-separated txt (gzipped), no header, 137 columns (PV plants), 10-minute intervals.
- **Size:** ~52,560 steps x 137 variables, ~30MB.
- **Standard split:** 7:1:2, chronological.
- **Key papers:** LSTNet (Lai et al., SIGIR 2018, introduced it), iTransformer, Moirai.
- **Gotchas:** no header row and no timestamp column in the raw file, timestamps have to be reconstructed from the known start date and 10 minute frequency, document whatever start date you use.
- **Status:** [x] standardized to `datasets/processed/energy/solar_energy.parquet` (timestamps reconstructed assuming 2006 start, see card), EDA done, card at `literature/notes/solar_energy.md`.

---

## 7. Traffic

- **Domain / task:** Traffic, long-term forecasting, extreme dimensionality stress test.
- **Source:** Caltrans PeMS derived, redistributed by the Autoformer/Informer authors. Headless mirror: HuggingFace `thuml/Time-Series-Library`, config `traffic`.
  ```python
  from huggingface_hub import hf_hub_download
  hf_hub_download("thuml/Time-Series-Library", "traffic/traffic.csv", repo_type="dataset")
  ```
- **Raw format:** CSV, 862 sensor columns, hourly road occupancy rates (0 to 1), San Francisco Bay Area.
- **Size:** ~17,544 steps x 862 variables, ~200MB.
- **Standard split:** 7:1:2, chronological.
- **Key papers:** Informer, PatchTST, iTransformer, TimesNet.
- **Status:** [x] standardized to `datasets/processed/traffic/traffic.parquet`, EDA done, card at `literature/notes/traffic.md`

---

## 8 to 10. PEMS03, PEMS04, PEMS08

- **Domain / task:** Traffic, spatio-temporal forecasting, cross-variate stress test (this is a different, sensor-graph-based PEMS product than the "Traffic" dataset above, do not conflate them).
- **Source:** Caltrans PeMS, standard redistribution via the ASTGCN / STSGCN lineage of repos (used by iTransformer, S-Mamba). These are commonly hosted as `.npz` files on Google Drive or Baidu in the original repos, which is not headless-friendly.
- **Headless approach:**
  1. First check whether `thuml/iTransformer` or `thuml/Time-Series-Library` GitHub repos link a direct/HuggingFace mirror at the time you do this (check their README data section, it changes over time).
  2. If only Google Drive is available, use `gdown <file_id>` (add `gdown` via `uv pip install gdown`) rather than a manual browser download, it works headlessly for public files given the file ID from the share link.
  3. Fallback: search Kaggle for a PEMS03/04/08 mirror and use the Kaggle API (`kaggle datasets download ...`), headless once an API token is configured.
- **Raw format:** `.npz` arrays, sensor readings (flow / speed / occupancy depending on the file) at 5 minute intervals, plus a sensor adjacency/distance file for the graph structure.
- **Sizes:** PEMS03 ~358 sensors / ~26,208 steps, PEMS04 ~307 sensors / ~16,992 steps, PEMS08 ~170 sensors / ~17,856 steps. Each tens of MB.
- **Standard split — CORRECTED 2026-07-08 (was wrong before Step 5):** originally described here as an open choice between graph-lineage 6:2:2 (ASTGCN-style) and an LTSF-style 7:1:2 attributed to iTransformer/S-Mamba. Step 5 literature confirmation (code-level check of ASTGCN's `prepareData.py`, STSGCN, iTransformer's actual `Dataset_PEMS` loader, and S-Mamba's paper text) found that framing was wrong: **all four papers actually use 6:2:2 on PEMS03/04/08.** There is no real competing 7:1:2 convention for these three datasets specifically — 7:1:2 was this project's own import from the unrelated LTSF family (Traffic/Electricity/ETT), not something any cited paper does here. Per user decision (2026-07-08): the processed parquet files now store **both** — `train_end_idx`/`val_end_idx`/`test_end_idx` are the literature-correct 6:2:2 split (primary), and `train_end_idx_ltsf712`/`val_end_idx_ltsf712`/`test_end_idx_ltsf712` preserve the original 7:1:2 columns alongside. See `scripts/standardize/standardize_pems.py` and the three dataset cards for full detail.
- **Key papers:** ASTGCN (Guo et al., AAAI 2019, introduced the redistribution and the 6:2:2 convention), STSGCN (Song et al., AAAI 2020), iTransformer (Liu et al., ICLR 2024), S-Mamba — all four confirmed to use 6:2:2 on these datasets, contrary to what this section previously implied.
- **Gotchas:** flag this one clearly in `STATUS.md` if the headless routes above don't pan out, it is the most likely dataset in this batch to need a manual step, don't spend excessive time fighting it, PEMS04 or PEMS08 alone (smaller, well documented) is enough to unblock everything else while we sort out the rest. **Update:** no HF/official direct mirror exists; got all three headlessly via `gdown` against iTransformer's official Google Drive bundle (paused and confirmed this approach with the user first, see `datasets/manifests/pems.log`). This bundle's PEMS folder does not include the sensor adjacency/distance files, only needed for graph-based baselines, not for TSFM point-forecast eval — flagged for later if graph baselines are ever added.
- **Status:** [x] standardized to `datasets/processed/traffic/pems{03,04,08}.parquet` (re-standardized 2026-07-08 to store both the literature-correct 6:2:2 split as primary and the original 7:1:2 as alternate columns, after Step 5 found the prior "competing conventions" framing was wrong — see above), EDA re-verified against the new primary split, cards at `literature/notes/pems{03,04,08}.md`

---

## 11. Weather (Jena)

- **Domain / task:** Weather, multivariate long-term forecasting.
- **Source:** Max Planck Institute Jena weather station, redistributed by the Autoformer authors. Headless mirror: HuggingFace `thuml/Time-Series-Library`, config `weather`.
  ```python
  from huggingface_hub import hf_hub_download
  hf_hub_download("thuml/Time-Series-Library", "weather/weather.csv", repo_type="dataset")
  ```
- **Raw format:** CSV, 21 meteorological variables, 10-minute intervals, year 2020.
- **Size:** ~52,696 steps x 21 variables, ~15MB.
- **Standard split:** 7:1:2, chronological.
- **Key papers:** Autoformer (introduced this redistribution), PatchTST, iTransformer, DLinear.
- **Status:** [x] standardized to `datasets/processed/weather/weather.parquet`, EDA done (found upstream column-name encoding corruption + strong seasonal train/test shift, see card), card at `literature/notes/weather.md`

---

## 12. PTB-XL

- **Domain / task:** ECG, classification (also used for forecasting/representation work in some TSFM papers). This is the Minimal tier's ECG entry and the entry point to the project's scientific payload.
- **Source:** PhysioNet, open access (free account required, no full credentialing like MIMIC).
- **Headless download:**
  ```
  wget -r -N -c -np https://physionet.org/files/ptb-xl/1.0.3/
  ```
  If the open-access files require authentication at download time, register a free PhysioNet account first and pass credentials with `--user`/`--password`, or use `wget --http-user=<user> --ask-password`. Verify completeness against the published SHA256SUMS.txt after downloading, PhysioNet's recursive wget crawl can silently drop files on transient failures, this is a documented issue.
- **Raw format:** WFDB format (`.dat` + `.hea` pairs) at both 100Hz and 500Hz sampling, 12-lead, 10 second recordings, plus `ptbxl_database.csv` (metadata, `scp_codes`, `strat_fold`) and `scp_statements.csv` (label definitions and diagnostic superclass/subclass hierarchy).
- **Size:** ~21,837 records, ~2GB if you take only the 100Hz version, larger (~10 to 11GB) if you also pull the 500Hz version. Start with 100Hz only for this phase, it is what most baseline papers use for a first pass and keeps us well inside budget.
- **Standard split:** official 10-fold `strat_fold` column, folds 1 to 8 train, fold 9 validation, fold 10 test. This is stratified and patient-respecting by construction, folds 9 and 10 additionally underwent extra human validation so they are considered the highest label quality. Never resplit at the record level, always at the patient/fold level.
- **Standard labels:** `scp_codes` mapped through `scp_statements.csv` to diagnostic superclass (5 classes: NORM, MI, STTC, CD, HYP) or diagnostic subclass, depending on the paper. Multi-label in general.
- **Key papers:** Wagner et al., Scientific Data 2020 (introduced the dataset), ECG-FM, HuBERT-ECG, most modern ECG classification work uses the official fold split and reports macro-AUC or macro-F1.
- **Reading tool:** the `wfdb` Python package (`uv pip install wfdb`) is the standard way to load WFDB records, e.g. `wfdb.rdsamp(path)`.
- **Gotchas:** subject-level split integrity is the single most common leakage error in ECG papers, per `BENCHMARK_SPEC.md` this project treats it as a first-class requirement, so when you standardize this dataset carry `strat_fold` through untouched rather than re-deriving your own split.
- **Status:** [x] standardized to `datasets/processed/ecg/ptbxl.parquet` (custom classification schema, not the forecasting schema — record_id/signal/labels/strat_fold), patient/split leakage integrity asserted in code and confirmed zero violations, EDA done (class balance nearly identical across train/val/test, confirming stratification worked), card at `literature/notes/ptbxl.md`.

---

## 13. GIFT-Eval (subset)

- **Domain / task:** Multi-domain, zero-shot forecasting benchmark and non-leaking reference corpus.
- **Source:** HuggingFace `Salesforce/GiftEval` (23 datasets, 7 domains, 10 frequencies, 144k+ series). Full dataset is ~5GB, this phase only wants a filtered subset (target <1GB).
- **Headless download (full, for reference):**
  ```
  huggingface-cli download Salesforce/GiftEval --repo-type dataset --local-dir <path>
  ```
- **Headless download (subset, preferred for this phase):** use `--include` with a glob for specific domain/config folders, or load specific configs directly:
  ```python
  from datasets import load_dataset
  ds = load_dataset("Salesforce/GiftEval", name="<specific_config_name>")
  ```
  Pick 2 to 3 domains that overlap with our existing Required domains (energy, transport) to keep this coherent with the rest of the benchmark, list the exact configs chosen in the dataset card once decided.
- **Standard protocol:** GIFT-Eval enforces its own fixed train/test splits and ships a separate non-leaking pretraining corpus (`Salesforce/GiftEvalPretrain`, ~230B points, do not download this now, it is pretraining-corpus scale and out of scope for this phase, relevant later for the leakage audit).
- **Key papers:** Aksu et al., NeurIPS 2024 (introduced it), used as the primary modern zero-shot leaderboard by TimesFM-2.5, Chronos-2, Moirai-2.0.
- **Chosen configs this phase:** `electricity`, `solar` (Energy overlap), `LOOP_SEATTLE` (Traffic/transport overlap), all available frequencies per config. See `datasets/manifests/gift_eval.log` for rationale.
- **Status:** [x] standardized to `datasets/processed/multi_domain/gift_eval.parquet`, EDA done. **Important finding: `electricity` and `solar` configs are very likely the same underlying source data as our own dataset #5/#6 under different curation** — flagged for a proper Phase-2 leakage-matrix check, see card at `literature/notes/gift_eval.md`.

---

## 14. Monash Archive (subset)

- **Domain / task:** Multi-domain, zero-shot forecasting, scale-free MASE evaluation across many frequencies.
- **Source:** forecastingdata.org, individual datasets hosted as `.tsf` files on Zenodo, direct download, no auth needed.
- **Headless download:** pick roughly 15 to 20 of the smaller datasets from forecastingdata.org (each has its own direct Zenodo record/link), download the `.tsf` files with `wget`/`requests`. Favor datasets already highlighted in the Chronos and TimesFM papers as reference points, since that gives us a like-for-like comparison later. Record the exact list chosen in the dataset card.
- **Raw format:** `.tsf` (Monash's own tab-separated time series format), needs a small parser, either write one or use an existing reader (the `gluonts` or `sktime` ecosystems both have Monash `.tsf` readers, check what's already installed before writing a new one).
- **Standard protocol:** MASE across the archive is the standard cross-frequency metric, missing values are typically handled with LOCF (last observation carried forward) imputation per the archive's own convention.
- **Key papers:** Godahewa et al., NeurIPS 2021 (introduced the archive), used by Chronos, TimesFM, Time-MoE, Moirai as pretraining/eval reference.
- **Chosen 20 datasets this phase:** m1_yearly, m1_quarterly, m1_monthly, nn5_daily_with_missing, nn5_weekly, tourism_monthly, tourism_quarterly, tourism_yearly, cif_2016, car_parts_without_missing, fred_md, pedestrian_counts, hospital, covid_deaths, australian_electricity_demand, electricity_weekly, rideshare_without_missing, saugeenday, solar_10_minutes, sunspot_without_missing. See `datasets/manifests/monash.log` for exact rationale and exclusions.
- **Status:** [x] standardized to `datasets/processed/multi_domain/monash.parquet`. 10 of 20 sub-datasets shipped an `@horizon` tag directly; the other 10 did not and were deliberately left unsplit until Step 5. **Update 2026-07-08: all 10 now resolved** via Godahewa et al.'s own frequency-based horizon rule (Section 4.1), cross-checked against Chronos/TimesFM's published Monash tables where available — see `horizon_source` column in the parquet and full citations in the card; re-standardized and re-ran EDA, all 20/20 sub-datasets now have an explicit train/test split. **Major finding: `solar_10_minutes` confirmed byte-identical to our own solar_energy.parquet AND GIFT-Eval's solar/10T (three-way duplicate)**, card at `literature/notes/monash.md`.

---

## Deferred / already present, tracked but not a download task this phase

| Dataset | Why deferred | Notes |
|---|---|---|
| MIMIC-IV-ECG | Needs its own PhysioNet credentialing, not started, separate from MIMIC-IV clinical | See `CLAUDE.md` known open items |
| MIMIC-IV clinical (hosp/icu) | Already present at `/workspace/TimeSeriesBenchMark/physionet.org/files/mimiciv/3.1/` (~10GB, `hosp/`+`icu/` `.csv.gz` tables, confirmed clinical not ECG waveform) | Useful later for Phase 3 EHR direction, not part of this phase's ECG benchmark work, see `EXISTING_ASSETS.md` |
| WESAD | Already present on server, confirmed | Location confirmed at `/workspace/TimeSeriesBenchMark/datasets/WESAD/` (19GB, 15 subjects), symlinked into `datasets/raw/wearables/wesad/`, no fresh download needed. See `EXISTING_ASSETS.md`. |
| PPG-DaLiA | Team extension tier, not selected for this phase | UCI ML Repository, direct zip, headless, revisit later |
| Other lab directories (`Emotiv`, `Brain`, `SensorLLM`, `GlycoLM`, `TSZechen`) | Inventoried 2026-07-08, confirmed EEG/HAR/CGM research (188GB/882GB/8GB/65GB/2TB respectively) — none overlap this project's Minimal/Recommended scope, all fall under domains `CLAUDE.md` already excludes this phase | See `EXISTING_ASSETS.md` for full per-directory breakdown. "engauge" was searched for explicitly across all five and not found — still genuinely unresolved, not a naming match for anything here. |
| GIFT-Eval full, GiftEvalPretrain | Pretraining-corpus scale, out of budget and out of scope | Relevant later for the leakage audit, not now |
| EEG (TUAB/TUEV), CGM, ILI, Exchange Rate | Full tier or excluded domains per `BENCHMARK_SPEC.md` | Not this phase |

---

## Dataset card template (copy this into a per-dataset note under `literature/notes/<n>.md` once a dataset moves to `[x]`)

```markdown
# <Dataset name>

- Domain / task:
- Source (headless download command):
- License:
- Raw format and schema:
- Size on disk (actual, after download):
- Standard split convention (with citation):
- Standard normalization convention (with citation):
- Standard horizons / windows (if forecasting):
- 2 to 3 papers that use it and how:
- Preprocessing applied for this project (exact steps):
- Known gotchas / leakage risks:
- Where the raw and processed files live (path, resolved via config/paths.yaml):
```
