# PHASE1_PLAN.md

Companion to `PROJECT.md`, `BENCHMARK_SPEC.md`, and the closed-out bridge phase (`DATASETS.md`, `STATUS.md`). This is the plan for actually starting Phase 1: unified inference harness, first model runs, reproduction gate. Written before touching real compute, so it should be reviewed and adjusted once you're back on the server.

Compute reality check: still the single A100 80GB, not the collaborator's cluster. That shapes the model priority order below, not the harness design itself, the harness should not need rewriting once real compute lands.

## 1. What "Phase 1" actually means here

Per `PROJECT.md`, Phase 1 covers dataset setup (done), standardization (done), the unified inference harness (not started), and a reproduction gate (not started). This plan covers the last two.

"Reproduction gate" means: before trusting any new number this project produces, first reproduce a known published number (e.g. DLinear's published ETTh1 MSE at horizon 96, or Chronos-Bolt's published GIFT-Eval zero-shot numbers) closely enough to trust the harness. If the harness cannot reproduce a published baseline, nothing built on top of it is trustworthy. This should be milestone 1, not an afterthought.

## 2. Harness architecture

```
scripts/harness/
  data.py            # loads standardized parquet -> windowed eval instances
  metrics.py         # MSE, MAE, MASE, CRPS, WQL, coverage, macro-F1, AUROC, ECE
  contamination.py   # attaches a contamination label to every result row
  registry.py        # model name -> adapter class lookup
  eval_runner.py      # orchestrates dataset x model x horizon, writes results
  models/
    base.py           # ForecastModel / ClassifierModel abstract interfaces
    seasonal_naive.py
    dlinear.py
    chronos_bolt.py    # adapter, needs `chronos-forecasting` pip package
    timesfm.py         # adapter, needs `timesfm` pip package
    ttm.py             # adapter, needs `tsfm_public` / granite-tsfm

config/
  phase1_eval.yaml         # which datasets, models, horizons to run
  contamination_matrix.yaml # seeded from the bridge phase's literature findings

run_phase1.py          # thin CLI entrypoint
```

Design principles, matching how the bridge phase already works:
- **No hardcoded paths.** Everything reads dataset locations from the existing `config/paths.yaml`, model checkpoint cache locations get their own key in the same file.
- **Model-agnostic core.** `eval_runner.py` never imports a specific model library directly. It only knows the `ForecastModel` interface (`fit(context) -> None` for models that need per-series/per-dataset fitting, `predict(context, horizon) -> point, quantiles`). Adding model 16 means writing one adapter file, not touching the runner.
- **Every result row carries a contamination label.** Per `BENCHMARK_SPEC.md`'s requirement, this is not a Phase 2 afterthought bolted on later, it's a column in the output schema from the first run. `contamination.py` looks up (dataset, model) in `contamination_matrix.yaml` and defaults to `undocumented/unknown` if there's no entry, never silently blank.
- **Raw results, not pre-aggregated.** The runner writes one row per (dataset, series/item_id, model, horizon, window) with the raw metrics. Aggregation, significance testing, and rank tables are a separate downstream step (Phase 2's job per `BENCHMARK_SPEC.md`), so nothing about how results get summarized is baked in early.

## 3. Dataset schema the harness has to handle

Two schemas exist from the bridge phase, the harness needs both code paths:

**Forecasting schema** (ETT family, Electricity, Traffic, Weather, Solar, PEMS03/04/08, GIFT-Eval, Monash): one row per channel/series, columns `item_id`, `group_id`, full float32 `target` array, plus split boundaries. Two sub-cases:
- **LTSF-style** (ETT/Electricity/Traffic/Weather/Solar/PEMS): explicit `train_end_idx`/`val_end_idx`/`test_end_idx`, standard horizons `{96, 192, 336, 720}`. PEMS03/04/08 additionally carry `*_ltsf712` alternate columns, default to the primary (literature-correct 6:2:2) columns unless a run config explicitly asks for the LTSF-style alternate.
- **Archive-style** (GIFT-Eval, Monash): no fixed `train_end_idx`/`val_end_idx`, GIFT-Eval needs the Table 13 prediction-length lookup already recorded in `gift_eval.md`, Monash uses the `horizon`/`horizon_source` columns already in `monash.parquet`.

**Classification schema** (PTB-XL only): one row per record, `signal` (1000x12), `diagnostic_superclass`, `strat_fold`, `split`. Entirely separate eval path, point/probabilistic forecast metrics do not apply.

`data.py` should expose one function per schema (`load_forecasting_windows(dataset, horizon, split)`, `load_ptbxl(split)`) rather than trying to force both into one interface.

## 4. Model priority, given single-GPU reality

Not the full ~15 model catalog at once. Sequenced so the harness gets validated cheaply before spending GPU time on anything expensive.

**Wave 1, no GPU really needed, run first to validate the harness end to end:**
- Seasonal Naive
- DLinear

These are `BENCHMARK_SPEC.md`'s non-negotiable baselines anyway, every later claim has to beat them, so they double as the reproduction gate target.

**Wave 2, light TSFMs, comfortably fit single A100 (<8GB VRAM each per the model catalog):**
- Chronos-Bolt (base, ~200M)
- TimesFM-2.5 (200M)
- TTM-r2 (IBM Tiny Time Mixers, smallest of the general TSFMs, sometimes CPU-viable)

First real "does a foundation model beat a linear baseline zero-shot" result. This is where the contamination labels start actually mattering, several of these datasets have known or suspected overlap with these models' pretraining corpora, so the numbers need the asterisk from day one, not added retroactively.

**Wave 3, still single-GPU feasible, needs actual training not just zero-shot:**
- PatchTST, iTransformer (transformer baselines, trained per dataset)
- S-Mamba
- Moirai-1.1-R (S/B variants, 8-24GB)

**Wave 4, larger or domain-specific, revisit once Wave 1-3 numbers exist:**
- MOMENT, Chronos-2, Moirai-MoE
- GPT4TS (LLM-based, heaviest of the general set)
- AutoARIMA (classical, no GPU, but slow per-series, budget wall-clock not VRAM)
- ECG-FM, HuBERT-ECG for PTB-XL, needs the classification eval path built first. Note `ptbxl.md`'s existing finding: ECG-FM is plausibly contaminated on PTB-XL's test fold via its Challenge-2021 pretraining, so this comparison should be run and reported with that caveat attached, not skipped because of it, that contrast (contaminated domain FM vs clean general TSFM zero-shot) is headline finding #2 for this whole project.

## 5. Dataset priority

1. ETT family + Electricity + Weather first. Smallest, best understood from the bridge phase, cheapest way to validate the harness.
2. Traffic, Solar, PEMS03/04/08 next. Same forecasting schema, exercises the PEMS dual-split-convention logic specifically.
3. GIFT-Eval, Monash. Exercises the archive-style horizon lookup and multi-frequency handling.
4. PTB-XL last. Needs the separate classification path, matters most once an ECG domain FM baseline is ready to compare against.

## 6. Milestones

- **M1 (reproduction gate):** Seasonal Naive + DLinear on ETTh1, all four horizons, results match published DLinear numbers within a small tolerance. Do not proceed past this until it passes, this is what makes every later number trustworthy.
- **M2:** Same two models across all Minimal-tier forecasting datasets. Full non-negotiable-baseline table.
- **M3:** Add Chronos-Bolt + TimesFM-2.5 zero-shot on the same datasets, contamination labels attached and visible in the output.
- **M4:** Extend to Recommended-tier (Solar, PEMS, GIFT-Eval, Monash).
- **M5:** PTB-XL classification path + ECG-FM baseline.
- **M6:** Significance testing layer (paired Wilcoxon, Diebold-Mariano, bootstrap CI, Holm correction), once enough model x dataset combinations exist to compare. This is really Phase 2's start, listed here just so the raw-results schema from M1 onward is already shaped to support it without rework.

## 7. What this plan deliberately does not cover yet

- Few-shot adaptation (5%/10% target-domain, per `BENCHMARK_SPEC.md`), that's layered on top of the same harness once zero-shot is solid.
- Fine-tuning regimes (LoRA/head-only vs full budget-matched), same reasoning.
- The full significance-testing suite, stubbed as M6 above, not built out yet.
- Anything on MIMIC-IV-ECG, still deferred pending its own credentialing.
