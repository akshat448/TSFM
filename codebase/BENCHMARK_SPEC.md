# BENCHMARK_SPEC.md (condensed)

Full source: `documents/TSFM_BenchmarkSpecification.md`. This file exists so Claude Code does not have to re-read the full spec every session, but if anything here seems ambiguous, that file is the source of truth.

## Domain verdicts
| Domain | Verdict |
|---|---|
| Energy (ETT, ECL, Solar) | Required, the reproducible spine |
| Traffic (PEMS, Traffic) | Required |
| Weather (Jena) | Required |
| ECG (PTB-XL, MIMIC-IV-ECG) | Recommended, this is the scientific payload |
| EEG | Optional, only if ECG lands ahead of schedule |
| CGM, EHR, Wearables, Finance, Climate, Server Telemetry | Excluded from the benchmark itself, flagged as Phase 3 architecture targets |

## Task verdicts
| Task | Verdict |
|---|---|
| Zero-shot forecasting (point + probabilistic) | Required |
| Few-shot forecasting | Recommended |
| Cross-domain transfer | Recommended |
| ECG classification | Recommended (needed to evaluate domain FMs fairly) |
| Imputation, representation/linear probing | Optional |
| Anomaly detection, event prediction, retrieval, segmentation | Excluded for this phase |

## Dataset tiers
- **Minimal** (the reproducibility spine): ETTh1, ETTh2, ETTm1, ETTm2, Electricity (ECL), Traffic, Weather, PTB-XL.
- **Recommended** (the actual ICLR target): everything in Minimal plus GIFT-Eval subset, Solar-Energy, PEMS03/04/08, a Monash subset, MIMIC-IV-ECG.
- **Full** (optional, post week 6): TUAB/TUEV (EEG), C-MAPSS, ILI and Exchange Rate as labeled hard controls only.

**This phase's scope** (decided, see `DATASETS.md` for the working tracker): Minimal, plus the Recommended items that are small and headless-downloadable now: Solar-Energy, PEMS03/04/08, a GIFT-Eval subset, a Monash subset. MIMIC-IV-ECG is deferred (needs its own credentialing, separate from the MIMIC-IV clinical tables already on the server). PPG-DaLiA and the wearables extension are deferred as a dataset download task since WESAD is already present on the server, see `EXISTING_ASSETS.md`.

## Model set (~15 models across six families), reference only, not in scope this phase
- General TSFMs: Chronos-Bolt, TimesFM-2.5, Moirai + Moirai-MoE, MOMENT, TTM-r2, Chronos-2.
- Domain FMs: ECG-FM, HuBERT-ECG.
- Transformer baselines: DLinear, PatchTST, iTransformer.
- State-space: S-Mamba.
- LLM-based: GPT4TS.
- Statistical: Seasonal Naive, AutoARIMA.

DLinear and Seasonal Naive are non-negotiable, every TSFM claim in this project has to beat them.

## Evaluation protocol (Phase 2 will hold everything to this, structure data now so nothing needs redoing)
- **Splits:** 6:2:2 for ETT (Time-Series-Library convention), 7:1:2 for Electricity/Traffic/Weather/Solar/PEMS. PTB-XL uses its official fold split (folds 1 to 8 train, fold 9 validation, fold 10 test). All physiological data uses subject-level splits, no patient appears in more than one split, this is the single most common leakage error in ECG/EEG papers.
- **Zero-shot labeling:** every zero-shot number gets a contamination label, certified-unseen, known-overlap, or undocumented/unknown.
- **Fine-tuning:** two regimes per model that supports it, parameter-efficient (LoRA or head-only) and full fine-tuning, under a fixed equal budget.
- **Few-shot:** target-domain adaptation at 5% and 10% of training data, fixed seeds, multiple draws for variance.
- **Significance testing:** required for any SOTA claim, paired Wilcoxon signed-rank, Diebold-Mariano, bootstrap confidence intervals, geometric-mean rank aggregation, Holm correction for multiple comparisons.
- **Metrics:** MSE, MAE, MASE for point forecasts; CRPS, WQL, empirical coverage for probabilistic; macro-F1, AUROC, ECE for classification.
- **Leakage auditing (the centerpiece):** an explicit model by dataset contamination matrix, checking each model's documented pretraining corpus against each test set.

## Why this matters for the current data-only phase
Even though no models are being run yet, the way datasets get split, normalized, and documented now should already match this protocol (subject-level splits for anything physiological, train-only normalization statistics, the 6:2:2 / 7:1:2 conventions, standard horizons of 96/192/336/720 for the LTSF datasets). That way nothing downstream needs to be redone in Phase 2.
