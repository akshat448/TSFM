# PROJECT.md (condensed)

Full source: `documents/TSFM_Project_Overview_Document.md` (identical content to `documents/TSFM Project Document.pdf` and `.docx`).

## The core idea
Two connected, deliberately sequential goals:

1. Build a contamination controlled, cross domain evaluation of existing Time Series Foundation Models (TSFMs). Find out where they actually generalize versus where their reported strength rests on saturated or contaminated benchmarks.
2. Use the evidence that evaluation produces, not intuition, to design, train, and evaluate a new TSFM architecture that directly targets the failure modes the benchmark exposes.

Architecture work comes last on purpose. A new model proposed without a clean evaluation baseline invites the same criticism the field currently applies to everyone else. Right now, everything we are doing is instrument building.

## Why evaluation has to come first, four documented weaknesses in the field
- **Saturation.** Almost every TSFM paper reports on the same handful of datasets (ETT, Electricity, Traffic, Weather), where improvements are now marginal and hard to interpret.
- **Contamination.** Several leading models are suspected or confirmed to include eval data in pretraining. No benchmark currently publishes a model by dataset leakage audit.
- **Domain narrowness.** High value domains like physiological signals (ECG) and continuous glucose monitoring are barely tested. It is unknown whether general pretraining transfers there.
- **Evaluation inconsistency.** Splits, horizons, normalization, and aggregation differ paper to paper. Significance testing is the exception, not the norm.

## Three phase structure
| Phase | Weeks | Focus |
|---|---|---|
| Phase 1 | 1 to 3 | Dataset setup, standardization, unified inference harness, reproduction gate |
| Phase 2 | 3 to 6 | Full benchmarking sweep: zero-shot, few-shot, fine-tuning, cross-domain transfer matrix, significance testing |
| Phase 3 | 7 to 8+ | Gap analysis, architecture design brief, paper drafting, begin new model development |

**Where we actually are:** doing early Phase 1 dataset work ahead of schedule, on borrowed single-GPU compute, while the collaborator's full compute and storage allocation is arranged. See `CLAUDE.md` for the exact scope boundary of this bridge phase.

## The two headline findings the whole benchmark exists to produce
1. An honest, leakage-audited, cross-domain comparison showing where general TSFMs actually generalize and where their reported zero-shot strength rests on contamination.
2. A general-purpose TSFM versus domain-specific FM result on ECG (against ECG-FM and HuBERT-ECG), showing whether broad pretraining transfers into a physiological domain that already has its own specialists.

The contamination matrix makes finding 1 novel regardless of which way the numbers fall, which is what de-risks the publication.

## Publication target
Primary: ICLR main track. Fallback: NeurIPS Datasets and Benchmarks track.

## Infrastructure needed for the full project (coming later from the collaborator, not needed for this phase)
- Storage: approximately 5TB (datasets, checkpoints, cached forecasts, intermediate artifacts).
- Compute: cluster of 8x H100 or B200 80GB GPUs.

## Companion documents
- `BENCHMARK_SPEC.md`: condensed domain/task/dataset/model selection and evaluation protocol.
- `DATASETS.md`: the working dataset tracker for the current phase.
- `documents/TSFM_OpenQuestions.md`: the 20 ranked open research questions that will inform the Phase 3 architecture direction (Q19 contamination-free eval and Q20 cross-domain aggregation are effectively what this benchmark already targets; Q12 CGM foundation model and Q18 multimodal physiological FM are the flagged high-upside greenfield targets for later).
