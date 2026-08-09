# Pretrained TSFM Evaluation Standard

## Scope

This contract applies to zero-shot forecasting by Chronos-Bolt Base, TimesFM-2.5 200M, Moirai-1.1-R Base, Moirai-MoE Base, MOMENT-1-large, TTM-r2, and Chronos-2. Classification, few-shot adaptation, and fine-tuning are separate protocols.

## Comparable evaluation

- Evaluate the standardized Minimal and Recommended forecasting datasets only; retain each dataset's published split and native GIFT-Eval/Monash protocol.
- For LTSF-style test splits, score at most 100 deterministic, evenly spaced rolling origins per item and horizon. The first and final legal origins are retained.
- The forecast horizon is never extended recursively unless the selected checkpoint explicitly requires documented rolling extension; record that choice in the result row.
- Pass only history available at each forecast origin. Adapter-specific inference preprocessing is permitted only when required by its upstream model; predictions are converted back to original units before metric calculation.
- Score MSE, MAE, and train-split-only MASE. Score WQL and quantile-derived CRPS only when the model returns the standard quantiles. Missing probabilistic output remains null, never fabricated.

## Reproducibility and contamination

- Stage checkpoints through the manifest, record their immutable Hub revision, package versions, Python/CUDA details, git revision, input config hash, device, and model preprocessing policy in `run_manifest.json`.
- Store raw one-row-per-window metrics and do not aggregate inside the runner. Every row receives a contamination label from `contamination_matrix.yaml`.
- Keep known duplicated datasets for documented overlap analysis, but attach a duplicate-group identifier so later rank aggregation cannot mistake them for independent evidence.

## Execution contract

- `stage` only downloads manifest-listed checkpoints into `checkpoint_root`.
- Create environments with `scripts/setup_model_envs.sh <project-root> [model]`. Each family has a separate Python 3.11 environment because upstream dependency pins are not mutually compatible.
- `smoke` runs one synthetic CPU forecast per requested model and checks shapes, finiteness, and quantile ordering.
- `run` is resume-safe: it writes a dataset/model completion marker only after its Parquet file is present and non-empty.
- RTX execution uses one Slurm job and exactly one GPU. Models run in the manifest's serialized order; rerunning the job skips validated units.
- A Slurm job is never submitted automatically. The operator fills partition/account/QoS in `config/slurm.yaml`, checks the allocation, then runs the printed `sbatch` command.
