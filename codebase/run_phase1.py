#!/usr/bin/env python3
"""
Phase 1 eval entrypoint. Reads config/phase1_eval.yaml (what to run) and
config/paths.yaml (where the data actually lives, the project's existing
convention, never hardcoded), runs each dataset x model combination, and
writes one raw results file per dataset under <output_dir>/<dataset>.parquet.

Usage:
    python run_phase1.py --eval-config config/phase1_eval.yaml --paths config/paths.yaml
    python run_phase1.py --eval-config config/phase1_eval.yaml --paths config/paths.yaml --dataset ETTh1 --model dlinear

Reproduction gate (PHASE1_PLAN.md milestone M1): run this on ETTh1 with
seasonal_naive + dlinear only first, and manually compare the DLinear MSE at
each horizon against the published DLinear/LTSF-Linear numbers before running
anything else. If it doesn't come close, stop and debug the harness, don't
proceed to more datasets or models on an unverified pipeline.
"""
from __future__ import annotations

import argparse
import os
import sys
from pathlib import Path

import pandas as pd
import yaml

sys.path.insert(0, str(Path(__file__).parent / "scripts"))

from harness.contamination import ContaminationMatrix
from harness.eval_runner import run_zero_shot_ltsf, run_trained_ltsf
from harness.registry import try_register_wave2


def load_yaml(path: str) -> dict:
    with open(path) as f:
        return yaml.safe_load(f)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--eval-config", default="config/phase1_eval.yaml")
    parser.add_argument("--paths", default="config/paths.yaml")
    parser.add_argument("--contamination-matrix", default="config/contamination_matrix.yaml")
    parser.add_argument("--dataset", default=None, help="run only this dataset, default: all listed in eval-config")
    parser.add_argument("--model", default=None, help="run only this model, default: all listed in eval-config")
    args = parser.parse_args()

    eval_cfg = load_yaml(args.eval_config)
    paths_cfg = load_yaml(args.paths)
    contamination = ContaminationMatrix(args.contamination_matrix)

    missing_wave2 = try_register_wave2()
    if missing_wave2:
        print(f"[info] Wave 2 adapters not registered (pip package missing), skipping if requested: {missing_wave2}")

    data_root = paths_cfg.get("data_root", ".")
    output_dir = Path(eval_cfg.get("output_dir", "results/phase1"))
    output_dir.mkdir(parents=True, exist_ok=True)

    dataset_names = [args.dataset] if args.dataset else list(eval_cfg["datasets"].keys())
    model_names = [args.model] if args.model else list(eval_cfg["models"].keys())

    for dataset_name in dataset_names:
        ds_cfg = eval_cfg["datasets"][dataset_name]
        parquet_path = os.path.join(data_root, ds_cfg["parquet_path"]) if not os.path.isabs(ds_cfg["parquet_path"]) else ds_cfg["parquet_path"]
        if not os.path.exists(parquet_path):
            print(f"[skip] {dataset_name}: parquet not found at {parquet_path}, check paths.yaml's data_root")
            continue

        all_results = []
        for model_name in model_names:
            model_cfg = eval_cfg["models"][model_name]
            kwargs = dict(model_cfg.get("kwargs", {}))
            if model_name == "seasonal_naive":
                kwargs["season_length"] = ds_cfg["season_length"]

            print(f"[run] dataset={dataset_name} model={model_name} kind={model_cfg['kind']}")
            try:
                if model_cfg["kind"] == "zero_shot":
                    df = run_zero_shot_ltsf(
                        dataset_name, parquet_path, model_name, kwargs, contamination,
                        horizons=ds_cfg["horizons"], season_length_for_mase=ds_cfg["season_length"],
                    )
                elif model_cfg["kind"] == "trained":
                    df = run_trained_ltsf(
                        dataset_name, parquet_path, model_name, kwargs, contamination,
                        horizons=ds_cfg["horizons"], season_length_for_mase=ds_cfg["season_length"],
                    )
                else:
                    raise ValueError(f"unknown model kind {model_cfg['kind']!r} for {model_name}")
            except KeyError as e:
                print(f"[skip] {dataset_name}/{model_name}: {e} (adapter not registered, install its pip package first)")
                continue

            all_results.append(df)
            print(f"       -> {len(df)} rows, mean MSE {df['mse'].mean():.4f}, mean MASE {df['mase'].mean():.4f}")

        if all_results:
            combined = pd.concat(all_results, ignore_index=True)
            out_path = output_dir / f"{dataset_name}.parquet"
            combined.to_parquet(out_path)
            print(f"[write] {out_path} ({len(combined)} rows)")


if __name__ == "__main__":
    main()
