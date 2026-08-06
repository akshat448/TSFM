#!/usr/bin/env python3
"""
Parallel standardization orchestrator for the TSFM benchmark.
Processes all datasets concurrently using ProcessPoolExecutor.

Usage:
    python scripts/standardize/run_standardize.py --all
    python scripts/standardize/run_standardize.py --dataset ETTh1 electricity
    python scripts/standardize/run_standardize.py --family ltsf
    python scripts/standardize/run_standardize.py --list
"""
from __future__ import annotations

import argparse
import logging
import sys
from concurrent.futures import ProcessPoolExecutor, as_completed
from pathlib import Path

# Add repo root to path
repo_root = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(repo_root))

from scripts.standardize.base import load_paths
from scripts.standardize.ltsf_standardizer import (
    build_ett_standardizer,
    build_electricity_standardizer,
    build_traffic_standardizer,
    build_weather_standardizer,
    build_solar_standardizer,
    build_ili_standardizer,
    build_exchange_rate_standardizer,
)
from scripts.standardize.pems_standardizer import build_pems_standardizer
from scripts.standardize.gift_eval_standardizer import build_gift_eval_standardizer
from scripts.standardize.monash_standardizer import build_monash_standardizer
from scripts.standardize.ptbxl_standardizer import build_ptbxl_standardizer

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
)
logger = logging.getLogger("run_standardize")


def _build_ett(dataset_name: str, paths: dict):
    return build_ett_standardizer(dataset_name, paths)

def _build_pems(dataset_name: str, paths: dict):
    return build_pems_standardizer(dataset_name, paths)

def get_all_standardizers(paths: dict):
    """Return list of (name, builder_func) tuples."""
    standardizers = []

    # LTSF family — no lambdas, picklable
    for ds in ["ETTh1", "ETTh2", "ETTm1", "ETTm2"]:
        standardizers.append((ds, lambda p, d=ds: _build_ett(d, p)))
    standardizers.append(("electricity", build_electricity_standardizer))
    standardizers.append(("traffic", build_traffic_standardizer))
    standardizers.append(("weather", build_weather_standardizer))
    standardizers.append(("solar_energy", build_solar_standardizer))

    # PEMS family
    for ds in ["PEMS03", "PEMS04", "PEMS08"]:
        standardizers.append((ds, lambda p, d=ds: _build_pems(d, p)))

    # Archive family
    standardizers.append(("gift_eval", build_gift_eval_standardizer))
    standardizers.append(("monash", build_monash_standardizer))

    # Classification
    standardizers.append(("ptbxl", build_ptbxl_standardizer))

    # Optional / Tier 3
    try:
        standardizers.append(("ili", build_ili_standardizer))
    except FileNotFoundError:
        logger.warning("ILI raw data not found, skipping")
    try:
        standardizers.append(("exchange_rate", build_exchange_rate_standardizer))
    except FileNotFoundError:
        logger.warning("Exchange Rate raw data not found, skipping")

    return standardizers

def run_single(name: str, builder, paths: dict) -> tuple[str, str | None]:
    """Run a single standardizer. Returns (name, error_or_none)."""
    try:
        std = builder(paths)
        std.run()
        return name, None
    except Exception as e:
        logger.error(f"[{name}] FAILED: {e}", exc_info=True)
        return name, str(e)


def main():
    parser = argparse.ArgumentParser(description="Standardize TSFM benchmark datasets")
    parser.add_argument("--all", action="store_true", help="Process all datasets")
    parser.add_argument("--dataset", nargs="+", help="Process specific datasets")
    parser.add_argument("--family", choices=["ltsf", "pems", "archive", "classification", "all"], help="Process by family")
    parser.add_argument("--list", action="store_true", help="List available datasets")
    parser.add_argument("--workers", type=int, default=4, help="Number of parallel workers")
    parser.add_argument("--sequential", action="store_true", help="Run sequentially (no parallelism)")
    args = parser.parse_args()

    paths = load_paths()
    all_std = get_all_standardizers(paths)

    if args.list:
        print("Available datasets:")
        for name, _ in all_std:
            print(f"  - {name}")
        return

    # Determine which to run
    to_run = []
    if args.all:
        to_run = all_std
    elif args.dataset:
        name_map = {n: b for n, b in all_std}
        for ds in args.dataset:
            if ds in name_map:
                to_run.append((ds, name_map[ds]))
            else:
                logger.warning(f"Unknown dataset: {ds}")
    elif args.family:
        family_map = {
            "ltsf": ["ETTh1", "ETTh2", "ETTm1", "ETTm2", "electricity", "traffic", "weather", "solar_energy", "ili", "exchange_rate"],
            "pems": ["PEMS03", "PEMS04", "PEMS08"],
            "archive": ["gift_eval", "monash"],
            "classification": ["ptbxl"],
            "all": [n for n, _ in all_std],
        }
        names = family_map[args.family]
        name_map = {n: b for n, b in all_std}
        to_run = [(n, name_map[n]) for n in names if n in name_map]
    else:
        parser.print_help()
        return

    if not to_run:
        logger.warning("No datasets selected to process")
        return

    logger.info(f"Processing {len(to_run)} dataset(s) with {args.workers} workers...")

    if args.sequential:
        results = []
        for name, builder in to_run:
            results.append(run_single(name, builder, paths))
    else:
        with ProcessPoolExecutor(max_workers=args.workers) as executor:
            futures = {executor.submit(run_single, name, builder, paths): name for name, builder in to_run}
            results = []
            for future in as_completed(futures):
                name = futures[future]
                try:
                    result = future.result()
                    results.append(result)
                except Exception as e:
                    logger.error(f"[{name}] CRASHED: {e}")
                    results.append((name, str(e)))

    # Summary
    print("\n" + "=" * 70)
    print("STANDARDIZATION SUMMARY")
    print("=" * 70)
    ok = [n for n, e in results if e is None]
    fail = [(n, e) for n, e in results if e is not None]
    print(f"  Success: {len(ok)}/{len(results)}")
    for n in ok:
        print(f"    [OK]   {n}")
    if fail:
        print(f"  Failed:  {len(fail)}/{len(results)}")
        for n, e in fail:
            print(f"    [FAIL] {n}: {e}")
    print("=" * 70)


if __name__ == "__main__":
    main()
