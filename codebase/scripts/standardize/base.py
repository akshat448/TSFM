"""
Base standardizer interface and shared utilities.
"""
from __future__ import annotations

import logging
from abc import ABC, abstractmethod
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
import yaml

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
)
logger = logging.getLogger("standardize")


def load_paths() -> dict[str, str]:
    repo_root = Path(__file__).resolve().parents[2]
    paths_file = repo_root / "config" / "paths.yaml"
    with open(paths_file) as f:
        return yaml.safe_load(f)


class BaseStandardizer(ABC):
    """Abstract base for all dataset standardizers."""

    def __init__(self, paths: dict[str, str] | None = None):
        self.paths = paths or load_paths()
        self.raw_dir = Path(self.paths["raw_dir"])
        self.processed_dir = Path(self.paths["processed_dir"])
        self.manifest_dir = Path(self.paths.get("manifest_dir", self.paths["raw_dir"].replace("/raw", "/manifests")))
        self.processed_dir.mkdir(parents=True, exist_ok=True)
        self.manifest_dir.mkdir(parents=True, exist_ok=True)

    @property
    @abstractmethod
    def name(self) -> str:
        ...

    @abstractmethod
    def standardize(self) -> pd.DataFrame:
        """Return the standardized DataFrame."""
        ...

    def save(self, df: pd.DataFrame, subpath: str) -> Path:
        out = self.processed_dir / subpath
        out.parent.mkdir(parents=True, exist_ok=True)
        df.to_parquet(out, index=False, engine="pyarrow")
        logger.info(f"[{self.name}] Saved {len(df)} rows → {out}")
        return out

    def run(self) -> Path:
        logger.info(f"[{self.name}] Starting standardization...")
        df = self.standardize()
        subpath = self.output_subpath
        out = self.save(df, subpath)
        self._write_manifest(df, out)
        return out

    def _write_manifest(self, df: pd.DataFrame, out_path: Path) -> None:
        manifest = {
            "dataset": self.name,
            "output": str(out_path),
            "rows": len(df),
            "columns": list(df.columns),
            "dtypes": {c: str(df[c].dtype) for c in df.columns},
        }
        mpath = self.manifest_dir / f"{self.name}_standardize.json"
        import json
        with open(mpath, "w") as f:
            json.dump(manifest, f, indent=2, default=str)
        logger.info(f"[{self.name}] Manifest → {mpath}")

    @property
    @abstractmethod
    def output_subpath(self) -> str:
        ...
