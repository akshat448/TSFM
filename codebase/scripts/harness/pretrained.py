"""Shared, dependency-light helpers for pretrained-TSFM benchmark runs."""
from __future__ import annotations

import hashlib
import json
import os
import platform
import sys
from datetime import datetime, timezone
from pathlib import Path


REQUIRED_MODEL_FIELDS = {"checkpoint", "adapter"}


def validate_model_manifest(manifest: dict) -> None:
    models = manifest.get("models")
    if not isinstance(models, dict) or not models:
        raise ValueError("models manifest must contain a non-empty 'models' mapping")
    for name, spec in models.items():
        if not isinstance(spec, dict):
            raise ValueError(f"{name}: model specification must be a mapping")
        for field in REQUIRED_MODEL_FIELDS:
            if not spec.get(field):
                raise ValueError(f"{name}: missing required {field!r}")


def select_evenly_spaced(origins: list[int], limit: int) -> list[int]:
    """Return at most ``limit`` deterministic, endpoint-preserving origins."""
    if limit < 1:
        raise ValueError("origin limit must be >= 1")
    if len(origins) <= limit:
        return origins
    if limit == 1:
        return [origins[-1]]
    last = len(origins) - 1
    indexes = [(i * last) // (limit - 1) for i in range(limit)]
    return [origins[i] for i in indexes]


def _marker_path(result_dir: Path, model: str, dataset: str) -> Path:
    return result_dir / f"{model}__{dataset}.done.json"


def unit_complete(result_dir: Path, model: str, dataset: str) -> bool:
    marker = _marker_path(result_dir, model, dataset)
    try:
        contents = json.loads(marker.read_text())
    except (FileNotFoundError, json.JSONDecodeError):
        return False
    return contents.get("model") == model and contents.get("dataset") == dataset and int(contents.get("row_count", 0)) > 0


def write_completion_marker(result_dir: Path, model: str, dataset: str, row_count: int) -> Path:
    if row_count < 1:
        raise ValueError("cannot mark an empty result unit complete")
    result_dir.mkdir(parents=True, exist_ok=True)
    marker = _marker_path(result_dir, model, dataset)
    temporary = marker.with_suffix(".tmp")
    temporary.write_text(json.dumps({"model": model, "dataset": dataset, "row_count": row_count}, indent=2) + "\n")
    os.replace(temporary, marker)
    return marker


def build_run_manifest(run_id: str, config_bytes: bytes, checkpoints: dict, device: str) -> dict:
    return {
        "run_id": run_id,
        "created_at_utc": datetime.now(timezone.utc).isoformat(),
        "config_sha256": hashlib.sha256(config_bytes).hexdigest(),
        "checkpoints": checkpoints,
        "device": device,
        "python": sys.version,
        "platform": platform.platform(),
    }


def render_sbatch_script(slurm: dict, models: list[str]) -> str:
    """Render one intentionally serialized, one-GPU benchmark job."""
    if not models:
        raise ValueError("at least one model is required")
    lines = [
        "#!/usr/bin/env bash",
        "#SBATCH --job-name=tsfm-pretrained",
        "#SBATCH --gpus=1",
        f"#SBATCH --cpus-per-task={int(slurm.get('cpus_per_task', 8))}",
        f"#SBATCH --mem={slurm.get('memory', '64G')}",
        f"#SBATCH --time={slurm.get('time', '24:00:00')}",
        "#SBATCH --output=logs/pretrained-%j.out",
        "#SBATCH --error=logs/pretrained-%j.err",
    ]
    for key, flag in (("partition", "partition"), ("account", "account"), ("qos", "qos")):
        if slurm.get(key):
            lines.append(f"#SBATCH --{flag}={slurm[key]}")
    lines += [
        "set -euo pipefail",
        'PROJECT_ROOT="${PROJECT_ROOT:?set PROJECT_ROOT before submitting}"',
        'MODEL_ENV_ROOT="${MODEL_ENV_ROOT:-$PROJECT_ROOT/models/envs}"',
        "mkdir -p logs",
        "nvidia-smi",
        'cd "$PROJECT_ROOT/codebase"',
    ]
    for model in models:
        lines.extend([
            f'source "$MODEL_ENV_ROOT/{model}/bin/activate"',
            f"python run_pretrained_models.py run --model {model} --device cuda --resume",
            "deactivate",
        ])
    return "\n".join(lines) + "\n"
