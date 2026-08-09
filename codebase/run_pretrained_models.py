#!/usr/bin/env python3
"""Stage, smoke-test, run, or render Slurm submission for pretrained TSFMs."""
from __future__ import annotations

import argparse
import json
import subprocess
import sys
from pathlib import Path

import numpy as np
import yaml

sys.path.insert(0, str(Path(__file__).parent / "scripts"))

from harness.pretrained import build_run_manifest, render_sbatch_script, validate_model_manifest


def load_yaml(path: Path) -> dict:
    with path.open() as handle:
        return yaml.safe_load(handle) or {}


def selected_models(manifest: dict, requested: str | None) -> list[str]:
    models = manifest["models"]
    if requested:
        if requested not in models:
            raise ValueError(f"unknown model {requested!r}; choose from {sorted(models)}")
        return [requested]
    return manifest.get("serialized_order", list(models))


def stage(model_name: str, spec: dict, checkpoint_root: Path, cache_dir: Path) -> dict:
    from huggingface_hub import snapshot_download

    target = checkpoint_root / model_name
    target.parent.mkdir(parents=True, exist_ok=True)
    path = snapshot_download(
        repo_id=spec["checkpoint"],
        local_dir=target,
        cache_dir=cache_dir,
    )
    if not any(Path(path).glob("*.safetensors")) and not any(Path(path).glob("pytorch_model*.bin")):
        raise RuntimeError(f"{model_name}: staged checkpoint has no model weights; rerun stage after checking network/HF access")
    from huggingface_hub import HfApi

    revision = HfApi().model_info(spec["checkpoint"]).sha
    return {"repo": spec["checkpoint"], "revision": revision, "local_path": str(path)}


def smoke(model_name: str, spec: dict, checkpoint_root: Path, device: str) -> None:
    from harness.registry import build_forecast_model, register_pretrained_models

    local_path = checkpoint_root / model_name
    checkpoint = str(local_path) if local_path.exists() else spec["checkpoint"]
    register_pretrained_models()
    model = build_forecast_model(model_name, checkpoint=checkpoint, device=device)
    context_len = min(int(spec.get("context_limit", 512)), 512)
    series = (np.sin(np.linspace(0, 16, context_len)) + 10).astype(np.float32)
    result = model.predict(series, horizon=24)
    if result.point.shape != (24,) or not np.isfinite(result.point).all():
        raise RuntimeError(f"{model_name}: invalid point forecast")
    if result.quantiles:
        previous = None
        for q in sorted(result.quantiles):
            values = np.asarray(result.quantiles[q])
            if values.shape != (24,) or not np.isfinite(values).all():
                raise RuntimeError(f"{model_name}: invalid quantile {q}")
            if previous is not None and np.any(values < previous):
                raise RuntimeError(f"{model_name}: non-monotonic quantiles")
            previous = values
    print(f"[smoke] {model_name}: passed on {device}")


def run(model_name: str, spec: dict, paths: Path, eval_config: Path, checkpoint_root: Path, device: str) -> None:
    """Delegate to the established runner with a generated single-model config."""
    config = load_yaml(eval_config)
    if model_name not in config.get("models", {}):
        config.setdefault("models", {})[model_name] = {"kind": "zero_shot", "kwargs": {}}
    config["models"] = {model_name: config["models"][model_name]}
    config["models"][model_name]["kwargs"] = {
        **config["models"][model_name].get("kwargs", {}),
        "checkpoint": str(checkpoint_root / model_name),
        "device": device,
    }
    generated = Path("results") / "pretrained" / f"{model_name}.eval.yaml"
    generated.parent.mkdir(parents=True, exist_ok=True)
    generated.write_text(yaml.safe_dump(config, sort_keys=False))
    subprocess.run(
        [sys.executable, "run_phase1.py", "--eval-config", str(generated), "--paths", str(paths), "--model", model_name],
        check=True,
    )


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("action", choices=("stage", "smoke", "run", "submit"))
    parser.add_argument("--model")
    parser.add_argument("--device", default="cpu")
    parser.add_argument("--manifest", default="config/pretrained_models.yaml")
    parser.add_argument("--paths", default="config/paths.yaml")
    parser.add_argument("--checkpoint-root", help="override checkpoint_root from paths.yaml")
    parser.add_argument("--cache-dir", help="override huggingface_cache from paths.yaml")
    parser.add_argument("--eval-config", default="config/phase1_eval.yaml")
    parser.add_argument("--slurm-config", default="config/slurm.yaml")
    parser.add_argument("--output", default="scripts/slurm/run_pretrained_serial.sbatch")
    parser.add_argument("--resume", action="store_true", help="accepted for serialized retry compatibility")
    args = parser.parse_args()

    manifest_path, paths_path = Path(args.manifest), Path(args.paths)
    manifest, paths = load_yaml(manifest_path), load_yaml(paths_path)
    validate_model_manifest(manifest)
    checkpoint_root = Path(args.checkpoint_root or paths["checkpoint_root"])
    cache_dir = Path(args.cache_dir or paths["huggingface_cache"])
    models = selected_models(manifest, args.model)

    if args.action == "stage":
        staged = {name: stage(name, manifest["models"][name], checkpoint_root, cache_dir) for name in models}
        print(json.dumps(staged, indent=2))
    elif args.action == "smoke":
        for name in models:
            smoke(name, manifest["models"][name], checkpoint_root, args.device)
    elif args.action == "run":
        for name in models:
            run(name, manifest["models"][name], paths_path, Path(args.eval_config), checkpoint_root, args.device)
    else:
        slurm_path = Path(args.slurm_config)
        if not slurm_path.exists():
            raise SystemExit(f"copy config/slurm.example.yaml to {slurm_path} and fill the site settings first")
        output = Path(args.output)
        output.parent.mkdir(parents=True, exist_ok=True)
        output.write_text(render_sbatch_script(load_yaml(slurm_path), models))
        print(f"wrote {output}; submit manually with: sbatch {output}")


if __name__ == "__main__":
    main()
