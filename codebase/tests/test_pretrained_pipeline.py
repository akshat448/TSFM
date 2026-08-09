from __future__ import annotations

import json
import subprocess
import sys
import tempfile
import unittest
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "scripts"))

from harness.pretrained import (  # noqa: E402
    build_run_manifest,
    render_sbatch_script,
    select_evenly_spaced,
    unit_complete,
    validate_model_manifest,
)
from harness.models.chronos_bolt import ChronosBoltAdapter  # noqa: E402


class PretrainedPipelineTests(unittest.TestCase):
    def test_manifest_requires_checkpoint_and_adapter(self):
        with self.assertRaisesRegex(ValueError, "checkpoint"):
            validate_model_manifest({"models": {"chronos_bolt": {"adapter": "chronos_bolt"}}})

        with self.assertRaisesRegex(ValueError, "adapter"):
            validate_model_manifest({"models": {"chronos_bolt": {"checkpoint": "amazon/chronos-bolt-base"}}})

    def test_evenly_spaced_origins_preserve_both_bounds(self):
        self.assertEqual(select_evenly_spaced(list(range(10)), 4), [0, 3, 6, 9])
        self.assertEqual(select_evenly_spaced([4, 8], 100), [4, 8])

    def test_completion_marker_requires_matching_unit_identity(self):
        with tempfile.TemporaryDirectory() as tmp:
            result_dir = Path(tmp)
            marker = result_dir / "chronos_bolt__ETTh1.done.json"
            marker.write_text(json.dumps({"model": "chronos_bolt", "dataset": "ETTh1", "row_count": 12}))

            self.assertTrue(unit_complete(result_dir, "chronos_bolt", "ETTh1"))
            self.assertFalse(unit_complete(result_dir, "chronos_bolt", "ETTh2"))

    def test_run_manifest_includes_reproducibility_fields(self):
        manifest = build_run_manifest(
            run_id="smoke-001",
            config_bytes=b"models: {}\n",
            checkpoints={"chronos_bolt": {"repo": "amazon/chronos-bolt-base", "revision": "abc"}},
            device="cpu",
        )

        self.assertEqual(manifest["run_id"], "smoke-001")
        self.assertEqual(manifest["device"], "cpu")
        self.assertEqual(manifest["checkpoints"]["chronos_bolt"]["revision"], "abc")
        self.assertEqual(len(manifest["config_sha256"]), 64)

    def test_sbatch_renderer_enforces_one_gpu_and_serial_model_order(self):
        script = render_sbatch_script(
            {"cpus_per_task": 8, "memory": "64G", "time": "24:00:00"},
            ["chronos_bolt", "timesfm", "chronos_2"],
        )

        self.assertIn("#SBATCH --gpus=1", script)
        self.assertNotIn("#SBATCH --array", script)
        self.assertIn('PROJECT_ROOT="${PROJECT_ROOT:?set PROJECT_ROOT before submitting}"', script)
        self.assertIn('MODEL_ENV_ROOT="${MODEL_ENV_ROOT:-$PROJECT_ROOT/models/envs}"', script)
        self.assertIn('source "$MODEL_ENV_ROOT/chronos_bolt/bin/activate"', script)
        self.assertNotIn("/mnt/hdd1", script)
        self.assertLess(script.index("chronos_bolt"), script.index("timesfm"))
        self.assertLess(script.index("timesfm"), script.index("chronos_2"))

    def test_submit_cli_does_not_require_torch(self):
        root = Path(__file__).resolve().parents[1]
        with tempfile.TemporaryDirectory() as tmp:
            output = Path(tmp) / "job.sbatch"
            completed = subprocess.run(
                [
                    sys.executable,
                    str(root / "run_pretrained_models.py"),
                    "submit",
                    "--manifest", str(root / "config" / "pretrained_models.yaml"),
                    "--paths", str(root / "config" / "paths.yaml"),
                    "--slurm-config", str(root / "config" / "slurm.example.yaml"),
                    "--output", str(output),
                ],
                text=True,
                capture_output=True,
            )
            self.assertEqual(completed.returncode, 0, completed.stderr)
            self.assertIn("sbatch", completed.stdout)
            self.assertTrue(output.exists())

    def test_chronos_bolt_passes_context_as_pipeline_inputs(self):
        class Pipeline:
            def predict_quantiles(self, inputs, prediction_length, quantile_levels):
                import torch
                quantiles = torch.zeros((1, prediction_length, len(quantile_levels)))
                quantiles[:, :, 0] = 2
                quantiles[:, :, 1] = 1
                return (
                    quantiles,
                    torch.zeros((1, prediction_length)),
                )

        adapter = object.__new__(ChronosBoltAdapter)
        adapter._pipeline = Pipeline()
        result = adapter.predict(__import__("numpy").arange(32), horizon=4)
        self.assertEqual(result.point.shape, (4,))
        self.assertTrue((result.quantiles[0.2] >= result.quantiles[0.1]).all())


if __name__ == "__main__":
    unittest.main()
