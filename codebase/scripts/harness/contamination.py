"""
Attaches a contamination label to every result row. Reads config/contamination_matrix.yaml
(seeded from the bridge phase's literature review) and defaults to
"undocumented-unknown" for anything not explicitly listed, on purpose,
never leaves a result unlabeled or silently assumes "clean."
"""
from __future__ import annotations

import yaml

VALID_LABELS = {"certified-unseen", "known-overlap", "undocumented-unknown"}


class ContaminationMatrix:
    def __init__(self, yaml_path: str):
        with open(yaml_path) as f:
            self._data = yaml.safe_load(f) or {}
        for dataset, entry in self._data.items():
            default = entry.get("default")
            if default not in VALID_LABELS:
                raise ValueError(f"{dataset}: invalid or missing default label {default!r}")
            for model, label in (entry.get("overrides") or {}).items():
                if label not in VALID_LABELS:
                    raise ValueError(f"{dataset}/{model}: invalid override label {label!r}")

    def label(self, dataset: str, model: str) -> str:
        entry = self._data.get(dataset)
        if entry is None:
            return "undocumented-unknown"
        overrides = entry.get("overrides") or {}
        if model in overrides:
            return overrides[model]
        return entry.get("default", "undocumented-unknown")
