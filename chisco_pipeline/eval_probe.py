"""Held-out linear-probe sanity check, inserted between self-supervised
pretraining (encoder.py + losses.py) and the text-alignment/retrieval stage
(text_align.py). Not in the original spec's Part 5 — added because before
investing in the cross-attention alignment head, you want cheap evidence the
encoder learned *something* class-discriminative.

*** GO/NO-GO GATE — READ BEFORE SCALING UP ***
Train and evaluate this probe on a SMALL slice of CHISCO first: a handful of
subjects/runs, CPU or a single local GPU, fast iteration.

CORRECTION vs. the original "1% top-1" framing: real CHISCO (verified
against a downloaded chunk, see dataloader.py) has no fixed word/category
label -- each trial's ground truth is a free-form sentence ("text"), and the
paper's stimulus set is ~6,000 distinct phrases across 39 semantic
categories. There is no single universal "chance level" -- it depends on how
many distinct labels are actually present in whatever slice you probe on
(e.g. with N distinct sentences in a run/subject, chance top-1 is ~1/N, which
for a 28-trial single-run smoke test is ~3.6%, not ~1%). Compute the actual
class count for whatever `labels` tensor you pass in and compare against
that, not a fixed percentage. Only once accuracy is clearly and repeatably
above the *actual* chance level for your current label set should you move
to Modal or the RTX 6000 for full-dataset training with the real CUDA
`mamba-ssm` kernels — scaling up a pipeline that hasn't cleared its own
chance level just burns compute reproducing the same failure at higher
resolution.
"""

from __future__ import annotations

import torch
from torch import nn


class LinearProbeEvaluator(nn.Module):
    """Frozen brain encoder + a single trainable linear layer + a held-out
    top-1 classification metric.

    Shapes
    ------
    tokens: (B, T, d_model)   -- output of SincNetMambaEncoder (encoder frozen)
    pooled: (B, d_model)       -- mean-pooled over the token axis
    logits: (B, n_classes)     -- one logit per RSVP word/class label
    """

    def __init__(self, encoder: nn.Module, d_model: int, n_classes: int):
        super().__init__()
        self.encoder = encoder
        for p in self.encoder.parameters():
            p.requires_grad_(False)
        self.encoder.eval()
        self.classifier = nn.Linear(d_model, n_classes)

    def forward(self, cortical_eeg: torch.Tensor) -> torch.Tensor:
        """cortical_eeg: (B, C_cortical, T_raw) -> logits: (B, n_classes)"""
        with torch.no_grad():
            tokens = self.encoder(cortical_eeg)  # (B, T, d_model)
        pooled = tokens.mean(dim=1)  # (B, d_model)
        return self.classifier(pooled)

    @torch.no_grad()
    def top1_accuracy(self, cortical_eeg: torch.Tensor, labels: torch.Tensor) -> float:
        """labels: (B,) long tensor of class indices. Returns held-out top-1
        accuracy as a python float in [0, 1]."""
        logits = self.forward(cortical_eeg)  # (B, n_classes)
        preds = logits.argmax(dim=-1)
        return (preds == labels).float().mean().item()
