"""Dynamic spatial whitening front-end.

Uses the dedicated EOG channels (physically distant from cortex, so their
signal is dominated by ocular artifact rather than cognition) to build a
per-batch noise covariance estimate, then projects the cortical channels to
be statistically orthogonal to that noise subspace before they reach the
temporal backbone.

CHANNEL CORRECTION vs. original spec: real CHISCO (verified against a
downloaded chunk, see dataloader.py) has NO usable EMG channel -- it was
dropped upstream by the dataset authors' own preprocessing before the
released derivative was produced. The only artifact-reference signal
available is 2 EOG channels (VEO, HEO). This module is written generically
(it just takes an index tensor for the noise channels), so it works
unchanged with 2 noise channels instead of the spec's assumed EOG+EMG set --
but callers must pass `CHISCODataloader.eog_indices()`, not an EOG+EMG pair.

NOTE ON TERMINOLOGY: "orthogonal projection ... using a learnable spatial
whitening layer" conflates two related-but-distinct operations. What's
implemented here is a regression-based artifact-subtraction (a generalized
least-squares "regress-out", the same family as EOG-regression methods used
in classical EEG preprocessing), not a literal whitening transform (which
would normalize the *cortical* channels' own covariance to identity). True
whitening of the noise subspace is applied to compute the projection
coefficients; the cortical signal itself is left at its native scale after
artifact removal. This is the correct reading of "orthogonal to the noise
covariance," and is called out explicitly since the spec's two clauses
("whitening" + "orthogonal projection") don't literally describe the same
operator.
"""

from __future__ import annotations

import torch
from torch import nn


class SpatialWhiteningFrontEnd(nn.Module):
    """Per-batch dynamic EOG artifact projection.

    Shapes
    ------
    Input  eeg: (B, C_total, T)
    Output cortical_clean: (B, C_cortical, T)

    Where C_total = C_cortical + C_noise. For real CHISCO, C_cortical=122,
    C_noise=2 (VEO, HEO -- no EMG channel exists). C_cortical/C_noise are
    fixed at construction time via the index tensors from
    `CHISCODataloader.cortical_indices()` / `.eog_indices()`.
    """

    def __init__(
        self,
        cortical_indices: torch.Tensor,
        eog_emg_indices: torch.Tensor,
        eps: float = 1e-4,
    ) -> None:
        super().__init__()
        self.register_buffer("cortical_indices", cortical_indices.long())
        self.register_buffer("eog_emg_indices", eog_emg_indices.long())
        self.eps = eps

        n_cortical = cortical_indices.numel()
        n_noise = eog_emg_indices.numel()

        # Learnable projection coefficients mapping the whitened noise
        # subspace onto each cortical channel's expected artifact
        # contribution. Initialized near zero so at the start of training the
        # front-end is close to a no-op (identity pass-through of raw
        # cortical channels), letting the backbone see real signal
        # immediately rather than an untrained, potentially destructive
        # projection.
        self.artifact_coupling = nn.Parameter(0.01 * torch.randn(n_cortical, n_noise))

    def forward(self, eeg: torch.Tensor) -> torch.Tensor:
        """
        eeg: (B, C_total, T) -> (B, C_cortical, T)

        Steps (per batch element, vectorized over B):
          1. Split into cortical (B, C_cortical, T) and noise (B, C_noise, T).
          2. Estimate the noise channels' batch covariance (C_noise, C_noise)
             and whiten them: noise_w = Cov^{-1/2} @ noise.
          3. Subtract the learned projection of whitened noise from the
             cortical channels: cortical_clean = cortical - W @ noise_w.
        """
        cortical = eeg[:, self.cortical_indices, :]  # (B, C_cortical, T)
        noise = eeg[:, self.eog_emg_indices, :]  # (B, C_noise, T)

        B, C_noise, T = noise.shape
        noise_centered = noise - noise.mean(dim=-1, keepdim=True)

        # Batch noise covariance over pooled (B*T) samples: (C_noise, C_noise).
        # Pooling across the batch is what makes this a *dynamic batch* noise
        # profile (spec: "compute a dynamic batch noise covariance matrix"),
        # as opposed to a fixed, dataset-level covariance.
        flat = noise_centered.permute(1, 0, 2).reshape(C_noise, -1)  # (C_noise, B*T)
        cov = (flat @ flat.T) / max(flat.shape[1] - 1, 1)  # (C_noise, C_noise)
        cov = cov + self.eps * torch.eye(C_noise, device=cov.device, dtype=cov.dtype)

        # Symmetric inverse-square-root via eigendecomposition -> whitening matrix.
        eigvals, eigvecs = torch.linalg.eigh(cov)
        eigvals = eigvals.clamp_min(self.eps)
        inv_sqrt = eigvecs @ torch.diag(eigvals.rsqrt()) @ eigvecs.T  # (C_noise, C_noise)

        noise_whitened = torch.einsum("ij,bjt->bit", inv_sqrt, noise_centered)  # (B, C_noise, T)

        artifact_estimate = torch.einsum(
            "oc,bct->bot", self.artifact_coupling, noise_whitened
        )  # (B, C_cortical, T)

        return cortical - artifact_estimate
