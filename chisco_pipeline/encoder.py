"""SincNet front-end + 20ms tokenizer + causal Mamba (S6) backbone.

BACKEND NOTE: the official `mamba-ssm` package ships CUDA-only selective-scan
kernels and will not build/run on this machine (macOS/Darwin, no CUDA). The
`CausalSelectiveSSM` below is a from-scratch pure-PyTorch implementation of
the same recurrence (diagonal state-space model with input-dependent
discretization, i.e. the "S6" mechanism Mamba is built on). It is a
sequential per-timestep loop, so it is slower than the fused CUDA kernel, but
it is numerically equivalent in spirit and runs unmodified on CPU/MPS here
and later on a CUDA box. Swap in real `mamba-ssm` blocks later purely as a
speed optimization if desired — the surrounding architecture doesn't change.

SCALE NOTE: everything in this module is meant to be exercised first on a
small local/synthetic slice of CHISCO (a handful of subjects, CPU or a single
GPU) so shapes, gradients, and the loss dynamics can be debugged cheaply and
quickly. Only once the `LinearProbeEvaluator` (see eval_probe.py) shows
held-out top-1 accuracy clearly above the ~1% chance-level floor for a
reasonably-sized RSVP vocabulary should this move to a bigger run (Modal or
the RTX 6000 you have access to) with the full dataset and the real CUDA
`mamba-ssm` kernels. Debugging a slow, wrong pipeline on expensive hardware
is a waste of that hardware; getting >1% top-1 locally first is the gate.
"""

from __future__ import annotations

import math

import torch
from torch import nn


class SincConv1d(nn.Module):
    """Learnable band-pass filterbank (SincNet), replacing fixed filter banks
    or a continuous-wavelet-transform front-end per spec.

    Each output channel is a band-pass filter parameterized by a learnable
    (low_hz, band_hz) pair; the filter itself is the analytic sinc-difference
    kernel, windowed by a Hamming window. Gradients flow into the two cutoff
    parameters directly, so the network learns *which frequency bands matter*
    rather than convolving with a fixed basis.

    Shapes
    ------
    Input  x: (B, 1, T)          -- single raw channel
    Output y: (B, n_filters, T') -- T' depends on stride/padding (see forward)
    """

    def __init__(
        self,
        n_filters: int,
        kernel_size: int,
        sample_rate_hz: float,
        min_low_hz: float = 0.6,
        min_band_hz: float = 1.0,
        stride: int = 1,
    ) -> None:
        super().__init__()
        if kernel_size % 2 == 0:
            kernel_size += 1  # enforce odd kernel for a symmetric, zero-centered sinc
        self.n_filters = n_filters
        self.kernel_size = kernel_size
        self.sample_rate_hz = sample_rate_hz
        self.stride = stride
        self.min_low_hz = min_low_hz
        self.min_band_hz = min_band_hz

        nyquist = sample_rate_hz / 2.0
        # Spec: "Initialize the lowest frequency cutoff strictly at >0.5 Hz."
        # min_low_hz defaults to 0.6 Hz to satisfy that strict inequality with
        # margin; low_hz_ below is initialized at min_low_hz, i.e. the very
        # first filter's low edge starts at 0.6 Hz > 0.5 Hz.
        low_hz = torch.linspace(min_low_hz, nyquist - (n_filters + 1) * min_band_hz, n_filters)
        band_hz = torch.full((n_filters,), (nyquist - min_low_hz) / n_filters)

        self.low_hz_ = nn.Parameter(low_hz.unsqueeze(1))  # (n_filters, 1)
        self.band_hz_ = nn.Parameter(band_hz.unsqueeze(1))  # (n_filters, 1)

        # Precompute the symmetric time axis (excludes t=0, handled analytically)
        # and the Hamming window, both non-learnable.
        half = (kernel_size - 1) // 2
        n = torch.arange(-half, half + 1).float()
        self.register_buffer("n_", n / sample_rate_hz)  # (kernel_size,) in seconds
        window = torch.hamming_window(kernel_size, periodic=False)
        self.register_buffer("window_", window)

    def _sinc_filters(self) -> torch.Tensor:
        """Build the (n_filters, 1, kernel_size) band-pass kernels for this
        forward pass, from the current low/band parameters."""
        low = self.min_low_hz + torch.abs(self.low_hz_)  # (n_filters, 1)
        high = torch.clamp(
            low + self.min_band_hz + torch.abs(self.band_hz_),
            self.min_low_hz,
            self.sample_rate_hz / 2.0,
        )

        n = self.n_.unsqueeze(0)  # (1, kernel_size)
        half = (self.kernel_size - 1) // 2
        n_no_zero = n.clone()
        n_no_zero[:, half] = 1.0  # avoid /0 at the center tap; overwritten below

        def sinc_band(edge_hz: torch.Tensor) -> torch.Tensor:
            # sin(2*pi*f*t) / (pi*t), with the t=0 limit = 2*f
            arg = 2 * math.pi * edge_hz * n_no_zero
            band = torch.sin(arg) / (math.pi * n_no_zero)
            band[:, half] = 2 * edge_hz.squeeze(1)
            return band

        band_pass = sinc_band(high) - sinc_band(low)  # (n_filters, kernel_size)
        band_pass = band_pass * self.window_.unsqueeze(0)
        band_pass = band_pass / (band_pass.abs().amax(dim=1, keepdim=True) + 1e-8)
        return band_pass.unsqueeze(1)  # (n_filters, 1, kernel_size)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """x: (B, 1, T) -> (B, n_filters, T') ; T' = floor((T - kernel_size)/stride) + 1
        with 'same'-ish padding applied so T' ~= ceil(T/stride)."""
        filters = self._sinc_filters().to(dtype=x.dtype)
        padding = self.kernel_size // 2
        return torch.nn.functional.conv1d(x, filters, stride=self.stride, padding=padding)


class TemporalTokenizer(nn.Module):
    """Groups the SincNet feature stream into fixed-duration (20ms) tokens.

    Rationale (spec Part 3): "Tokenize the incoming continuous signal into
    20ms chunks to capture local wave propagation without discarding temporal
    resolution" -- i.e. coarsen enough to make the sequence length tractable
    for the SSM, but keep each token short enough that within-token wave
    propagation across channels isn't averaged away.

    Shapes
    ------
    Input  x: (B, n_filters, T)          -- T at `sample_rate_hz` (post-SincNet)
    Output tokens: (B, n_tokens, d_model) -- n_tokens = T // samples_per_token
    """

    def __init__(self, n_filters: int, d_model: int, sample_rate_hz: float, token_ms: float = 20.0):
        super().__init__()
        self.samples_per_token = max(1, round(sample_rate_hz * token_ms / 1000.0))
        self.proj = nn.Conv1d(
            n_filters, d_model, kernel_size=self.samples_per_token, stride=self.samples_per_token
        )
        self.norm = nn.LayerNorm(d_model)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """x: (B, n_filters, T) -> (B, n_tokens, d_model)"""
        n_tokens = x.shape[-1] // self.samples_per_token
        x = x[..., : n_tokens * self.samples_per_token]
        tokens = self.proj(x)  # (B, d_model, n_tokens)
        tokens = tokens.transpose(1, 2)  # (B, n_tokens, d_model)
        return self.norm(tokens)


class CausalSelectiveSSM(nn.Module):
    """Pure-PyTorch causal selective state-space layer (the S6 recurrence
    Mamba is built on), diagonal in the state dimension for tractability.

    For each channel d and state index n, with input-dependent discretization
    step delta_t (softplus of a learned projection of x_t):
        A_bar_t = exp(delta_t * A_d)                      (decay, per-channel)
        B_bar_t = delta_t * B_t                            (input-dependent)
        h_t     = A_bar_t * h_{t-1} + B_bar_t * x_t         (state update)
        y_t     = C_t . h_t + D_d * x_t                     (output read-out)
    A is initialized negative (stable decay) and kept negative via
    -softplus, so the recurrence cannot blow up regardless of what the input-
    dependent projections learn.

    Shapes
    ------
    Input  x: (B, T, d_model)
    Output y: (B, T, d_model)
    """

    def __init__(self, d_model: int, d_state: int = 16):
        super().__init__()
        self.d_model = d_model
        self.d_state = d_state

        self.A_log = nn.Parameter(torch.log(torch.rand(d_model, d_state) * 0.9 + 0.1))
        self.D = nn.Parameter(torch.ones(d_model))

        self.x_proj = nn.Linear(d_model, d_state * 2 + d_model, bias=False)  # -> B_t, C_t, delta_raw
        self.dt_bias = nn.Parameter(torch.zeros(d_model))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """x: (B, T, d_model) -> (B, T, d_model). Sequential scan over T."""
        B_batch, T, D = x.shape
        N = self.d_state

        proj = self.x_proj(x)  # (B, T, 2N + D)
        B_t, C_t, delta_raw = torch.split(proj, [N, N, D], dim=-1)
        delta = torch.nn.functional.softplus(delta_raw + self.dt_bias)  # (B, T, D)

        A = -torch.exp(self.A_log)  # (D, N), strictly negative -> stable decay

        h = x.new_zeros(B_batch, D, N)
        ys = []
        for t in range(T):
            delta_t = delta[:, t, :]  # (B, D)
            A_bar = torch.exp(delta_t.unsqueeze(-1) * A.unsqueeze(0))  # (B, D, N)
            Bx = delta_t.unsqueeze(-1) * B_t[:, t, :].unsqueeze(1) * x[:, t, :].unsqueeze(-1)  # (B, D, N)
            h = A_bar * h + Bx
            y_t = torch.einsum("bdn,bn->bd", h, C_t[:, t, :]) + self.D * x[:, t, :]
            ys.append(y_t)
        return torch.stack(ys, dim=1)  # (B, T, D)


class CausalMambaBlock(nn.Module):
    """One pre-norm Mamba-style residual block: LN -> gated MLP-in ->
    selective SSM -> gated MLP-out -> residual add. Purely causal by
    construction (the SSM only ever reads h_{<=t}).

    Shapes: (B, T, d_model) -> (B, T, d_model)

    backend="pure_pytorch" (default): uses `CausalSelectiveSSM` above, a
    from-scratch sequential-loop implementation. Runs anywhere (CPU/MPS/CUDA)
    but is not fused/parallelized -- fine for local dev and smoke tests, slow
    at real training scale.

    backend="mamba_ssm": delegates to the real `mamba_ssm.Mamba` CUDA kernel
    (https://github.com/state-spaces/mamba). This is the fast path the
    RTX 6000 / Modal scale-up gate (see __init__.py, eval_probe.py) was about
    -- only usable where `mamba_ssm` is installed AND a CUDA GPU is present;
    raises immediately at construction otherwise rather than silently falling
    back, so a misconfigured cluster job fails fast instead of quietly
    training 100x slower than intended. `mamba_ssm.Mamba` already implements
    the full gated in_proj/conv/ssm/out_proj block itself (this class's
    `self.in_proj`/`self.ssm`/`self.out_proj` are unused in this mode) -- only
    the surrounding pre-norm + residual wrapper is shared between backends.
    """

    def __init__(self, d_model: int, d_state: int = 16, expand: int = 2, backend: str = "pure_pytorch"):
        super().__init__()
        self.backend = backend
        self.norm = nn.LayerNorm(d_model)

        if backend == "pure_pytorch":
            d_inner = d_model * expand
            self.in_proj = nn.Linear(d_model, d_inner * 2)
            self.ssm = CausalSelectiveSSM(d_inner, d_state=d_state)
            self.act = nn.SiLU()
            self.out_proj = nn.Linear(d_inner, d_model)
        elif backend == "mamba_ssm":
            if not torch.cuda.is_available():
                raise RuntimeError(
                    "CausalMambaBlock(backend='mamba_ssm') requires a CUDA GPU; none visible. "
                    "Use backend='pure_pytorch' for CPU/MPS/local runs."
                )
            try:
                from mamba_ssm import Mamba
            except ImportError as e:
                raise ImportError(
                    "backend='mamba_ssm' requires the `mamba-ssm` package (pip install mamba-ssm, "
                    "needs CUDA toolchain to build) -- see slurm/train_rtx6000.sbatch for the "
                    "install step used on the cluster."
                ) from e
            self.mamba = Mamba(d_model=d_model, d_state=d_state, expand=expand)
        else:
            raise ValueError(f"unknown backend {backend!r}, expected 'pure_pytorch' or 'mamba_ssm'")

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        residual = x
        x = self.norm(x)
        if self.backend == "mamba_ssm":
            x = self.mamba(x)
        else:
            x, gate = self.in_proj(x).chunk(2, dim=-1)
            x = self.ssm(self.act(x))
            x = x * self.act(gate)
            x = self.out_proj(x)
        return residual + x


class SincNetMambaEncoder(nn.Module):
    """Full brain encoder: per-channel SincNet -> channel-mix -> 20ms
    tokenizer -> stack of causal Mamba blocks.

    Shapes
    ------
    Input  cortical_eeg: (B, C_cortical, T_raw) -- output of
        SpatialWhiteningFrontEnd, at `sample_rate_hz`.
    Output tokens: (B, n_tokens, d_model) -- n_tokens ~= T_raw / samples_per_20ms,
        one latent vector per 20ms chunk, per spec Part 3.
    """

    def __init__(
        self,
        n_cortical_channels: int,
        sample_rate_hz: float,
        n_sinc_filters: int = 32,
        sinc_kernel_size: int = 101,
        d_model: int = 256,
        n_layers: int = 6,
        d_state: int = 16,
        mamba_backend: str = "pure_pytorch",
    ) -> None:
        super().__init__()
        self.sample_rate_hz = sample_rate_hz

        self.sinc = SincConv1d(
            n_filters=n_sinc_filters, kernel_size=sinc_kernel_size, sample_rate_hz=sample_rate_hz
        )
        # Applies the same learned filterbank to every cortical channel
        # independently, then mixes channels x filters -> d_model with a
        # pointwise conv (this is the "temporal layer is SincNet" requirement
        # combined with the reality that EEG has many spatial channels: the
        # sinc filters are shared across channels, spatial mixing is a
        # separate learned step).
        self.channel_mix = nn.Conv1d(n_sinc_filters * n_cortical_channels, d_model, kernel_size=1)
        self.n_cortical_channels = n_cortical_channels
        self.n_sinc_filters = n_sinc_filters

        self.tokenizer = TemporalTokenizer(d_model, d_model, sample_rate_hz, token_ms=20.0)
        self.blocks = nn.ModuleList(
            [CausalMambaBlock(d_model, d_state=d_state, backend=mamba_backend) for _ in range(n_layers)]
        )
        self.final_norm = nn.LayerNorm(d_model)

    def forward(self, cortical_eeg: torch.Tensor) -> torch.Tensor:
        """cortical_eeg: (B, C_cortical, T_raw) -> (B, n_tokens, d_model)"""
        B, C, T = cortical_eeg.shape
        assert C == self.n_cortical_channels, (
            f"expected {self.n_cortical_channels} cortical channels, got {C}"
        )

        x = cortical_eeg.reshape(B * C, 1, T)  # (B*C, 1, T)
        x = self.sinc(x)  # (B*C, n_sinc_filters, T)
        x = x.reshape(B, C * self.n_sinc_filters, -1)  # (B, C*n_filters, T)
        x = self.channel_mix(x)  # (B, d_model, T)

        tokens = self.tokenizer(x)  # (B, n_tokens, d_model)
        for block in self.blocks:
            tokens = block(tokens)
        return self.final_norm(tokens)
