"""Momentum-teacher self-supervision with a heteroscedastic-NLL objective.

HONEST FRAMING (see plan doc, correction #2): this is DINO-*style*
centering/sharpening grafted onto a continuous heteroscedastic-regression
loss, not vanilla DINO. Vanilla DINO computes cross-entropy between two
softmax distributions over a fixed set of prototype logits. Here the
teacher's centered, temperature-sharpened output is instead treated as a
continuous regression *target vector*, and the student predicts a
(mean, precision) pair for it under a diagonal Gaussian negative
log-likelihood. The DINO machinery (EMA teacher, centering, sharpening) is
real and does the same job (prevent representation collapse without negative
pairs / a codebook); the loss head on top of it is not the DINO loss.

SCALE NOTE: run this on a small synthetic or small-subject slice first (see
encoder.py). The collapse-avoidance mechanism below (correction #3: the
Frobenius penalty on precision must stay active even while beta(t)=0) is
exactly the kind of failure mode that's cheap to catch on a toy run and
expensive to debug on a multi-GPU job — confirm the loss doesn't NaN or drive
precision to the floor on a handful of batches locally before scaling to
Modal / the RTX 6000.
"""

from __future__ import annotations

import copy

import torch
from torch import nn


class EMATeacher(nn.Module):
    """Wraps a student encoder with a momentum-averaged, gradient-free copy.

    teacher_params <- m * teacher_params + (1-m) * student_params, applied
    after every student optimizer step via `update()`. The teacher is never
    trained directly (no backward pass ever touches it), which is what makes
    its output usable as a stable regression target rather than something the
    student could trivially collapse onto.
    """

    def __init__(self, student: nn.Module, momentum: float = 0.996):
        super().__init__()
        self.momentum = momentum
        self.teacher = copy.deepcopy(student)
        for p in self.teacher.parameters():
            p.requires_grad_(False)

    @torch.no_grad()
    def update(self, student: nn.Module) -> None:
        for t_p, s_p in zip(self.teacher.parameters(), student.parameters()):
            t_p.mul_(self.momentum).add_(s_p.detach(), alpha=1 - self.momentum)

    @torch.no_grad()
    def forward(self, *args, **kwargs) -> torch.Tensor:
        return self.teacher(*args, **kwargs)


class HeteroscedasticPredictionHead(nn.Module):
    """Student head producing, per latent dimension, a predicted mean Z_hat
    and a predicted precision Pi = softplus(pi_raw) + eps.

    The epsilon floor keeps Pi > 0 strictly (so log(Pi) in the NLL never
    hits -inf), but note (plan correction #3): the floor alone does not stop
    Pi from being driven arbitrarily close to it when the -beta*log(Pi) term
    is switched off early in training. The Frobenius penalty in
    `TeacherStudentDistillationLoss` is what actually prevents that collapse;
    this head only guarantees Pi stays in (eps, inf).

    Shapes: x: (B, T, d_model) -> mean: (B, T, d_model), precision: (B, T, d_model)
    """

    def __init__(self, d_model: int, eps: float = 1e-3):
        super().__init__()
        self.eps = eps
        self.mean_head = nn.Linear(d_model, d_model)
        self.precision_head = nn.Linear(d_model, d_model)

    def forward(self, x: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        mean = self.mean_head(x)
        precision = torch.nn.functional.softplus(self.precision_head(x)) + self.eps
        return mean, precision


class TeacherStudentDistillationLoss(nn.Module):
    r"""Precision-weighted heteroscedastic NLL between a student's prediction
    and a centered, sharpened momentum-teacher target.

    Teacher target construction (per spec):
        Z_teacher_centered = teacher_out - running_center          (subtract batch running mean)
        Z_target = softmax(Z_teacher_centered / tau_teacher, dim=-1)  (sharpen)
    `running_center` is an EMA of the batch mean of raw teacher outputs,
    updated every forward call — this is what isolates slow, non-cognitive
    drift (which the center absorbs) from the faster structure the student is
    asked to match (which survives centering).

    Loss (spec equation, applied elementwise over the d_model axis and
    averaged over batch/tokens):
        L = sum_i [ Pi_i * (Z_hat_i - Z_target_i)^2 - beta(t) * log(Pi_i) ]
            + lambda * ||Pi - I||_F^2

    Shape note (plan correction #4): Pi is produced elementwise
    (softplus per latent dimension), i.e. it is a DIAGONAL precision — a
    (B, T, d_model) tensor, not a full (d_model, d_model) matrix. The
    Frobenius penalty against the identity matrix therefore reduces to
    sum((Pi_i - 1)^2), which is what's implemented below; a full covariance
    would need O(d_model^2) parameters per token and isn't what the
    elementwise Softplus head can produce.

    Annealing beta(t) (spec: starts at 0, decays to 1 -- re-read as *rises*
    from 0 to 1, "forcing standard MSE early in training" only in the sense
    that the log(Pi) regularizer is off early, not that the loss is literally
    MSE, since Pi still scales the squared error; see plan correction #3 for
    why the Frobenius term must NOT be annealed alongside beta).
    """

    def __init__(
        self,
        center_momentum: float = 0.9,
        tau_teacher: float = 0.04,
        frobenius_lambda: float = 0.01,
        anneal_steps: int = 10_000,
    ):
        super().__init__()
        self.center_momentum = center_momentum
        self.tau_teacher = tau_teacher
        self.frobenius_lambda = frobenius_lambda
        self.anneal_steps = anneal_steps
        self.register_buffer("center", None, persistent=False)

    def beta(self, step: int) -> float:
        """Linear anneal 0 -> 1 over `anneal_steps`, then held at 1."""
        return min(1.0, step / max(1, self.anneal_steps))

    @torch.no_grad()
    def _update_center(self, teacher_out: torch.Tensor) -> torch.Tensor:
        batch_mean = teacher_out.mean(dim=(0, 1))  # (d_model,)
        if self.center is None:
            self.center = batch_mean.clone()
        else:
            self.center.mul_(self.center_momentum).add_(batch_mean, alpha=1 - self.center_momentum)
        return self.center

    def forward(
        self,
        student_mean: torch.Tensor,
        student_precision: torch.Tensor,
        teacher_out: torch.Tensor,
        step: int,
    ) -> dict:
        """
        student_mean, student_precision, teacher_out: (B, T, d_model)
        teacher_out is assumed already detached (produced under
        `EMATeacher.forward`'s `torch.no_grad()`).

        Returns dict with 'loss' (scalar) plus diagnostic scalars.
        """
        center = self._update_center(teacher_out)
        target = torch.softmax((teacher_out - center) / self.tau_teacher, dim=-1)

        beta_t = self.beta(step)
        sq_err = (student_mean - target) ** 2
        nll = student_precision * sq_err - beta_t * torch.log(student_precision)

        # Active at every step regardless of beta_t -- this is the mechanism
        # that keeps precision from collapsing toward the epsilon floor while
        # the -beta*log(Pi) term is still weak/off (plan correction #3).
        frobenius = (student_precision - 1.0) ** 2

        loss = nll.mean() + self.frobenius_lambda * frobenius.mean()

        return {
            "loss": loss,
            "nll_term": nll.mean().detach(),
            "frobenius_term": frobenius.mean().detach(),
            "beta": beta_t,
            "mean_precision": student_precision.mean().detach(),
        }
