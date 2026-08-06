"""
Metrics required by BENCHMARK_SPEC.md's evaluation protocol:
  point forecasts: MSE, MAE, MASE
  probabilistic: CRPS, WQL, empirical coverage
  classification (PTB-XL): macro-F1, AUROC, ECE

All functions take plain numpy arrays and return plain floats, so they don't
care which model produced the forecast. Kept dependency-light (numpy +
sklearn, both already in the project's venv) on purpose.
"""
from __future__ import annotations

import numpy as np

QUANTILE_LEVELS = [0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9]


# ---------------------------------------------------------------------------
# Point forecast metrics
# ---------------------------------------------------------------------------

def mse(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    return float(np.mean((y_true - y_pred) ** 2))


def mae(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    return float(np.mean(np.abs(y_true - y_pred)))


def mase(y_true: np.ndarray, y_pred: np.ndarray, train_series: np.ndarray, season_length: int = 1) -> float:
    """
    Mean Absolute Scaled Error (Hyndman & Koehler 2006). Scale is the MAE of
    a naive seasonal forecast on the TRAINING series only, never on the test
    window, so this never leaks test information into the scale.
    """
    train_series = np.asarray(train_series, dtype=np.float64)
    if len(train_series) <= season_length:
        raise ValueError("train_series too short for the given season_length")
    naive_errors = np.abs(train_series[season_length:] - train_series[:-season_length])
    scale = np.mean(naive_errors)
    if scale == 0:
        return float("nan")  # degenerate series (e.g. all-zero client), flag rather than divide by zero
    return float(np.mean(np.abs(y_true - y_pred)) / scale)


# ---------------------------------------------------------------------------
# Probabilistic forecast metrics
# ---------------------------------------------------------------------------

def quantile_loss(y_true: np.ndarray, y_pred_q: np.ndarray, q: float) -> float:
    """Pinball loss at a single quantile level q."""
    diff = y_true - y_pred_q
    return float(np.mean(np.maximum(q * diff, (q - 1) * diff)))


def weighted_quantile_loss(y_true: np.ndarray, quantile_preds: dict[float, np.ndarray],
                            quantile_levels: list[float] = QUANTILE_LEVELS) -> float:
    """
    WQL as used by GluonTS-style TSFM evals: sum of pinball losses across
    quantile levels, normalized by the sum of |y_true|.
    """
    y_true = np.asarray(y_true, dtype=np.float64)
    total_ql = 0.0
    for q in quantile_levels:
        if q not in quantile_preds:
            raise ValueError(f"missing quantile prediction for level {q}")
        total_ql += quantile_loss(y_true, np.asarray(quantile_preds[q]), q) * len(y_true)
    denom = np.sum(np.abs(y_true))
    if denom == 0:
        return float("nan")
    return float(2 * total_ql / denom)


def crps_from_quantiles(y_true: np.ndarray, quantile_preds: dict[float, np.ndarray],
                         quantile_levels: list[float] = QUANTILE_LEVELS) -> float:
    """
    CRPS approximated as the quantile-averaged pinball loss (a standard
    discretization when only a finite quantile set is available, matches
    what Chronos/TimesFM-style papers report as "CRPS" from quantile heads).
    """
    y_true = np.asarray(y_true, dtype=np.float64)
    losses = [quantile_loss(y_true, np.asarray(quantile_preds[q]), q) for q in quantile_levels]
    return float(2 * np.mean(losses))


def empirical_coverage(y_true: np.ndarray, lower: np.ndarray, upper: np.ndarray) -> float:
    """Fraction of true values falling within [lower, upper]. Compare against
    the nominal interval level to check calibration, e.g. an 80% interval
    (q=0.1 to q=0.9) should cover close to 80% of true values."""
    y_true = np.asarray(y_true)
    inside = (y_true >= lower) & (y_true <= upper)
    return float(np.mean(inside))


# ---------------------------------------------------------------------------
# Classification metrics (PTB-XL)
# ---------------------------------------------------------------------------

def macro_f1(y_true: np.ndarray, y_pred_binary: np.ndarray) -> float:
    """y_true, y_pred_binary: shape (n_samples, n_classes), multi-hot / multi-label."""
    from sklearn.metrics import f1_score
    return float(f1_score(y_true, y_pred_binary, average="macro", zero_division=0))


def macro_auroc(y_true: np.ndarray, y_pred_probs: np.ndarray) -> float:
    """y_true: multi-hot (n_samples, n_classes). y_pred_probs: same shape, predicted probabilities.
    Classes with only one label value present in y_true are skipped (AUROC undefined for them)
    and the skip is not silent, callers should log which classes were dropped."""
    from sklearn.metrics import roc_auc_score
    valid_cols = [c for c in range(y_true.shape[1]) if len(np.unique(y_true[:, c])) > 1]
    if not valid_cols:
        return float("nan")
    return float(roc_auc_score(y_true[:, valid_cols], y_pred_probs[:, valid_cols], average="macro"))


def expected_calibration_error(y_true_binary: np.ndarray, y_pred_prob: np.ndarray, n_bins: int = 10) -> float:
    """ECE for a single binary class (call once per class, then average if a
    macro-ECE across classes is wanted, kept explicit rather than silently
    averaged here since class-level calibration is usually what's actually needed)."""
    y_true_binary = np.asarray(y_true_binary, dtype=np.float64)
    y_pred_prob = np.asarray(y_pred_prob, dtype=np.float64)
    bin_edges = np.linspace(0.0, 1.0, n_bins + 1)
    ece = 0.0
    n = len(y_true_binary)
    for i in range(n_bins):
        lo, hi = bin_edges[i], bin_edges[i + 1]
        mask = (y_pred_prob >= lo) & (y_pred_prob < hi if i < n_bins - 1 else y_pred_prob <= hi)
        if not np.any(mask):
            continue
        bin_conf = np.mean(y_pred_prob[mask])
        bin_acc = np.mean(y_true_binary[mask])
        ece += (np.sum(mask) / n) * abs(bin_conf - bin_acc)
    return float(ece)
