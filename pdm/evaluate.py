"""Metrics used in the C-MAPSS / PHM08 literature."""
from __future__ import annotations

import numpy as np

# Values reported by the original authors on the official C-MAPSS test sets
# (RMSE, PHM score). Protocols differ slightly between papers (window length, RUL cap,
# number of runs averaged), so treat these as ballpark references, not exact targets.
PUBLISHED = {
    "FD001": {
        "CNN, Babu et al. 2016": (18.45, 1287),
        "LSTM, Zheng et al. 2017": (16.14, 338),
        "DCNN, Li et al. 2018": (12.61, 274),
    },
    "FD002": {
        "CNN, Babu et al. 2016": (30.29, 13570),
        "LSTM, Zheng et al. 2017": (24.49, 4450),
        "DCNN, Li et al. 2018": (22.36, 10412),
    },
    "FD003": {
        "CNN, Babu et al. 2016": (19.82, 1596),
        "LSTM, Zheng et al. 2017": (16.18, 852),
        "DCNN, Li et al. 2018": (12.64, 284),
    },
    "FD004": {
        "CNN, Babu et al. 2016": (29.16, 7886),
        "LSTM, Zheng et al. 2017": (28.17, 5550),
        "DCNN, Li et al. 2018": (23.31, 12466),
    },
}


def rmse(y_true, y_pred) -> float:
    y_true, y_pred = np.asarray(y_true, float), np.asarray(y_pred, float)
    return float(np.sqrt(np.mean((y_pred - y_true) ** 2)))


def phm_score(y_true, y_pred) -> float:
    """PHM08 challenge scoring function (Saxena et al. 2008).

    d = predicted - true. Late predictions (d > 0, i.e. we think there's more life left
    than there is) are penalised harder than early ones:
        s = exp(-d/13) - 1  if d < 0
        s = exp( d/10) - 1  if d >= 0
    Lower is better; the score is summed over engines.
    """
    d = np.asarray(y_pred, float) - np.asarray(y_true, float)
    s = np.where(d < 0, np.exp(-d / 13.0) - 1.0, np.exp(d / 10.0) - 1.0)
    return float(s.sum())


def summarize(y_true, y_pred, cap: float | None = None) -> dict:
    """RMSE / score / MAE on the given truth, optionally also on truth clipped at ``cap``."""
    y_true = np.asarray(y_true, float)
    y_pred = np.asarray(y_pred, float)
    out = {
        "rmse": rmse(y_true, y_pred),
        "score": phm_score(y_true, y_pred),
        "mae": float(np.mean(np.abs(y_pred - y_true))),
        "n": int(len(y_true)),
    }
    if cap is not None:
        yc = np.minimum(y_true, cap)
        out["rmse_capped_truth"] = rmse(yc, y_pred)
        out["score_capped_truth"] = phm_score(yc, y_pred)
    return out
