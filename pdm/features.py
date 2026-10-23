"""Preprocessing: operating-condition normalisation, sensor selection, windowing."""
from __future__ import annotations

import json
from dataclasses import asdict, dataclass, field

import numpy as np
import pandas as pd
from numpy.lib.stride_tricks import sliding_window_view

from .data import SENSORS, SETTINGS

DEFAULT_WINDOW = 30


@dataclass
class Preprocessor:
    """Fit on training data only; apply the same transform to validation/test.

    * FD002/FD004 run at six operating conditions, which shift every sensor. We cluster
      the three operating settings into conditions and z-score each sensor *within* its
      condition, so what remains is degradation rather than flight regime.
    * Sensors that are constant (or flip between at most a few discrete values) inside a
      condition carry no degradation signal and are dropped.
    """

    n_conditions: int = 1
    centers: list = field(default_factory=list)   # condition centroids in settings space
    sensors: list = field(default_factory=list)   # kept sensors
    mean: dict = field(default_factory=dict)      # {cond: [mean per sensor]}
    std: dict = field(default_factory=dict)

    # ---------------------------------------------------------------- fitting
    def fit(self, df: pd.DataFrame, max_levels: int = 3) -> "Preprocessor":
        multi = df["setting1"].std() > 1.0
        if multi:
            from sklearn.cluster import KMeans

            km = KMeans(n_clusters=6, n_init=10, random_state=0).fit(df[SETTINGS].to_numpy())
            self.n_conditions = 6
            self.centers = km.cluster_centers_.tolist()
        else:
            self.n_conditions = 1
            self.centers = [df[SETTINGS].mean().tolist()]
        cond = self.conditions(df)
        levels = df[SENSORS].groupby(cond).nunique().max()
        self.sensors = [s for s in SENSORS if levels[s] > max_levels]
        g = df[self.sensors].groupby(cond)
        self.mean = {int(k): v.tolist() for k, v in g.mean().iterrows()}
        self.std = {int(k): v.replace(0, 1.0).tolist() for k, v in g.std().iterrows()}
        return self

    def conditions(self, df: pd.DataFrame) -> np.ndarray:
        if self.n_conditions == 1:
            return np.zeros(len(df), dtype=int)
        c = np.asarray(self.centers)
        x = df[SETTINGS].to_numpy()
        # scale settings so all three matter roughly equally in the distance
        scale = np.abs(c).max(axis=0) + 1e-9
        d = (((x[:, None, :] - c[None]) / scale) ** 2).sum(-1)
        return d.argmin(1)

    def transform(self, df: pd.DataFrame) -> pd.DataFrame:
        """Return a copy with kept sensors z-scored per operating condition."""
        out = df[["unit", "cycle"] + (["rul"] if "rul" in df else [])].copy()
        cond = self.conditions(df)
        x = df[self.sensors].to_numpy(dtype=float)
        mu = np.array([self.mean[int(k)] for k in cond])
        sd = np.array([self.std[int(k)] for k in cond])
        z = (x - mu) / sd
        for i, s in enumerate(self.sensors):
            out[s] = z[:, i]
        return out

    # ---------------------------------------------------------------- persistence
    def to_json(self) -> str:
        return json.dumps(asdict(self))

    @classmethod
    def from_json(cls, s: str) -> "Preprocessor":
        d = json.loads(s)
        d["mean"] = {int(k): v for k, v in d["mean"].items()}
        d["std"] = {int(k): v for k, v in d["std"].items()}
        return cls(**d)


# -------------------------------------------------------------------- windowing
def _pad_front(arr: np.ndarray, window: int) -> np.ndarray:
    """Repeat the first row so even very short trajectories yield a full window."""
    if len(arr) >= window:
        return arr
    return np.vstack([np.repeat(arr[:1], window - len(arr), axis=0), arr])


def unit_windows(arr: np.ndarray, window: int) -> np.ndarray:
    """All windows ending at every cycle of one engine: shape (n_cycles, window, F).

    Windows that would start before cycle 1 are front-padded with the first reading.
    """
    n = len(arr)
    padded = np.vstack([np.repeat(arr[:1], window - 1, axis=0), arr])
    w = sliding_window_view(padded, window, axis=0)  # (n, F, window)
    assert w.shape[0] == n
    return np.ascontiguousarray(w.transpose(0, 2, 1))


def make_sequences(df: pd.DataFrame, feature_cols: list[str], window: int = DEFAULT_WINDOW,
                   last_only: bool = False):
    """Sliding windows for sequence models.

    Returns ``X`` (N, window, F), ``y`` (N,) or None, and ``index`` (DataFrame of unit, cycle)
    so predictions can be mapped back to engines.
    """
    Xs, ys, idx = [], [], []
    for unit, g in df.groupby("unit", sort=True):
        g = g.sort_values("cycle")
        arr = g[feature_cols].to_numpy(dtype=np.float32)
        w = unit_windows(arr, window)
        if last_only:
            w = w[-1:]
            g = g.iloc[-1:]
        Xs.append(w)
        idx.append(g[["unit", "cycle"]])
        if "rul" in g:
            ys.append(g["rul"].to_numpy(dtype=np.float32))
    X = np.concatenate(Xs).astype(np.float32)
    y = np.concatenate(ys) if ys else None
    return X, y, pd.concat(idx, ignore_index=True)


def window_features(df: pd.DataFrame, feature_cols: list[str], window: int = DEFAULT_WINDOW,
                    last_only: bool = False):
    """Hand-crafted rolling-window features for tree / linear models.

    For each sensor over the trailing ``window`` cycles: last value, mean, std, min, max and
    least-squares slope (degradation rate). Plus the current cycle (engine age).
    """
    X, y, idx = make_sequences(df, feature_cols, window, last_only)
    t = np.arange(window, dtype=np.float32)
    tc = t - t.mean()
    slope = np.einsum("nwf,w->nf", X - X.mean(1, keepdims=True), tc) / (tc ** 2).sum()
    parts = {
        "last": X[:, -1, :], "mean": X.mean(1), "std": X.std(1),
        "min": X.min(1), "max": X.max(1), "slope": slope,
    }
    names, cols = [], []
    for k, v in parts.items():
        cols.append(v)
        names += [f"{c}_{k}" for c in feature_cols]
    feats = pd.DataFrame(np.hstack(cols), columns=names)
    feats["cycle"] = idx["cycle"].to_numpy(dtype=float)
    return feats, y, idx
