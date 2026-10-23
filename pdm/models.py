"""Baseline (gradient boosting on window features) and deep models (1D-CNN, LSTM)."""
from __future__ import annotations

import time
from dataclasses import dataclass

import numpy as np


# ------------------------------------------------------------------- baseline
def make_baseline(seed: int = 0):
    from sklearn.ensemble import HistGradientBoostingRegressor

    return HistGradientBoostingRegressor(
        max_iter=400, learning_rate=0.05, max_leaf_nodes=31, min_samples_leaf=40,
        l2_regularization=1.0, early_stopping=True, validation_fraction=0.15,
        n_iter_no_change=30, random_state=seed,
    )


def make_linear():
    from sklearn.linear_model import Ridge
    from sklearn.pipeline import make_pipeline
    from sklearn.preprocessing import StandardScaler

    return make_pipeline(StandardScaler(), Ridge(alpha=1.0))


# ------------------------------------------------------------------- deep models
def _torch():
    import torch  # imported lazily so the baseline works without torch

    return torch


def build_net(kind: str, n_features: int, window: int):
    torch = _torch()
    nn = torch.nn

    class CNN1D(nn.Module):
        """1D-CNN in the spirit of Li et al. (2018): stacked temporal convolutions."""

        def __init__(self):
            super().__init__()
            self.body = nn.Sequential(
                nn.Conv1d(n_features, 32, 9, padding=4), nn.ReLU(),
                nn.Conv1d(32, 32, 9, padding=4), nn.ReLU(),
                nn.Conv1d(32, 32, 9, padding=4), nn.ReLU(),
                nn.Conv1d(32, 8, 3, padding=1), nn.ReLU(),
            )
            self.head = nn.Sequential(nn.Flatten(), nn.Dropout(0.3),
                                      nn.Linear(8 * window, 64), nn.ReLU(), nn.Linear(64, 1))

        def forward(self, x):  # x: (B, W, F)
            return self.head(self.body(x.transpose(1, 2))).squeeze(-1)

    class LSTMNet(nn.Module):
        def __init__(self):
            super().__init__()
            self.lstm = nn.LSTM(n_features, 64, num_layers=2, batch_first=True, dropout=0.2)
            self.head = nn.Sequential(nn.Linear(64, 32), nn.ReLU(), nn.Linear(32, 1))

        def forward(self, x):
            out, _ = self.lstm(x)
            return self.head(out[:, -1]).squeeze(-1)

    if kind == "cnn":
        return CNN1D()
    if kind == "lstm":
        return LSTMNet()
    raise ValueError(kind)


def get_device(prefer: str = "auto"):
    torch = _torch()
    if prefer == "cpu":
        return torch.device("cpu")
    return torch.device("cuda" if torch.cuda.is_available() else "cpu")


@dataclass
class TrainResult:
    model: object
    history: list
    best_val_rmse: float
    seconds: float
    device: str


def train_net(kind, X_tr, y_tr, X_val=None, y_val=None, epochs=80, batch_size=512, lr=1e-3,
              patience=15, seed=0, device="auto", verbose=True) -> TrainResult:
    """Train with Adam + MSE, early-stopping on validation RMSE (restores best weights)."""
    torch = _torch()
    torch.manual_seed(seed)
    np.random.seed(seed)
    dev = get_device(device)
    net = build_net(kind, X_tr.shape[2], X_tr.shape[1]).to(dev)
    opt = torch.optim.Adam(net.parameters(), lr=lr)
    sched = torch.optim.lr_scheduler.ReduceLROnPlateau(opt, factor=0.5, patience=5)
    Xt = torch.tensor(X_tr, dtype=torch.float32, device=dev)
    yt = torch.tensor(y_tr, dtype=torch.float32, device=dev)
    has_val = X_val is not None and len(X_val) > 0
    best, best_state, bad, hist = np.inf, None, 0, []
    t0 = time.time()
    g = torch.Generator(device="cpu").manual_seed(seed)
    for ep in range(epochs):
        net.train()
        perm = torch.randperm(len(Xt), generator=g).to(dev)
        tot = 0.0
        for i in range(0, len(Xt), batch_size):
            b = perm[i:i + batch_size]
            opt.zero_grad()
            loss = torch.nn.functional.mse_loss(net(Xt[b]), yt[b])
            loss.backward()
            opt.step()
            tot += loss.item() * len(b)
        tr_rmse = float(np.sqrt(tot / len(Xt)))
        val_rmse = float(np.sqrt(np.mean((predict_net(net, X_val, dev) - y_val) ** 2))) if has_val else tr_rmse
        sched.step(val_rmse)
        hist.append({"epoch": ep + 1, "train_rmse": tr_rmse, "val_rmse": val_rmse})
        if verbose:
            print(f"  [{kind}] epoch {ep + 1:3d}  train RMSE {tr_rmse:6.2f}  val RMSE {val_rmse:6.2f}")
        if val_rmse < best - 1e-3:
            best, bad = val_rmse, 0
            best_state = {k: v.detach().cpu().clone() for k, v in net.state_dict().items()}
        else:
            bad += 1
            if bad >= patience:
                break
    if best_state is not None:
        net.load_state_dict(best_state)
    net = net.cpu().eval()
    return TrainResult(net, hist, float(best), time.time() - t0, str(dev))


def predict_net(net, X, device=None, batch_size=4096) -> np.ndarray:
    torch = _torch()
    if X is None or len(X) == 0:
        return np.array([])
    dev = device or next(net.parameters()).device
    net.eval()
    out = []
    with torch.no_grad():
        for i in range(0, len(X), batch_size):
            xb = torch.tensor(X[i:i + batch_size], dtype=torch.float32, device=dev)
            out.append(net(xb).cpu().numpy())
    return np.concatenate(out)


def make_ensemble(nets):
    """Average the outputs of several independently seeded networks (reduces variance)."""
    torch = _torch()

    class Ensemble(torch.nn.Module):
        def __init__(self):
            super().__init__()
            self.members = torch.nn.ModuleList(nets)

        def forward(self, x):
            return torch.stack([m(x) for m in self.members]).mean(0)

    return Ensemble().eval()


def save_net(net, path, meta: dict):
    """Save a single net or an ensemble (list of member state dicts) + metadata."""
    torch = _torch()
    members = list(net.members) if hasattr(net, "members") else [net]
    torch.save({"state_dicts": [m.state_dict() for m in members], "meta": meta}, path)


def load_net(path):
    torch = _torch()
    ck = torch.load(path, map_location="cpu", weights_only=True)
    m = ck["meta"]
    nets = []
    for sd in ck["state_dicts"]:
        n = build_net(m["kind"], m["n_features"], m["window"])
        n.load_state_dict(sd)
        nets.append(n.eval())
    return (nets[0] if len(nets) == 1 else make_ensemble(nets)), m
