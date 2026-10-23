"""End-to-end training / evaluation / artifact management shared by scripts and the app."""
from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import pandas as pd

from . import evaluate, maintenance, models
from .data import DATA_DIR, DEFAULT_CAP, ROOT, add_test_rul, add_train_rul, load_subset, split_units
from .features import DEFAULT_WINDOW, Preprocessor, make_sequences, window_features

ARTIFACT_DIR = ROOT / "models"
MODEL_LABELS = {"gbm": "Gradient boosting (window features)", "cnn": "1D-CNN (PyTorch)",
                "lstm": "LSTM (PyTorch)"}


@dataclass
class Prepared:
    fd: str
    pre: Preprocessor
    train: pd.DataFrame      # normalised, capped RUL, training engines
    val: pd.DataFrame        # normalised, capped RUL, held-out engines (full lives)
    val_true: pd.Series      # uncapped RUL for val rows
    test: pd.DataFrame       # normalised, uncapped RUL, official test engines
    test_rul: pd.Series      # official truth at last cycle
    raw_test: pd.DataFrame
    raw_train: pd.DataFrame


def prepare(fd="FD001", cap=DEFAULT_CAP, val_frac=0.2, seed=42, data_dir=DATA_DIR,
            units_limit: int | None = None) -> Prepared:
    raw_train, raw_test, test_rul = load_subset(fd, data_dir)
    if units_limit:  # tiny subset for smoke tests
        raw_train = raw_train[raw_train.unit <= units_limit]
        raw_test = raw_test[raw_test.unit <= units_limit]
        test_rul = test_rul[test_rul.index <= units_limit]
    tr_units, val_units = split_units(raw_train["unit"].unique(), val_frac, seed)
    lab = add_train_rul(raw_train, cap)
    true_unc = add_train_rul(raw_train, None)["rul"]
    tr_df, val_df = lab[lab.unit.isin(tr_units)], lab[lab.unit.isin(val_units)]
    pre = Preprocessor().fit(raw_train[raw_train.unit.isin(tr_units)])
    test_lab = add_test_rul(raw_test, test_rul)
    return Prepared(fd, pre, pre.transform(tr_df), pre.transform(val_df),
                    true_unc[val_df.index].reset_index(drop=True), pre.transform(test_lab),
                    test_rul, raw_test, raw_train)


def predict(kind: str, model, df: pd.DataFrame, sensors, window: int, last_only=False, cap=DEFAULT_CAP):
    """Predict RUL for every cycle (or the last cycle) of every engine in ``df``."""
    if kind == "gbm":
        X, _, idx = window_features(df, sensors, window, last_only)
        p = model.predict(X)
    else:
        X, _, idx = make_sequences(df, sensors, window, last_only)
        p = models.predict_net(model, X)
    idx = idx.copy()
    idx["pred_rul"] = np.clip(p, 0, cap)
    return idx


def train_all(fd="FD001", kinds=("gbm", "cnn", "lstm"), window=DEFAULT_WINDOW, cap=DEFAULT_CAP,
              epochs=60, out_dir: Path | None = ARTIFACT_DIR, units_limit=None, device="auto",
              verbose=True, cost=maintenance.CostModel(), n_seeds=3) -> dict:
    prep = prepare(fd, cap, units_limit=units_limit)
    sensors = prep.pre.sensors
    out = Path(out_dir) / fd if out_dir else None
    if out:
        out.mkdir(parents=True, exist_ok=True)
        (out / "preprocessor.json").write_text(prep.pre.to_json())

    results, val_preds, trained = {}, {}, {}
    for kind in kinds:
        if verbose:
            print(f"== {fd} / {kind} ==")
        if kind == "gbm":
            import time

            import joblib

            t0 = time.time()
            Xtr, ytr, _ = window_features(prep.train, sensors, window)
            m = models.make_baseline().fit(Xtr, ytr)
            secs, dev, hist = time.time() - t0, "cpu", []
            if out:
                joblib.dump(m, out / "gbm.joblib", compress=3)
        else:
            Xtr, ytr, _ = make_sequences(prep.train, sensors, window)
            Xv, yv, _ = make_sequences(prep.val, sensors, window)
            runs = [models.train_net(kind, Xtr, ytr, Xv, yv, epochs=epochs, device=device,
                                     seed=seed, verbose=verbose) for seed in range(n_seeds)]
            m = runs[0].model if n_seeds == 1 else models.make_ensemble([r.model for r in runs])
            secs, dev, hist = sum(r.seconds for r in runs), runs[0].device, runs[0].history
            if out:
                models.save_net(m, out / f"{kind}.pt", {"kind": kind, "n_features": len(sensors),
                                                        "window": window, "cap": cap})
        trained[kind] = m
        test_last = predict(kind, m, prep.test, sensors, window, last_only=True, cap=cap)
        y_true = prep.test_rul.loc[test_last["unit"]].to_numpy()
        vp = predict(kind, m, prep.val, sensors, window, cap=cap)
        val_preds[kind] = vp["pred_rul"].to_numpy()
        val_capped = prep.val.sort_values(["unit", "cycle"])["rul"].to_numpy()
        res = evaluate.summarize(y_true, test_last["pred_rul"].to_numpy(), cap=cap)
        res.update({"val_rmse_capped": evaluate.rmse(val_capped, val_preds[kind]),
                    "train_seconds": round(secs, 1), "device": dev,
                    "epochs_run": len(hist) if hist else None})
        results[kind] = res
        if verbose:
            print(f"   test RMSE {res['rmse']:.2f} | PHM score {res['score']:.0f} | {secs:.1f}s on {dev}")
        if out and hist:
            pd.DataFrame(hist).to_csv(out / f"{kind}_history.csv", index=False)

    # validation trajectories (full lives) for the maintenance decision layer
    vidx = prep.val.sort_values(["unit", "cycle"])[["unit", "cycle"]].reset_index(drop=True)
    vt = vidx.copy()
    vt["true_rul"] = add_train_rul(prep.raw_train, None).set_index(["unit", "cycle"]).loc[
        list(zip(vidx.unit, vidx.cycle)), "rul"].to_numpy()
    for kind, p in val_preds.items():
        vt[f"pred_{kind}"] = np.round(p, 2)
    best_kind = min(results, key=lambda k: results[k]["val_rmse_capped"])
    traj = vt.rename(columns={f"pred_{best_kind}": "pred_rul"})
    thresholds = np.arange(0, 101, 2)
    table = maintenance.policy_table(traj, thresholds, cost)
    t_best = maintenance.best_threshold(table)
    lives = vt.groupby("unit")["cycle"].max().to_numpy()
    ages = np.unique(np.linspace(0.4 * lives.min(), lives.max(), 80).astype(int))
    age_tab = pd.DataFrame([maintenance.simulate_age_policy(lives, a, cost) for a in ages])
    best_age = age_tab.loc[age_tab["cost_per_cycle"].idxmin()]
    row = table.set_index("threshold").loc[t_best]
    decision = {
        "model": best_kind, "cost_model": cost.__dict__, "best_threshold": t_best,
        "cbm_cost_per_cycle": float(row["cost_per_cycle"]), "cbm_fail_rate": float(row["fail_rate"]),
        "cbm_mean_rul_wasted": float(row["mean_rul_wasted"]),
        "tbm_best_age": float(best_age["age"]), "tbm_cost_per_cycle": float(best_age["cost_per_cycle"]),
        "tbm_fail_rate": float(best_age["fail_rate"]),
        "run_to_failure_cost_per_cycle": float(cost.c_fail * len(lives) / lives.sum()),
        "n_val_engines": int(len(lives)),
    }
    summary = {"fd": fd, "window": window, "cap": cap, "sensors": sensors,
               "n_conditions": prep.pre.n_conditions, "results": results, "decision": decision,
               "published": evaluate.PUBLISHED.get(fd, {})}
    if out:
        vt.to_csv(out / "val_trajectories.csv", index=False)
        table.to_csv(out / "policy_table.csv", index=False)
        age_tab.to_csv(out / "age_policy_table.csv", index=False)
        (out / "metrics.json").write_text(json.dumps(summary, indent=2))
    summary["_models"] = trained
    summary["_prep"] = prep
    return summary


# ------------------------------------------------------------------- loading for the app
def available_subsets(art_dir=ARTIFACT_DIR):
    return sorted(p.name for p in Path(art_dir).glob("FD00*") if (p / "metrics.json").exists())


def load_artifacts(fd, art_dir=ARTIFACT_DIR):
    d = Path(art_dir) / fd
    metrics = json.loads((d / "metrics.json").read_text())
    pre = Preprocessor.from_json((d / "preprocessor.json").read_text())
    loaded = {}
    if (d / "gbm.joblib").exists():
        import joblib

        loaded["gbm"] = joblib.load(d / "gbm.joblib")
    for k in ("cnn", "lstm"):
        if (d / f"{k}.pt").exists():
            try:
                loaded[k] = models.load_net(d / f"{k}.pt")[0]
            except ImportError:  # torch not installed
                pass
    return metrics, pre, loaded
