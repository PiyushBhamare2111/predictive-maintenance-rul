"""Generate README figures (docs/*.png) and showcase images (docs/media/*.png, 1920x1080).

Needs the downloaded data and trained artifacts:
    python scripts/download_data.py && python scripts/train.py --fd all
    python scripts/make_figures.py
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

import matplotlib  # noqa: E402

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402

from pdm import theme as T  # noqa: E402
from pdm.data import SENSOR_NAMES, add_train_rul  # noqa: E402
from pdm.pipeline import ARTIFACT_DIR, MODEL_LABELS, available_subsets, load_artifacts, predict, prepare  # noqa: E402

DOCS = ROOT / "docs"
MEDIA = DOCS / "media"
DOCS.mkdir(exist_ok=True)
MEDIA.mkdir(exist_ok=True)
T.use_matplotlib()
SHORT = {"gbm": "Gradient boosting", "cnn": "1D-CNN", "lstm": "LSTM"}


def save(fig, name, media=False):
    path = (MEDIA if media else DOCS) / name
    fig.savefig(path, dpi=100 if media else 150, bbox_inches=None if media else "tight")
    plt.close(fig)
    print("wrote", path.relative_to(ROOT))


def sensor_trends(prep, ax_grid=None, sensors=("s2", "s3", "s4", "s7", "s11", "s12")):
    raw = add_train_rul(prep.raw_train, None)
    fig, axes = plt.subplots(2, 3, figsize=(14, 7), sharex=True)
    for ax, s in zip(axes.ravel(), sensors):
        for _, g in raw.groupby("unit"):
            ax.plot(-g["rul"], g[s], color=T.BLUE, alpha=0.10, lw=0.8)
        med = raw.assign(b=(raw["rul"] // 5) * 5).groupby("b")[s].median()
        med = med[med.index <= 300]
        ax.plot(-med.index, med.values, color=T.ACCENT, lw=2.5)
        ax.set_title(SENSOR_NAMES[s], fontsize=11)
    for ax in axes[1]:
        ax.set_xlabel("cycles before failure")
    fig.suptitle("FD001: sensors drift as the engine approaches failure  (100 engines, median in orange)",
                 x=0.01, ha="left", fontsize=14, fontweight="bold", color=T.TEXT)
    fig.tight_layout()
    return fig


def dropped_sensors(prep):
    raw = prep.raw_train
    stds = (raw[[f"s{i}" for i in range(1, 22)]].std() / raw[[f"s{i}" for i in range(1, 22)]].mean().abs())
    fig, ax = plt.subplots(figsize=(12, 3.6))
    kept = set(prep.pre.sensors)
    cols = [T.ACCENT if s in kept else T.GRID for s in stds.index]
    ax.bar(stds.index, stds.values * 100, color=cols, width=0.7)
    ax.set_ylabel("coefficient of variation (%)")
    ax.set_title(f"FD001: {21 - len(kept)} flat sensors dropped (grey), {len(kept)} kept (orange)")
    return fig


def engine_plot(ax, traj, unit, kinds, legend=True):
    g = traj[traj.unit == unit]
    ax.plot(g.cycle, g.true_rul.clip(upper=125), color=T.MUTED, lw=1.2, ls="--", label="True RUL (capped at 125)")
    ax.plot(g.cycle, g.true_rul, color=T.TRUE, lw=2.4, label="True RUL")
    for k in kinds:
        ax.plot(g.cycle, g[f"pred_{k}"], color=T.MODEL_COLORS[k], lw=1.6, alpha=0.95, label=SHORT[k])
    ax.set_xlabel("cycle")
    ax.set_ylabel("remaining useful life (cycles)")
    ax.set_ylim(0, None)
    if legend:
        ax.legend(loc="upper right")


def main():
    fds = available_subsets()
    if "FD001" not in fds:
        sys.exit("Train FD001 first: python scripts/train.py")
    metrics = {fd: json.loads((ARTIFACT_DIR / fd / "metrics.json").read_text()) for fd in fds}
    m1 = metrics["FD001"]
    kinds = list(m1["results"])
    best = min(kinds, key=lambda k: m1["results"][k]["rmse"])
    prep = prepare("FD001")
    _, pre, loaded = load_artifacts("FD001")
    traj = pd.read_csv(ARTIFACT_DIR / "FD001" / "val_trajectories.csv")
    policy = pd.read_csv(ARTIFACT_DIR / "FD001" / "policy_table.csv")
    age = pd.read_csv(ARTIFACT_DIR / "FD001" / "age_policy_table.csv")
    dec = m1["decision"]

    # 1 sensor trends + dropped sensors
    save(sensor_trends(prep), "sensor_trends.png")
    save(dropped_sensors(prep), "sensor_selection.png")

    # 2 predicted vs true on official test set
    test_preds = {k: predict(k, loaded[k], prep.test, pre.sensors, m1["window"], last_only=True)
                  for k in kinds}
    y = prep.test_rul.loc[test_preds[kinds[0]].unit].to_numpy()
    fig, axes = plt.subplots(1, 2, figsize=(14, 5.5), gridspec_kw={"width_ratios": [1, 1.6]})
    p = test_preds[best].pred_rul.to_numpy()
    ax = axes[0]
    ax.plot([0, 150], [0, 150], color=T.MUTED, lw=1, ls="--")
    ax.scatter(y, p, s=36, color=T.ACCENT, edgecolor=T.BG, linewidth=1)
    ax.set_xlabel("true RUL at last observed cycle")
    ax.set_ylabel("predicted RUL")
    ax.set_title(f"{SHORT[best]} on 100 test engines  (RMSE {m1['results'][best]['rmse']:.1f})")
    ax = axes[1]
    o = np.argsort(y)
    ax.plot(np.arange(len(y)), y[o], color=T.TRUE, lw=2.6, label="True RUL")
    for k in kinds:
        ax.plot(np.arange(len(y)), test_preds[k].pred_rul.to_numpy()[o], color=T.MODEL_COLORS[k],
                lw=1.3, marker="o", ms=3, alpha=0.9, label=SHORT[k])
    ax.set_xlabel("test engine (sorted by true RUL)")
    ax.set_ylabel("RUL (cycles)")
    ax.set_title("Predicted vs true RUL, every model")
    ax.legend(loc="upper left")
    fig.tight_layout()
    save(fig, "pred_vs_true_test.png")

    # 3 error distribution
    fig, ax = plt.subplots(figsize=(10, 4.8))
    bins = np.arange(-60, 61, 5)
    for k in kinds:
        e = test_preds[k].pred_rul.to_numpy() - y
        ax.hist(e, bins=bins, histtype="step", lw=2.2, color=T.MODEL_COLORS[k],
                label=f"{SHORT[k]}  (mean {e.mean():+.1f})")
    ax.axvline(0, color=T.MUTED, lw=1)
    ax.text(-58, ax.get_ylim()[1] * 0.92, "early  (safe, wastes life)", color=T.MUTED)
    ax.text(22, ax.get_ylim()[1] * 0.92, "late  (dangerous)", color=T.RED)
    ax.set_xlabel("prediction error = predicted − true (cycles)")
    ax.set_ylabel("engines")
    ax.set_title("FD001 test-set error distribution")
    ax.legend(loc="upper left", bbox_to_anchor=(0, 0.85))
    save(fig, "error_distribution.png")

    # 4 engine trajectories (validation engines, full life)
    units = traj.groupby("unit").cycle.max().sort_values()
    pick = [units.index[len(units) // 4], units.index[len(units) // 2], units.index[-2]]
    fig, axes = plt.subplots(1, 3, figsize=(16, 4.8), sharey=True)
    for i, (ax, u) in enumerate(zip(axes, pick)):
        engine_plot(ax, traj, u, kinds, legend=(i == 0))
        ax.set_title(f"Held-out engine #{u}  ({units[u]} cycles to failure)")
    fig.tight_layout()
    save(fig, "engine_trajectories.png")

    # 5 model comparison
    fig, ax = plt.subplots(figsize=(12, 5))
    w = 0.2
    x = np.arange(len(fds))
    for i, k in enumerate(kinds):
        vals = [metrics[fd]["results"][k]["rmse"] for fd in fds]
        bars = ax.bar(x + (i - 1) * w, vals, w * 0.92, color=T.MODEL_COLORS[k], label=SHORT[k])
        ax.bar_label(bars, fmt="%.1f", color=T.TEXT, fontsize=9, padding=2)
    li = [metrics[fd]["published"].get("DCNN, Li et al. 2018", (np.nan,))[0] for fd in fds]
    ax.scatter(x + 2 * w, li, marker="D", s=60, color=T.TEXT, zorder=5, label="Published DCNN (Li et al. 2018)")
    ax.set_xticks(x, fds)
    ax.set_ylabel("test RMSE (cycles, lower is better)")
    ax.set_title("Model comparison on the official C-MAPSS test sets")
    ax.legend(loc="upper left", ncol=4)
    ax.grid(axis="x", visible=False)
    ax.set_ylim(0, ax.get_ylim()[1] * 1.22)
    save(fig, "model_comparison.png")

    # 6 maintenance cost curve
    def cost_fig(figsize):
        fig, ax = plt.subplots(figsize=figsize)
        ax.plot(policy.threshold, policy.cost_per_cycle, color=T.ACCENT, lw=2.6,
                label=f"Condition-based: act when predicted RUL ≤ T ({SHORT[dec['model']]})")
        ax.axhline(dec["tbm_cost_per_cycle"], color=T.BLUE, lw=1.6, ls="--",
                   label=f"Best time-based PM (replace at age {dec['tbm_best_age']:.0f})")
        ax.axhline(dec["run_to_failure_cost_per_cycle"], color=T.RED, lw=1.6, ls=":",
                   label="Run to failure")
        ax.scatter([dec["best_threshold"]], [dec["cbm_cost_per_cycle"]], s=110, color=T.ACCENT,
                   edgecolor=T.TEXT, zorder=5)
        ax.annotate(f"optimum T = {dec['best_threshold']:.0f} cycles\n${dec['cbm_cost_per_cycle']:.0f} per cycle",
                    (dec["best_threshold"], dec["cbm_cost_per_cycle"]), xytext=(60, 80),
                    textcoords="offset points", color=T.TEXT,
                    arrowprops=dict(arrowstyle="-", color=T.MUTED))
        ax.set_xlabel("maintenance threshold T (cycles of predicted RUL)")
        ax.set_ylabel("expected cost per operating cycle ($)")
        c = dec["cost_model"]
        ax.set_title(f"Maintenance policy cost  (failure ${c['c_fail']:,.0f}, planned ${c['c_pm']:,.0f}, "
                     f"lead time {c['lead_time']} cycles)")
        ax.set_ylim(0, dec["run_to_failure_cost_per_cycle"] * 1.15)
        ax.legend(loc="upper right")
        return fig, ax

    fig, ax = cost_fig((11, 5))
    save(fig, "cost_curve.png")
    fig, ax = plt.subplots(figsize=(11, 3.6))
    ax.plot(policy.threshold, policy.fail_rate * 100, color=T.RED, lw=2.2, label="unplanned failures (%)")
    ax.plot(policy.threshold, policy.mean_rul_wasted, color=T.ACCENT, lw=2.2, label="useful life thrown away (cycles)")
    ax.set_xlabel("maintenance threshold T")
    ax.set_title("The trade-off: risk of failure vs. wasted life")
    ax.legend()
    save(fig, "tradeoff.png")

    # ---------------------------------------------------------------- 1920x1080 showcase
    def headline(fig, title, sub):
        fig.text(0.04, 0.93, title, fontsize=34, fontweight="bold", color=T.TEXT)
        fig.text(0.04, 0.885, sub, fontsize=17, color=T.MUTED)
        fig.text(0.96, 0.04, "github.com/PiyushBhamare2111/predictive-maintenance-rul",
                 ha="right", fontsize=12, color=T.MUTED)

    # hero
    fig = plt.figure(figsize=(19.2, 10.8))
    headline(fig, "Predicting jet-engine failure before it happens",
             "NASA C-MAPSS turbofan data  ·  remaining useful life  ·  cost-optimal maintenance")
    ax = fig.add_axes([0.05, 0.12, 0.56, 0.66])
    engine_plot(ax, traj, pick[2], kinds)
    ax.set_title(f"Held-out engine #{pick[2]}: models track its remaining life", fontsize=16)
    r = m1["results"][best]
    savings = 1 - dec["cbm_cost_per_cycle"] / dec["run_to_failure_cost_per_cycle"]
    tiles = [(f"{r['rmse']:.1f}", f"cycles RMSE on 100 test engines\n({SHORT[best]}, FD001)"),
             (f"{dec['best_threshold']:.0f}", "cycle RUL threshold that\nminimises maintenance cost"),
             (f"{savings:.0%}", "lower cost per cycle than\nrun-to-failure"),
             (f"{1 - dec['cbm_cost_per_cycle'] / dec['tbm_cost_per_cycle']:.0%}",
              "lower than the best\ntime-based PM schedule")]
    for i, (big, small) in enumerate(tiles):
        y0 = 0.68 - i * 0.165
        fig.patches.append(plt.Rectangle((0.66, y0), 0.30, 0.14, transform=fig.transFigure,
                                         facecolor=T.PANEL, edgecolor=T.GRID))
        fig.text(0.68, y0 + 0.07, big, fontsize=40, fontweight="bold", color=T.ACCENT, va="center")
        fig.text(0.80, y0 + 0.07, small, fontsize=15, color=T.TEXT, va="center")
    save(fig, "01_hero.png", media=True)

    # sensors
    fig = sensor_trends(prep)
    fig.set_size_inches(19.2, 10.8)
    fig.suptitle("")
    fig.tight_layout(rect=(0.02, 0.05, 0.98, 0.84))
    headline(fig, "21 sensors, 100 engines, run to failure",
             "Flat sensors are dropped; the rest drift as the engine wears out (median in orange)")
    save(fig, "02_sensor_degradation.png", media=True)

    # cost
    with plt.rc_context({"font.size": 18, "axes.titlesize": 20}):
        fig, ax = cost_fig((19.2, 10.8))
    fig.subplots_adjust(left=0.07, right=0.96, top=0.78, bottom=0.1)
    headline(fig, "From a prediction to a maintenance decision",
             "Pick the RUL threshold that balances unplanned failures against early replacement")
    save(fig, "03_maintenance_cost.png", media=True)

    # model comparison
    fig = plt.figure(figsize=(19.2, 10.8))
    headline(fig, "Baseline vs deep learning, honestly compared",
             "Test RMSE on all four C-MAPSS subsets vs a published deep CNN (lower is better)")
    ax = fig.add_axes([0.06, 0.1, 0.9, 0.68])
    for i, k in enumerate(kinds):
        vals = [metrics[fd]["results"][k]["rmse"] for fd in fds]
        bars = ax.bar(x + (i - 1) * w, vals, w * 0.92, color=T.MODEL_COLORS[k], label=SHORT[k])
        ax.bar_label(bars, fmt="%.1f", color=T.TEXT, fontsize=13, padding=3)
    ax.scatter(x + 2 * w, li, marker="D", s=120, color=T.TEXT, zorder=5, label="Published DCNN (Li et al. 2018)")
    ax.set_xticks(x, fds, fontsize=15)
    ax.set_ylabel("test RMSE (cycles)", fontsize=14)
    ax.legend(loc="upper left", ncol=4, fontsize=14)
    ax.grid(axis="x", visible=False)
    ax.set_ylim(0, ax.get_ylim()[1] * 1.18)
    save(fig, "04_model_comparison.png", media=True)


if __name__ == "__main__":
    main()
