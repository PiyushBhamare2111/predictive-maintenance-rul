"""Streamlit app: Predictive Maintenance - Remaining Useful Life (NASA C-MAPSS).

Run:  streamlit run app.py   ->  http://localhost:8501
"""
from __future__ import annotations

import numpy as np
import pandas as pd
import plotly.graph_objects as go
import streamlit as st

from pdm import maintenance as M
from pdm import theme as T
from pdm.data import SENSOR_NAMES
from pdm.pipeline import ARTIFACT_DIR, available_subsets, load_artifacts, predict, prepare

st.set_page_config(page_title="Predictive Maintenance · RUL", page_icon="⚙️", layout="wide")
st.markdown(f"""
<style>
  .stApp {{ background: {T.BG}; }}
  section[data-testid="stSidebar"] {{ background: {T.PANEL}; }}
  h1, h2, h3 {{ letter-spacing: -0.01em; }}
  .rec {{ border-radius: 10px; padding: 14px 18px; border: 1px solid {T.GRID}; background: {T.PANEL}; }}
  .rec b {{ font-size: 1.25rem; }}
  [data-testid="stMetricValue"] {{ color: {T.ACCENT}; }}
</style>""", unsafe_allow_html=True)

SHORT = {"gbm": "Gradient boosting", "cnn": "1D-CNN", "lstm": "LSTM"}
REC_COLORS = {"MAINTAIN NOW": T.RED, "PLAN": "#eda100", "OK": T.AQUA}
REC_ICONS = {"MAINTAIN NOW": "🔴", "PLAN": "🟡", "OK": "🟢"}


@st.cache_resource(show_spinner="Loading models…")
def artifacts(fd):
    return load_artifacts(fd)


@st.cache_data(show_spinner="Preparing data…")
def prepared(fd):
    p = prepare(fd)
    return p.raw_train, p.raw_test, p.test, p.test_rul


@st.cache_data
def val_traj(fd):
    return pd.read_csv(ARTIFACT_DIR / fd / "val_trajectories.csv")


@st.cache_data(show_spinner="Running models on test engines…")
def test_traj(fd):
    metrics, pre, loaded = artifacts(fd)
    _, _, test, _ = prepared(fd)
    out = test[["unit", "cycle", "rul"]].rename(columns={"rul": "true_rul"}).sort_values(["unit", "cycle"])
    out = out.reset_index(drop=True)
    for k, m in loaded.items():
        out[f"pred_{k}"] = predict(k, m, test, pre.sensors, metrics["window"], cap=metrics["cap"])["pred_rul"].to_numpy()
    return out


def fig_layout(fig, title, h=380, **kw):
    fig.update_layout(**T.plotly_layout(title=dict(text=title, x=0, font=dict(size=16)), height=h, **kw))
    return fig


subsets = available_subsets()
st.sidebar.title("⚙️ Predictive Maintenance")
st.sidebar.caption("Remaining Useful Life of turbofan engines · NASA C-MAPSS")
page = st.sidebar.radio("Page", ["Engine explorer", "Maintenance policy", "Model comparison", "About"])
if not subsets:
    st.error("No trained models found in `models/`. Run `python scripts/train.py` first.")
    st.stop()
fd = st.sidebar.selectbox("Dataset subset", subsets, help="FD001/FD003: one operating condition. "
                          "FD002/FD004: six. FD003/FD004: two fault modes.")
metrics, pre, loaded = artifacts(fd)
dec = metrics["decision"]

# =========================================================== engine explorer
if page == "Engine explorer":
    st.title("Engine explorer")
    try:
        raw_train, raw_test, _, _ = prepared(fd)
    except FileNotFoundError:
        st.error("Raw data missing. Run `python scripts/download_data.py` then refresh.")
        st.stop()
    c1, c2, c3 = st.columns([1.4, 1, 1])
    source = c1.radio("Engines", ["Held-out engines (full life, run to failure)",
                                  "Official test engines (cut off before failure)"], key="source")
    is_val = source.startswith("Held")
    traj = val_traj(fd) if is_val else test_traj(fd)
    raw = raw_train if is_val else raw_test
    unit = c2.selectbox("Engine #", sorted(traj.unit.unique()))
    kinds = [k for k in ("gbm", "cnn", "lstm") if f"pred_{k}" in traj and k in loaded]
    model = c3.selectbox("Model for the recommendation", kinds, format_func=lambda k: SHORT[k],
                         index=kinds.index(dec["model"]) if dec["model"] in kinds else 0)
    g = traj[traj.unit == unit].sort_values("cycle")
    r = raw[raw.unit == unit].sort_values("cycle")
    last = int(g.cycle.max())
    start = int(g.cycle.min())
    default_cyc = max(start, int(last * 0.85)) if is_val else last
    cyc = st.slider("Replay the engine's life: current cycle", start, last, default_cyc)
    thr = st.sidebar.slider("Maintenance threshold (cycles)", 0, 100, int(dec["best_threshold"]),
                            help="Optimised on held-out engines on the Maintenance policy page.")
    now = g[g.cycle == cyc].iloc[0]
    pred_now = float(now[f"pred_{model}"])
    label, msg = M.recommend(pred_now, thr)
    m1, m2, m3, m4 = st.columns(4)
    m1.metric("Current cycle", f"{cyc}")
    m2.metric(f"Predicted RUL ({SHORT[model]})", f"{pred_now:.0f} cycles")
    m3.metric("True RUL", f"{now.true_rul:.0f} cycles")
    m4.metric("Error (pred − true)", f"{pred_now - now.true_rul:+.0f}")
    st.markdown(f'<div class="rec"><b style="color:{REC_COLORS[label]}">{REC_ICONS[label]} {label}</b>'
                f'<br>{msg}</div>', unsafe_allow_html=True)
    st.write("")

    fig = go.Figure()
    fig.add_trace(go.Scatter(x=g.cycle, y=g.true_rul, name="True RUL", line=dict(color=T.TRUE, width=3)))
    for k in kinds:
        fig.add_trace(go.Scatter(x=g.cycle, y=g[f"pred_{k}"], name=SHORT[k],
                                 line=dict(color=T.MODEL_COLORS[k], width=2)))
    fig.add_hline(y=thr, line=dict(color=T.RED, dash="dot"), annotation_text=f"threshold {thr}",
                  annotation_font_color=T.MUTED)
    fig.add_vline(x=cyc, line=dict(color=T.MUTED, dash="dash"))
    st.plotly_chart(fig_layout(fig, "Predicted vs true remaining useful life", 420,
                               xaxis_title="cycle", yaxis_title="RUL (cycles)"), width="stretch")

    default = [s for s in ("s2", "s4", "s11", "s15") if s in pre.sensors] or pre.sensors[:4]
    chosen = st.multiselect("Sensors", pre.sensors, default=default,
                            format_func=lambda s: f"{s} · {SENSOR_NAMES[s]}")
    cols = st.columns(2)
    for i, s in enumerate(chosen):
        f = go.Figure(go.Scatter(x=r.cycle, y=r[s], line=dict(color=T.ACCENT, width=1.6), name=s))
        if len(r) > 10:
            f.add_trace(go.Scatter(x=r.cycle, y=r[s].rolling(10, min_periods=1).mean(), name="10-cycle mean",
                                   line=dict(color=T.TEXT, width=1.5)))
        f.add_vline(x=cyc, line=dict(color=T.MUTED, dash="dash"))
        cols[i % 2].plotly_chart(fig_layout(f, SENSOR_NAMES[s], 280, showlegend=False),
                                 width="stretch")
    if not is_val:
        st.caption("Test engines are cut off before failure; the official truth is the RUL at the last "
                   "observed cycle, back-filled to earlier cycles.")

# =========================================================== maintenance policy
elif page == "Maintenance policy":
    st.title("Maintenance policy: when should we pull the engine?")
    st.markdown("Each held-out engine is replayed cycle by cycle. A work order is raised the first time "
                "the predicted RUL drops to the threshold **T**; the job happens after a **lead time**. "
                "If the engine fails first, we pay for an unplanned failure. Costs are compared **per "
                "operating cycle**, because maintaining earlier also means maintaining more often.")
    c = dec["cost_model"]
    a, b, cc, d = st.columns(4)
    c_fail = a.number_input("Unplanned failure cost ($)", 1000, 10_000_000, int(c["c_fail"]), step=10_000)
    c_pm = b.number_input("Planned replacement cost ($)", 100, 10_000_000, int(c["c_pm"]), step=5_000)
    lead = cc.slider("Lead time (cycles)", 0, 30, int(c["lead_time"]))
    kinds = [k for k in ("gbm", "cnn", "lstm") if k in metrics["results"]]
    model = d.selectbox("Model", kinds, format_func=lambda k: SHORT[k], index=kinds.index(dec["model"]))
    cost = M.CostModel(c_fail=c_fail, c_pm=c_pm, lead_time=lead)
    vt = val_traj(fd).rename(columns={f"pred_{model}": "pred_rul"})
    tab = M.policy_table(vt, np.arange(0, 101, 2), cost)
    t_best = M.best_threshold(tab)
    lives = vt.groupby("unit").cycle.max().to_numpy()
    ages = np.unique(np.linspace(0.4 * lives.min(), lives.max(), 80).astype(int))
    age = pd.DataFrame([M.simulate_age_policy(lives, x, cost) for x in ages])
    ab = age.loc[age.cost_per_cycle.idxmin()]
    rtf = c_fail * len(lives) / lives.sum()
    best = tab.set_index("threshold").loc[t_best]
    k1, k2, k3, k4 = st.columns(4)
    k1.metric("Optimal threshold", f"{t_best:.0f} cycles")
    k2.metric("CBM cost / cycle", f"${best.cost_per_cycle:,.0f}",
              f"{best.cost_per_cycle / rtf - 1:+.0%} vs run-to-failure", delta_color="inverse")
    k3.metric("Best time-based PM", f"${ab.cost_per_cycle:,.0f}", f"replace at age {ab.age:.0f}",
              delta_color="off")
    k4.metric("Unplanned failures at optimum", f"{best.fail_rate:.0%}")
    fig = go.Figure()
    fig.add_trace(go.Scatter(x=tab.threshold, y=tab.cost_per_cycle, name="Condition-based (predicted RUL ≤ T)",
                             line=dict(color=T.ACCENT, width=3)))
    fig.add_hline(y=ab.cost_per_cycle, line=dict(color=T.BLUE, dash="dash"),
                  annotation_text="best time-based PM", annotation_font_color=T.MUTED)
    fig.add_hline(y=rtf, line=dict(color=T.RED, dash="dot"), annotation_text="run to failure",
                  annotation_font_color=T.MUTED)
    fig.add_trace(go.Scatter(x=[t_best], y=[best.cost_per_cycle], mode="markers", name="optimum",
                             marker=dict(size=14, color=T.ACCENT, line=dict(color=T.TEXT, width=2))))
    st.plotly_chart(fig_layout(fig, "Expected cost per operating cycle", 420, xaxis_title="threshold T (cycles)",
                               yaxis_title="$ per cycle", yaxis_range=[0, rtf * 1.15]), width="stretch")
    f2 = go.Figure()
    f2.add_trace(go.Scatter(x=tab.threshold, y=tab.fail_rate * 100, name="unplanned failures (%)",
                            line=dict(color=T.RED, width=2.5)))
    f2.add_trace(go.Scatter(x=tab.threshold, y=tab.mean_rul_wasted, name="useful life wasted (cycles)",
                            line=dict(color=T.ACCENT, width=2.5)))
    st.plotly_chart(fig_layout(f2, "The trade-off behind the curve", 320, xaxis_title="threshold T"),
                    width="stretch")
    st.caption(f"Evaluated on {len(lives)} held-out run-to-failure engines of {fd} that the models never "
               "trained on. With this few engines the curve is noisy: treat the optimum as a starting point.")
    with st.expander("Policy table"):
        st.dataframe(tab.round(2), width="stretch", hide_index=True)

# =========================================================== model comparison
elif page == "Model comparison":
    st.title("Model comparison")
    rows = []
    for s in subsets:
        mm = artifacts(s)[0]
        for k, r in mm["results"].items():
            rows.append({"Subset": s, "Model": SHORT[k], "Test RMSE": r["rmse"], "PHM score": r["score"],
                         "MAE": r["mae"], "RMSE (truth capped 125)": r["rmse_capped_truth"],
                         "Train time (s)": r["train_seconds"], "Device": r["device"]})
        for name, (rm, sc) in mm["published"].items():
            rows.append({"Subset": s, "Model": f"Published: {name}", "Test RMSE": rm, "PHM score": sc})
    df = pd.DataFrame(rows)
    fig = go.Figure()
    for k in ("gbm", "cnn", "lstm"):
        d = df[df.Model == SHORT[k]]
        fig.add_trace(go.Bar(x=d.Subset, y=d["Test RMSE"], name=SHORT[k], marker_color=T.MODEL_COLORS[k],
                             text=d["Test RMSE"].round(1), textposition="outside"))
    d = df[df.Model == "Published: DCNN, Li et al. 2018"]
    fig.add_trace(go.Scatter(x=d.Subset, y=d["Test RMSE"], mode="markers", name="Published DCNN (Li 2018)",
                             marker=dict(symbol="diamond", size=13, color=T.TEXT)))
    st.plotly_chart(fig_layout(fig, "Test RMSE on the official test sets (lower is better)", 430,
                               barmode="group", hovermode="closest"), width="stretch")
    st.dataframe(df[df.Subset == fd].round(2), width="stretch", hide_index=True)
    st.markdown("**PHM score** = Σ exp(−d/13)−1 for early and exp(d/10)−1 for late predictions "
                "(d = predicted − true). Late predictions are punished harder because they cause failures.")
    hist = []
    for k in ("cnn", "lstm"):
        p = ARTIFACT_DIR / fd / f"{k}_history.csv"
        if p.exists():
            h = pd.read_csv(p)
            h["model"] = SHORT[k]
            hist.append(h)
    if hist:
        h = pd.concat(hist)
        f = go.Figure()
        for (mname, gg), col in zip(h.groupby("model"), (T.ACCENT, T.BLUE)):
            f.add_trace(go.Scatter(x=gg.epoch, y=gg.train_rmse, name=f"{mname} train", line=dict(color=col, dash="dot")))
            f.add_trace(go.Scatter(x=gg.epoch, y=gg.val_rmse, name=f"{mname} validation", line=dict(color=col)))
        st.plotly_chart(fig_layout(f, f"Training curves ({fd}, first seed)", 340, xaxis_title="epoch",
                                   yaxis_title="RMSE"), width="stretch")

# =========================================================== about
else:
    st.title("About this project")
    st.markdown(f"""
**Goal.** Predict how many more cycles (flights) a turbofan engine can run before failure (its
*remaining useful life*, RUL) from 21 sensor streams, then turn that prediction into a
maintenance decision that minimises cost.

**Data.** NASA C-MAPSS (Saxena et al., 2008): simulated engines that start healthy with random wear and
run until failure. Training engines are run to failure; test engines are cut off early and the true RUL at
the cut-off is given.

**Pipeline.** Drop flat sensors → z-score each sensor (per operating condition for FD002/FD004) →
label RUL with a piecewise-linear cap at {metrics['cap']} → {metrics['window']}-cycle sliding windows →
gradient boosting on window statistics, plus a 1D-CNN and an LSTM (3-seed ensembles) in PyTorch.

**Decision layer.** Condition-based maintenance: raise a work order when predicted RUL ≤ T, done after a
lead time. T is chosen to minimise expected cost per operating cycle on held-out engines and compared with
time-based preventive maintenance and run-to-failure.

Sensors used for {fd}: {', '.join(pre.sensors)}.
""")
