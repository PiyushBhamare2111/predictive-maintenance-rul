"""Turning RUL predictions into a maintenance decision.

Policy (condition-based maintenance, CBM): raise a work order the first cycle the
predicted RUL drops to or below a threshold ``T``. Like in a real plant, the job can't be
done instantly: it is executed ``lead_time`` cycles later (parts, crew, a hangar slot).
If the engine fails inside that lead time, it counts as an unplanned failure.

Cost model per engine life (a renewal cycle):
* If the policy triggers before the engine fails -> planned replacement cost ``c_pm``
  plus the value of the useful life thrown away (``c_waste`` per cycle of true RUL left).
* If the engine fails first -> unplanned failure cost ``c_fail`` (secondary damage,
  unscheduled downtime, expediting parts).

Because a policy that maintains earlier also means *more* maintenance events per year,
the fair comparison is cost per operating cycle (renewal-reward theorem):
    cost rate = sum(cost) / sum(cycles operated)
We also report the classic time-based PM baseline (replace every engine at a fixed age)
so the value of condition monitoring is visible.
"""
from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd


@dataclass
class CostModel:
    c_fail: float = 100_000.0   # unplanned failure (e.g. in-service shutdown + collateral)
    c_pm: float = 20_000.0      # planned replacement / overhaul
    c_waste: float = 0.0        # optional: value per cycle of discarded useful life
    lead_time: int = 10         # cycles needed to get parts + a maintenance slot after the alert


def simulate_policy(traj: pd.DataFrame, threshold: float, cost: CostModel = CostModel()) -> pd.DataFrame:
    """Simulate the CBM policy on run-to-failure trajectories.

    ``traj`` must have columns unit, cycle, true_rul (uncapped) and pred_rul, one row per
    cycle, covering each engine's full life up to failure.
    Returns one row per engine with the outcome.
    """
    rows = []
    for unit, g in traj.sort_values(["unit", "cycle"]).groupby("unit"):
        hit = g[g["pred_rul"] <= threshold]
        life = int(g["cycle"].max())
        r = hit.iloc[0] if len(hit) else None
        if r is not None and float(r["true_rul"]) >= cost.lead_time:
            left = float(r["true_rul"]) - cost.lead_time   # life left when the job is done
            rows.append({"unit": unit, "failed": False, "alert_cycle": int(r["cycle"]),
                         "cycles_run": int(r["cycle"]) + cost.lead_time,
                         "life": life, "rul_wasted": left,
                         "cost": cost.c_pm + cost.c_waste * left})
        else:
            rows.append({"unit": unit, "failed": True,
                         "alert_cycle": int(r["cycle"]) if r is not None else None,
                         "cycles_run": life, "life": life,
                         "rul_wasted": 0.0, "cost": cost.c_fail})
    return pd.DataFrame(rows)


def simulate_age_policy(lives: np.ndarray, age: float, cost: CostModel = CostModel()) -> dict:
    """Time-based PM: replace at fixed ``age`` cycles (or at failure, whichever first)."""
    lives = np.asarray(lives, float)
    failed = lives <= age
    cycles = np.where(failed, lives, age)
    costs = np.where(failed, cost.c_fail, cost.c_pm + cost.c_waste * (lives - age))
    return {"age": age, "fail_rate": float(failed.mean()),
            "cost_per_cycle": float(costs.sum() / cycles.sum()),
            "cost_per_engine": float(costs.mean())}


def policy_table(traj: pd.DataFrame, thresholds, cost: CostModel = CostModel()) -> pd.DataFrame:
    out = []
    for t in thresholds:
        r = simulate_policy(traj, t, cost)
        out.append({
            "threshold": float(t),
            "fail_rate": float(r["failed"].mean()),
            "mean_rul_wasted": float(r["rul_wasted"].mean()),
            "cost_per_engine": float(r["cost"].mean()),
            "cost_per_cycle": float(r["cost"].sum() / r["cycles_run"].sum()),
        })
    return pd.DataFrame(out)


def best_threshold(table: pd.DataFrame) -> float:
    return float(table.loc[table["cost_per_cycle"].idxmin(), "threshold"])


def recommend(pred_rul: float, threshold: float, warn_margin: float = 20.0) -> tuple[str, str]:
    """Traffic-light recommendation for a single engine's current predicted RUL."""
    if pred_rul <= threshold:
        return "MAINTAIN NOW", f"Predicted RUL {pred_rul:.0f} ≤ threshold {threshold:.0f} cycles: schedule replacement at next opportunity."
    if pred_rul <= threshold + warn_margin:
        return "PLAN", f"Predicted RUL {pred_rul:.0f} cycles is within {warn_margin:.0f} of the threshold: order parts, book the slot."
    return "OK", f"Predicted RUL {pred_rul:.0f} cycles: keep running, continue monitoring."
