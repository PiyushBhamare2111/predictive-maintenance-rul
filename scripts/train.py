"""Train baseline + deep models on C-MAPSS subsets, evaluate on the official test set,
and save artifacts to ``models/<FD>/``.

Examples:
    python scripts/train.py                     # FD001, all three models
    python scripts/train.py --fd all            # FD001-FD004
    python scripts/train.py --models gbm,cnn --epochs 30
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from pdm.pipeline import ARTIFACT_DIR, MODEL_LABELS, train_all  # noqa: E402


def main(argv=None):
    ap = argparse.ArgumentParser()
    ap.add_argument("--fd", default="FD001", help="FD001..FD004 or 'all'")
    ap.add_argument("--models", default="gbm,cnn,lstm")
    ap.add_argument("--epochs", type=int, default=60)
    ap.add_argument("--window", type=int, default=30)
    ap.add_argument("--cap", type=float, default=125)
    ap.add_argument("--device", default="auto", choices=["auto", "cpu"])
    ap.add_argument("--quiet", action="store_true")
    a = ap.parse_args(argv)
    fds = ["FD001", "FD002", "FD003", "FD004"] if a.fd == "all" else [a.fd.upper()]
    kinds = tuple(k.strip() for k in a.models.split(","))
    rows = []
    for fd in fds:
        s = train_all(fd, kinds, window=a.window, cap=a.cap, epochs=a.epochs,
                      device=a.device, verbose=not a.quiet)
        for k, r in s["results"].items():
            rows.append(f"| {fd} | {MODEL_LABELS[k]} | {r['rmse']:.2f} | {r['score']:.0f} | "
                        f"{r['train_seconds']:.0f}s ({r['device']}) |")
        d = s["decision"]
        print(f"\n{fd} decision layer ({d['model']}): threshold {d['best_threshold']:.0f} cycles, "
              f"cost/cycle {d['cbm_cost_per_cycle']:.0f} vs best age-based PM {d['tbm_cost_per_cycle']:.0f} "
              f"vs run-to-failure {d['run_to_failure_cost_per_cycle']:.0f}")
    print("\n| Subset | Model | Test RMSE | PHM score | Train time |\n|---|---|---|---|---|")
    print("\n".join(rows))
    print(f"\nArtifacts saved under {ARTIFACT_DIR}")


if __name__ == "__main__":
    main()
