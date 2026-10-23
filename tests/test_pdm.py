import math
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from pdm.data import COLUMNS, SENSORS, add_test_rul, add_train_rul, split_units  # noqa: E402
from pdm.evaluate import phm_score, rmse  # noqa: E402
from pdm.features import Preprocessor, make_sequences, unit_windows, window_features  # noqa: E402
from pdm.maintenance import CostModel, policy_table, simulate_age_policy, simulate_policy  # noqa: E402


def fake_fleet(n_units=4, min_life=40, seed=0):
    """Synthetic C-MAPSS-shaped data: a few sensors drift towards failure, others constant."""
    rng = np.random.default_rng(seed)
    rows = []
    for u in range(1, n_units + 1):
        life = min_life + 10 * u
        for c in range(1, life + 1):
            health = c / life
            sensors = [100.0 + health ** 2 * 5 + rng.normal(0, 0.3) if i % 3 == 0 else 1.0
                       for i in range(21)]
            rows.append([u, c, 0.0, 0.0, 100.0, *sensors])
    return pd.DataFrame(rows, columns=COLUMNS)


# ------------------------------------------------------------------ RUL labelling
def test_train_rul_linear_and_capped():
    df = pd.DataFrame({"unit": [1] * 200, "cycle": np.arange(1, 201)})
    out = add_train_rul(df, cap=125)
    assert out["rul"].iloc[-1] == 0
    assert out["rul"].iloc[-2] == 1
    assert out["rul"].max() == 125
    assert out["rul"].iloc[0] == 125            # 199 capped to 125
    assert out.loc[out.cycle == 100, "rul"].item() == 100
    unc = add_train_rul(df, cap=None)
    assert unc["rul"].iloc[0] == 199


def test_train_rul_per_unit():
    df = pd.DataFrame({"unit": [1, 1, 1, 2, 2], "cycle": [1, 2, 3, 1, 2]})
    assert add_train_rul(df, cap=None)["rul"].tolist() == [2, 1, 0, 1, 0]


def test_test_rul_offsets_from_truth():
    df = pd.DataFrame({"unit": [1, 1, 1, 2], "cycle": [1, 2, 3, 1]})
    truth = pd.Series([10.0, 5.0], index=[1, 2])
    assert add_test_rul(df, truth)["rul"].tolist() == [12, 11, 10, 5]


def test_split_units_disjoint():
    tr, va = split_units(range(1, 101), 0.2, seed=1)
    assert len(va) == 20 and len(tr) == 80 and not set(tr) & set(va)


# ------------------------------------------------------------------ windowing
def test_unit_windows_shape_and_padding():
    arr = np.arange(10, dtype=float).reshape(5, 2)    # 5 cycles, 2 features
    w = unit_windows(arr, 3)
    assert w.shape == (5, 3, 2)
    np.testing.assert_array_equal(w[0, :, 0], [0, 0, 0])      # front-padded
    np.testing.assert_array_equal(w[-1, :, 0], [4, 6, 8])     # last 3 cycles


def test_make_sequences_shapes():
    df = add_train_rul(fake_fleet(3), cap=125)
    feats = ["s1", "s4"]
    X, y, idx = make_sequences(df, feats, window=30)
    assert X.shape == (len(df), 30, 2) and y.shape == (len(df),)
    X, y, idx = make_sequences(df, feats, window=30, last_only=True)
    assert X.shape == (3, 30, 2) and (y == 0).all()
    assert idx["unit"].tolist() == [1, 2, 3]


def test_short_trajectory_padded():
    df = add_train_rul(fake_fleet(1, min_life=5), cap=125)   # 15 cycles < window 30
    X, _, _ = make_sequences(df, ["s1"], window=30, last_only=True)
    assert X.shape == (1, 30, 1)


def test_window_features_slope():
    df = pd.DataFrame({"unit": 1, "cycle": np.arange(1, 41), "s1": np.arange(40) * 2.0, "rul": 0.0})
    f, _, _ = window_features(df, ["s1"], window=10, last_only=True)
    assert f["s1_slope"].item() == pytest.approx(2.0)
    assert f["s1_last"].item() == 78 and f["cycle"].item() == 40


def test_preprocessor_drops_constant_sensors_and_roundtrips():
    df = fake_fleet(4)
    pre = Preprocessor().fit(df)
    assert pre.sensors == [s for i, s in enumerate(SENSORS) if i % 3 == 0]
    z = pre.transform(df)
    assert abs(z[pre.sensors].mean().mean()) < 1e-6
    pre2 = Preprocessor.from_json(pre.to_json())
    pd.testing.assert_frame_equal(z, pre2.transform(df))


# ------------------------------------------------------------------ metrics
def test_phm_score_hand_computed():
    # d = -13 -> e^1 - 1 ; d = +10 -> e^1 - 1 ; d = 0 -> 0
    assert phm_score([50], [37]) == pytest.approx(math.e - 1)
    assert phm_score([50], [60]) == pytest.approx(math.e - 1)
    assert phm_score([50], [50]) == 0
    # late predictions cost more than equally-early ones
    assert phm_score([50], [70]) > phm_score([50], [30])
    total = phm_score([10, 20, 30], [0, 25, 30])  # d = -10, +5, 0
    assert total == pytest.approx((math.exp(10 / 13) - 1) + (math.exp(0.5) - 1))


def test_rmse():
    assert rmse([0, 0], [3, 4]) == pytest.approx(math.sqrt(12.5))


# ------------------------------------------------------------------ maintenance
def test_policy_outcomes():
    cost = CostModel(c_fail=100, c_pm=20, lead_time=2)
    traj = pd.DataFrame({"unit": 1, "cycle": [1, 2, 3, 4, 5],
                         "true_rul": [4, 3, 2, 1, 0], "pred_rul": [9, 6, 3, 2, 1]})
    r = simulate_policy(traj, threshold=3, cost=cost).iloc[0]   # alert at cycle 3, true RUL 2
    assert not r.failed and r.cost == 20 and r.cycles_run == 5 and r.rul_wasted == 0
    r = simulate_policy(traj, threshold=2, cost=cost).iloc[0]   # alert at cycle 4, RUL 1 < lead
    assert r.failed and r.cost == 100
    tab = policy_table(traj, [2, 3], cost)
    assert tab["fail_rate"].tolist() == [1.0, 0.0]


def test_age_policy():
    out = simulate_age_policy([100, 200], age=150, cost=CostModel(c_fail=100, c_pm=20))
    assert out["fail_rate"] == 0.5
    assert out["cost_per_cycle"] == pytest.approx(120 / 250)


# ------------------------------------------------------------------ training smoke test
def test_training_smoke(tmp_path):
    pytest.importorskip("torch")
    from pdm.pipeline import train_all

    data = tmp_path / "data"
    data.mkdir()
    tr, te = fake_fleet(6, seed=1), fake_fleet(3, seed=2)
    te = te[te.cycle <= te.groupby("unit").cycle.transform("max") - 7]
    for name, df in (("train_FD001.txt", tr), ("test_FD001.txt", te)):
        df.to_csv(data / name, sep=" ", header=False, index=False)
    (data / "RUL_FD001.txt").write_text("7\n7\n7\n")

    import pdm.pipeline as pl

    orig = pl.load_subset
    pl.load_subset = lambda fd, d=None: orig(fd, data)
    try:
        s = train_all("FD001", kinds=("gbm", "cnn", "lstm"), epochs=1, out_dir=tmp_path / "m",
                      verbose=False, device="cpu", n_seeds=1)
    finally:
        pl.load_subset = orig
    for k in ("gbm", "cnn", "lstm"):
        assert np.isfinite(s["results"][k]["rmse"])
    assert (tmp_path / "m" / "FD001" / "cnn.pt").exists()
    metrics, pre, loaded = pl.load_artifacts("FD001", tmp_path / "m")
    assert set(loaded) == {"gbm", "cnn", "lstm"}
