"""Loading C-MAPSS files and labelling Remaining Useful Life (RUL)."""
from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
DATA_DIR = ROOT / "data" / "CMAPSS"

SETTINGS = ["setting1", "setting2", "setting3"]
SENSORS = [f"s{i}" for i in range(1, 22)]
COLUMNS = ["unit", "cycle", *SETTINGS, *SENSORS]

# Human-readable names from Saxena et al. (2008), Table 2.
SENSOR_NAMES = {
    "s1": "Fan inlet temp T2 (°R)", "s2": "LPC outlet temp T24 (°R)",
    "s3": "HPC outlet temp T30 (°R)", "s4": "LPT outlet temp T50 (°R)",
    "s5": "Fan inlet pressure P2 (psia)", "s6": "Bypass-duct pressure P15 (psia)",
    "s7": "HPC outlet pressure P30 (psia)", "s8": "Physical fan speed Nf (rpm)",
    "s9": "Physical core speed Nc (rpm)", "s10": "Engine pressure ratio epr",
    "s11": "HPC outlet static pressure Ps30 (psia)", "s12": "Fuel flow / Ps30 phi",
    "s13": "Corrected fan speed NRf (rpm)", "s14": "Corrected core speed NRc (rpm)",
    "s15": "Bypass ratio BPR", "s16": "Burner fuel-air ratio farB",
    "s17": "Bleed enthalpy htBleed", "s18": "Demanded fan speed Nf_dmd (rpm)",
    "s19": "Demanded corrected fan speed PCNfR_dmd (rpm)",
    "s20": "HPT coolant bleed W31 (lbm/s)", "s21": "LPT coolant bleed W32 (lbm/s)",
}

DEFAULT_CAP = 125


def _read(path: Path) -> pd.DataFrame:
    if not path.exists():
        raise FileNotFoundError(
            f"{path} not found. Run `python scripts/download_data.py` first."
        )
    df = pd.read_csv(path, sep=r"\s+", header=None, names=COLUMNS)
    df[["unit", "cycle"]] = df[["unit", "cycle"]].astype(int)
    return df


def load_subset(fd: str = "FD001", data_dir: Path | str = DATA_DIR):
    """Return (train, test, test_rul) for a subset.

    ``test_rul`` is a Series indexed by unit with the true RUL at the last
    observed cycle of each test engine.
    """
    data_dir = Path(data_dir)
    train = _read(data_dir / f"train_{fd}.txt")
    test = _read(data_dir / f"test_{fd}.txt")
    rul = pd.read_csv(data_dir / f"RUL_{fd}.txt", header=None, names=["rul"])
    rul.index = np.arange(1, len(rul) + 1)
    rul.index.name = "unit"
    return train, test, rul["rul"].astype(float)


def add_train_rul(df: pd.DataFrame, cap: float | None = DEFAULT_CAP) -> pd.DataFrame:
    """Label run-to-failure trajectories: RUL = last_cycle - cycle, clipped at ``cap``.

    The piecewise-linear cap reflects that an engine is "healthy" early in life and
    degradation only becomes observable later (Heimes 2008).
    """
    out = df.copy()
    last = out.groupby("unit")["cycle"].transform("max")
    out["rul"] = (last - out["cycle"]).astype(float)
    if cap is not None:
        out["rul"] = out["rul"].clip(upper=cap)
    return out


def add_test_rul(df: pd.DataFrame, test_rul: pd.Series, cap: float | None = None) -> pd.DataFrame:
    """Label every cycle of truncated test trajectories using the RUL at the final cycle."""
    out = df.copy()
    last = out.groupby("unit")["cycle"].transform("max")
    out["rul"] = (out["unit"].map(test_rul) + last - out["cycle"]).astype(float)
    if cap is not None:
        out["rul"] = out["rul"].clip(upper=cap)
    return out


def split_units(units, val_frac: float = 0.2, seed: int = 42):
    """Split engine IDs (not rows!) into train/validation to avoid leakage."""
    units = np.array(sorted(set(units)))
    rng = np.random.default_rng(seed)
    rng.shuffle(units)
    n_val = max(1, int(round(len(units) * val_frac)))
    return np.sort(units[n_val:]), np.sort(units[:n_val])
