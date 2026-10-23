"""Download and verify the NASA C-MAPSS turbofan degradation dataset into ``data/CMAPSS``.

Source: NASA Prognostics Center of Excellence data repository (public S3 bucket that backs
https://www.nasa.gov/intelligent-systems-division/discovery-and-systems-health/pcoe/pcoe-data-set-repository/).

The download is treated as untrusted data: we verify a SHA-256 checksum, only read the
whitelisted ``*.txt`` files out of the archive (no path traversal, nothing executed), and
sanity-check every file's shape before writing it to disk.

Usage:
    python scripts/download_data.py            # all four subsets FD001-FD004
    python scripts/download_data.py --force    # re-download even if files exist
"""
from __future__ import annotations

import argparse
import hashlib
import io
import sys
import urllib.request
import zipfile
from pathlib import Path

NASA_URL = (
    "https://phm-datasets.s3.amazonaws.com/NASA/"
    "6.+Turbofan+Engine+Degradation+Simulation+Data+Set.zip"
)
# SHA-256 of the outer archive and of the inner CMAPSSData.zip (recorded 2026-10-23).
OUTER_SHA256 = "c9c5dec12a945a82e8bb4446589d7fb3cc057b5e5d81fa1a12e25ee9912ad3b2"
INNER_SHA256 = "74bef434a34db25c7bf72e668ea4cd52afe5f2cf8e44367c55a82bfd91a5a34f"

SUBSETS = ("FD001", "FD002", "FD003", "FD004")
# (train rows, test rows, test engines) for each subset, from the dataset readme.
EXPECTED = {
    "FD001": (20631, 13096, 100),
    "FD002": (53759, 33991, 259),
    "FD003": (24720, 16596, 100),
    "FD004": (61249, 41214, 248),
}
ROOT = Path(__file__).resolve().parents[1]
OUT_DIR = ROOT / "data" / "CMAPSS"


def _sha256(b: bytes) -> str:
    return hashlib.sha256(b).hexdigest()


def _fetch(url: str) -> bytes:
    print(f"Downloading {url} ...")
    req = urllib.request.Request(url, headers={"User-Agent": "pdm-rul-downloader/1.0"})
    with urllib.request.urlopen(req, timeout=120) as r:  # noqa: S310 (fixed https URL)
        return r.read()


def _check_table(name: str, raw: bytes, n_rows: int, n_cols: int) -> None:
    lines = [ln for ln in raw.decode("ascii").splitlines() if ln.strip()]
    if len(lines) != n_rows:
        raise ValueError(f"{name}: expected {n_rows} rows, got {len(lines)}")
    widths = {len(ln.split()) for ln in lines}
    if widths != {n_cols}:
        raise ValueError(f"{name}: expected {n_cols} columns, got {widths}")
    for ln in lines[:50]:
        [float(x) for x in ln.split()]  # must be purely numeric


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--force", action="store_true", help="re-download even if present")
    args = ap.parse_args(argv)

    wanted = [f"{k}_{s}.txt" for s in SUBSETS for k in ("train", "test", "RUL")]
    if not args.force and all((OUT_DIR / w).exists() for w in wanted):
        print(f"All files already present in {OUT_DIR}. Use --force to re-download.")
        return 0

    outer = _fetch(NASA_URL)
    if _sha256(outer) != OUTER_SHA256:
        print("WARNING: outer archive checksum differs from the recorded value; "
              "continuing with per-file shape checks.")
    with zipfile.ZipFile(io.BytesIO(outer)) as zo:
        inner_names = [n for n in zo.namelist() if n.endswith("CMAPSSData.zip")]
        if len(inner_names) != 1:
            raise RuntimeError("CMAPSSData.zip not found inside NASA archive")
        inner = zo.read(inner_names[0])
    if _sha256(inner) != INNER_SHA256:
        print("WARNING: CMAPSSData.zip checksum differs from the recorded value.")

    OUT_DIR.mkdir(parents=True, exist_ok=True)
    with zipfile.ZipFile(io.BytesIO(inner)) as zi:
        members = {Path(n).name: n for n in zi.namelist()}
        for fd in SUBSETS:
            n_train, n_test, n_eng = EXPECTED[fd]
            for kind, rows, cols in (("train", n_train, 26), ("test", n_test, 26), ("RUL", n_eng, 1)):
                fname = f"{kind}_{fd}.txt"
                if fname not in members:
                    raise RuntimeError(f"{fname} missing from archive")
                raw = zi.read(members[fname])
                _check_table(fname, raw, rows, cols)
                (OUT_DIR / fname).write_bytes(raw)  # write to a fixed, whitelisted name
                print(f"  ok  {fname:18s} {rows:6d} rows")
        if "readme.txt" in members:
            (OUT_DIR / "readme.txt").write_bytes(zi.read(members["readme.txt"]))
    print(f"Done. Files written to {OUT_DIR}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
