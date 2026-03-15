from __future__ import annotations

import pathlib
import sys
from typing import Optional

import pandas as pd


DATA_PATH = pathlib.Path("data/2d-data.csv")
OUT_DIR = pathlib.Path("workflow/processed")
OUT_PATH = OUT_DIR / "2d-data.tidy.csv"

X_COL_CANDIDATES = ["x", "x_coord", "xcoordinate", "x-coordinate"]
Y_COL_CANDIDATES = ["y", "y_coord", "ycoordinate", "y-coordinate"]


def _pick_first_existing(df: pd.DataFrame, candidates: list[str]) -> Optional[str]:
    for c in candidates:
        if c in df.columns:
            return c
    return None


def load_and_tidy(path: pathlib.Path) -> pd.DataFrame:
    if not path.exists():
        raise FileNotFoundError(f"Missing input file: {path}")

    df = pd.read_csv(path)

    # Normalize column names for robust matching
    df.columns = [c.strip().lower() for c in df.columns]

    xcol = _pick_first_existing(df, X_COL_CANDIDATES)
    ycol = _pick_first_existing(df, Y_COL_CANDIDATES)

    if xcol is None or ycol is None:
        raise ValueError(
            f"Missing required x/y columns. "
            f"Expected x in {X_COL_CANDIDATES} and y in {Y_COL_CANDIDATES}. "
            f"Found: {list(df.columns)}"
        )

    tidy = df[[xcol, ycol]].rename(columns={xcol: "x", ycol: "y"}).copy()

    # Coerce to numeric
    tidy["x"] = pd.to_numeric(tidy["x"], errors="coerce")
    tidy["y"] = pd.to_numeric(tidy["y"], errors="coerce")

    before = len(tidy)
    tidy = tidy.dropna(subset=["x", "y"]).copy()
    dropped = before - len(tidy)
    if dropped > 0:
        print(f"Dropped {dropped} rows with non-numeric or missing x/y.", file=sys.stderr)

    if tidy.empty:
        raise ValueError("No valid rows remain after cleaning. Check input data/2d-data.csv.")

    return tidy


def main() -> None:
    tidy = load_and_tidy(DATA_PATH)
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    tidy.to_csv(OUT_PATH, index=False)
    print(f"Wrote tidy dataset to {OUT_PATH}")


if __name__ == "__main__":
    main()