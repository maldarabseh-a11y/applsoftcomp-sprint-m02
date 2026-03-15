from __future__ import annotations

import pathlib
import sys
from typing import Optional

import pandas as pd


DATA_PATH = pathlib.Path("data/1d-data.csv")
OUT_DIR = pathlib.Path("workflow/processed")
OUT_PATH = OUT_DIR / "1d-data.tidy.csv"

VALUE_COL_CANDIDATES = ["value", "outcome", "y", "response", "measurement"]
GROUP_COL_CANDIDATES = ["group", "treatment", "condition", "arm", "cohort"]
ID_COL_CANDIDATES = ["subject", "id", "participant", "case_id", "sample_id"]


def _pick_first_existing(df: pd.DataFrame, candidates: list[str]) -> Optional[str]:
    for c in candidates:
        if c in df.columns:
            return c
    return None


def load_and_tidy(path: pathlib.Path) -> pd.DataFrame:
    if not path.exists():
        raise FileNotFoundError(f"Missing input file: {path}")

    df = pd.read_csv(path)

    # Normalize column names
    df.columns = [c.strip().lower() for c in df.columns]

    value_col = _pick_first_existing(df, VALUE_COL_CANDIDATES)
    group_col = _pick_first_existing(df, GROUP_COL_CANDIDATES)
    id_col = _pick_first_existing(df, ID_COL_CANDIDATES)

    if value_col is None:
        raise ValueError(
            f"Missing measurement column. Expected one of {VALUE_COL_CANDIDATES}. Found: {list(df.columns)}"
        )
    if group_col is None:
        raise ValueError(
            f"Missing group column. Expected one of {GROUP_COL_CANDIDATES}. Found: {list(df.columns)}"
        )

    # Clean group labels
    df[group_col] = df[group_col].astype(str).str.strip()

    # Coerce value to numeric
    df["value"] = pd.to_numeric(df[value_col], errors="coerce")

    # Drop rows missing value or group
    before = len(df)
    df = df.dropna(subset=["value", group_col]).copy()
    after = len(df)
    dropped = before - after
    if dropped > 0:
        print(f"Dropped {dropped} rows with missing value or group", file=sys.stderr)

    if df["value"].isna().any():
        n_bad = int(df["value"].isna().sum())
        raise ValueError(f"Found {n_bad} rows where value could not be parsed as numeric.")

    # Add subject id if not present
    if id_col is None:
        df["subject"] = range(1, len(df) + 1)
    else:
        df["subject"] = df[id_col]

    tidy = df[["subject", group_col, "value"]].rename(columns={group_col: "group"})

    # Basic integrity checks
    if tidy["group"].isna().any():
        raise ValueError("Group contains missing values after cleaning.")
    if tidy["value"].isna().any():
        raise ValueError("Value contains missing values after cleaning.")

    return tidy


def main() -> None:
    tidy = load_and_tidy(DATA_PATH)
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    tidy.to_csv(OUT_PATH, index=False)
    print(f"Wrote tidy dataset to {OUT_PATH}")


if __name__ == "__main__":
    main()
