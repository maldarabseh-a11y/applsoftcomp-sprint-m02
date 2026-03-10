from __future__ import annotations

import pathlib
import sys

import pandas as pd


DATA_PATH = pathlib.Path("data/1d-multi-method-data.csv")
OUT_DIR = pathlib.Path("workflow/processed")
OUT_PATH = OUT_DIR / "1d-multi-method-data.tidy.csv"

VALUE_COL_CANDIDATES = ["value", "aucroc", "auc_roc", "auc-roc", "auc roc"]


def load_raw(path: pathlib.Path) -> pd.DataFrame:
    if not path.exists():
        raise FileNotFoundError(f"Missing input file: {path}")

    df = pd.read_csv(path)

    # Normalize column names for robust matching
    df.columns = [c.strip().lower() for c in df.columns]

    if "method" not in df.columns:
        raise ValueError(f"Missing required column 'method'. Found: {list(df.columns)}")

    # Find the measurement column
    value_col = None
    for c in VALUE_COL_CANDIDATES:
        if c in df.columns:
            value_col = c
            break
    if value_col is None:
        raise ValueError(
            f"Missing measurement column. Expected one of {VALUE_COL_CANDIDATES}. Found: {list(df.columns)}"
        )

    # Clean method labels
    df["method"] = df["method"].astype(str).str.strip()

    # Coerce numeric measurement to a standard column name
    df["value"] = pd.to_numeric(df[value_col], errors="coerce")

    # Drop invalid rows
    before = len(df)
    df = df.dropna(subset=["method", "value"]).copy()
    after = len(df)
    if after < before:
        print(f"Dropped {before - after} rows with missing method/value", file=sys.stderr)

    # AUC-ROC must be within [0, 1]
    bad = (~df["value"].between(0, 1)).sum()
    if bad > 0:
        raise ValueError(f"Found {bad} rows with value outside [0, 1]. Refusing to proceed.")

    return df[["method", "value"]]


def main() -> None:
    df = load_raw(DATA_PATH)
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    df.to_csv(OUT_PATH, index=False)
    print(f"Wrote tidy data to {OUT_PATH}")


if __name__ == "__main__":
    main()