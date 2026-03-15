from __future__ import annotations

import pathlib
import sys

import pandas as pd


DATA_PATH = pathlib.Path("data/digits-data.csv")
OUT_DIR = pathlib.Path("workflow/processed")
OUT_PATH = OUT_DIR / "digits.tidy.csv"


def main() -> None:
    if not DATA_PATH.exists():
        raise FileNotFoundError(f"Missing {DATA_PATH}")

    df = pd.read_csv(DATA_PATH)
    df.columns = [c.strip().lower() for c in df.columns]

    # Expect a label column named digit (common in this sprint)
    if "digit" not in df.columns:
        raise ValueError(f"Missing 'digit' column. Found: {list(df.columns)}")

    df["digit"] = pd.to_numeric(df["digit"], errors="coerce")
    if df["digit"].isna().any():
        raise ValueError("Some digit labels could not be parsed as numeric.")
    if not df["digit"].between(0, 9).all():
        raise ValueError("Digit labels must be in [0, 9].")

    # Coerce remaining columns to numeric features
    feature_cols = [c for c in df.columns if c != "digit"]
    for c in feature_cols:
        df[c] = pd.to_numeric(df[c], errors="coerce")

    before = len(df)
    df = df.dropna().copy()
    dropped = before - len(df)
    if dropped > 0:
        print(f"Dropped {dropped} rows with missing values after coercion.", file=sys.stderr)

    OUT_DIR.mkdir(parents=True, exist_ok=True)
    df.to_csv(OUT_PATH, index=False)
    print(f"Wrote tidy digits dataset to {OUT_PATH}")


if __name__ == "__main__":
    main()