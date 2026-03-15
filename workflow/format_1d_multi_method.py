"""
Format and validate the 1D multi method dataset.

Input (immutable):
  data/1d-multi-method-data.csv

Output (generated):
  workflow/processed/1d-multi-method-data.tidy.csv

Why this exists:
- The sprint project asks for a "format" step per dataset, even if the raw CSV is already close to tidy.
- This script makes schema, types, and validity constraints explicit and reproducible.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
import sys

import pandas as pd


@dataclass(frozen=True)
class Paths:
    project_root: Path
    raw_csv: Path
    tidy_csv: Path


def get_paths() -> Paths:
    project_root = Path(__file__).resolve().parents[1]
    raw_csv = project_root / "data" / "1d-multi-method-data.csv"
    tidy_csv = project_root / "workflow" / "processed" / "1d-multi-method-data.tidy.csv"
    return Paths(project_root=project_root, raw_csv=raw_csv, tidy_csv=tidy_csv)


def validate_columns(df: pd.DataFrame) -> None:
    required = {"method", "AUCROC"}
    missing = required - set(df.columns)
    if missing:
        raise ValueError(f"Missing required column(s): {sorted(missing)}. Found: {list(df.columns)}")


def clean_and_validate(df: pd.DataFrame) -> pd.DataFrame:
    validate_columns(df)

    out = df.copy()

    # Type normalization
    out["method"] = out["method"].astype(str)
    out["AUCROC"] = pd.to_numeric(out["AUCROC"], errors="coerce")

    # Drop missing values explicitly. This is safer than silently plotting NaNs.
    out = out.dropna(subset=["method", "AUCROC"]).reset_index(drop=True)

    # AUC-ROC must be within [0, 1]. If not, fail fast to avoid nonsense plots.
    if ((out["AUCROC"] < 0) | (out["AUCROC"] > 1)).any():
        bad = out.loc[(out["AUCROC"] < 0) | (out["AUCROC"] > 1), ["method", "AUCROC"]]
        raise ValueError(f"Found AUCROC values outside [0, 1]:\n{bad.to_string(index=False)}")

    # Create explicit names for downstream plotting.
    out = out.rename(columns={"method": "method_id", "AUCROC": "auc_roc"})
    out["method_label"] = out["method_id"].str.replace("Baseline_", "Baseline ", regex=False)
    out["method_label"] = out["method_label"].str.replace("_", " ", regex=False)

    # Sanity check. Expect 10 methods x 10 samples = 100 rows for this provided dataset.
    # This is not a hard requirement for correctness, but it flags accidental file mixups.
    unique_methods = out["method_id"].nunique()
    if unique_methods < 2:
        raise ValueError(f"Expected multiple methods. Found {unique_methods} unique method_id values.")

    return out[["method_id", "method_label", "auc_roc"]]


def main() -> int:
    paths = get_paths()
    if not paths.raw_csv.exists():
        print(f"ERROR: raw data not found at: {paths.raw_csv}", file=sys.stderr)
        return 2

    df_raw = pd.read_csv(paths.raw_csv)
    df_tidy = clean_and_validate(df_raw)

    paths.tidy_csv.parent.mkdir(parents=True, exist_ok=True)
    df_tidy.to_csv(paths.tidy_csv, index=False)

    print(f"Wrote tidy dataset: {paths.tidy_csv} (rows={len(df_tidy)}, methods={df_tidy['method_id'].nunique()})")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
