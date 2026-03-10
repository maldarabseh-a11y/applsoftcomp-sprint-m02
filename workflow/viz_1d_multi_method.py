"""
Create a figure for the 1D multi method dataset.

Outputs:
  figs/fig-1d-multi-method.png

Visualization goals:
- Use a non misleading plot that shows distribution (not just a mean).
- Use preattentive encodings to make "Proposed" pop against Baseline 1..9.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
import sys

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt


@dataclass(frozen=True)
class Paths:
    project_root: Path
    raw_csv: Path
    tidy_csv: Path
    out_png: Path


def get_paths() -> Paths:
    project_root = Path(__file__).resolve().parents[1]
    raw_csv = project_root / "data" / "1d-multi-method-data.csv"
    tidy_csv = project_root / "workflow" / "processed" / "1d-multi-method-data.tidy.csv"
    out_png = project_root / "figs" / "fig-1d-multi-method.png"
    return Paths(project_root=project_root, raw_csv=raw_csv, tidy_csv=tidy_csv, out_png=out_png)


def load_data(paths: Paths) -> pd.DataFrame:
    """
    Prefer the tidy file created by the format step.
    Fall back to the raw file with minimal validation if the tidy file is missing.
    """
    if paths.tidy_csv.exists():
        df = pd.read_csv(paths.tidy_csv)
        required = {"method_id", "method_label", "auc_roc"}
        missing = required - set(df.columns)
        if missing:
            raise ValueError(f"Tidy CSV missing required column(s): {sorted(missing)}")
        return df

    # Fallback: read raw and construct the expected columns
    if not paths.raw_csv.exists():
        raise FileNotFoundError(f"Raw data not found at: {paths.raw_csv}")

    df_raw = pd.read_csv(paths.raw_csv)
    if not {"method", "AUCROC"}.issubset(df_raw.columns):
        raise ValueError(f"Unexpected raw schema. Found columns: {list(df_raw.columns)}")

    df = df_raw.rename(columns={"method": "method_id", "AUCROC": "auc_roc"}).copy()
    df["method_id"] = df["method_id"].astype(str)
    df["auc_roc"] = pd.to_numeric(df["auc_roc"], errors="coerce")
    df = df.dropna(subset=["method_id", "auc_roc"]).reset_index(drop=True)

    if ((df["auc_roc"] < 0) | (df["auc_roc"] > 1)).any():
        raise ValueError("Found auc_roc values outside [0, 1]. Run the format step to inspect data.")

    df["method_label"] = df["method_id"].str.replace("Baseline_", "Baseline ", regex=False)
    df["method_label"] = df["method_label"].str.replace("_", " ", regex=False)
    return df[["method_id", "method_label", "auc_roc"]]


def method_order(df: pd.DataFrame) -> list[str]:
    """
    Put Proposed on top. Then sort baselines by median (descending) to make comparisons faster.
    """
    med = df.groupby("method_id")["auc_roc"].median().sort_values(ascending=False)
    baselines = [m for m in med.index.tolist() if m != "Proposed"]
    return ["Proposed"] + baselines


def plot(df: pd.DataFrame, out_png: Path) -> None:
    order = method_order(df)

    # Use matplotlib default cycle for the highlight color, but keep baselines neutral.
    cycle = plt.rcParams["axes.prop_cycle"].by_key().get("color", [])
    proposed_color = cycle[1] if len(cycle) > 1 else "C1"
    baseline_face = "0.85"
    baseline_point = "0.45"

    data = [df.loc[df["method_id"] == m, "auc_roc"].to_numpy() for m in order]

    fig, ax = plt.subplots(figsize=(8, 4.8), dpi=300)

    bp = ax.boxplot(
        data,
        vert=False,
        patch_artist=True,
        widths=0.55,
        medianprops=dict(linewidth=1.3),
        whiskerprops=dict(linewidth=1.0),
        capprops=dict(linewidth=1.0),
        flierprops=dict(marker="o", markersize=3, markerfacecolor="none", markeredgewidth=0.8),
    )

    # Style each box so Proposed becomes a preattentive anchor.
    for i, box in enumerate(bp["boxes"]):
        is_proposed = order[i] == "Proposed"
        box.set_facecolor(proposed_color if is_proposed else baseline_face)
        box.set_edgecolor("0.25")
        box.set_linewidth(2.0 if is_proposed else 1.0)

    for i, medline in enumerate(bp["medians"]):
        is_proposed = order[i] == "Proposed"
        medline.set_color("0.15")
        medline.set_linewidth(2.2 if is_proposed else 1.3)

    # Overlay all points so the viewer can see sample level variation.
    rng = np.random.default_rng(7)
    for idx, m in enumerate(order, start=1):
        x = df.loc[df["method_id"] == m, "auc_roc"].to_numpy()
        y = idx + rng.uniform(-0.12, 0.12, size=len(x))
        is_proposed = m == "Proposed"
        ax.scatter(
            x,
            y,
            s=18,
            alpha=0.9 if is_proposed else 0.55,
            color=proposed_color if is_proposed else baseline_point,
            edgecolor="none",
            zorder=3,
        )

    # Labels and scales. Use the full [0, 1] range to keep AUC ROC scale honest.
    ax.set_yticks(range(1, len(order) + 1))
    label_map = (
        df.drop_duplicates(subset=["method_id"])
        .set_index("method_id")["method_label"]
        .to_dict()
    )
    ax.set_yticklabels([label_map.get(m, m) for m in order])
    ax.invert_yaxis()

    ax.set_xlim(0, 1)
    ax.set_xlabel("AUC-ROC (higher is better)")
    ax.set_ylabel("Method")
    ax.set_title("Distribution of AUC-ROC across methods (10 samples each)")

    ax.grid(axis="x", linestyle="-", linewidth=0.6, alpha=0.35)

    # Direct annotation to reduce cognitive load.
    proposed_median = float(df.loc[df["method_id"] == "Proposed", "auc_roc"].median())
    ax.annotate(
        f"Proposed median = {proposed_median:.3f}",
        xy=(proposed_median, 1),
        xytext=(0.62, 2.1),
        arrowprops=dict(arrowstyle="->", linewidth=1.0, color="0.2"),
        fontsize=9,
        color="0.2",
    )

    out_png.parent.mkdir(parents=True, exist_ok=True)
    fig.tight_layout()
    fig.savefig(out_png, bbox_inches="tight")
    plt.close(fig)


def main() -> int:
    paths = get_paths()
    try:
        df = load_data(paths)
        plot(df, paths.out_png)
    except Exception as e:
        print(f"ERROR: {e}", file=sys.stderr)
        return 2

    print(f"Wrote figure: {paths.out_png}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
