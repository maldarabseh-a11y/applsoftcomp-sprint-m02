from __future__ import annotations

import pathlib
import subprocess
import sys

import pandas as pd
import matplotlib.pyplot as plt


TIDY_PATH = pathlib.Path("workflow/processed/2d-data.tidy.csv")
FIG_PATH = pathlib.Path("figs/fig-2d-data.png")


def ensure_tidy_exists() -> None:
    if TIDY_PATH.exists():
        return
    fmt_script = pathlib.Path("workflow/format_2d_data.py")
    if not fmt_script.exists():
        raise FileNotFoundError("Missing workflow/format_2d_data.py")
    subprocess.check_call([sys.executable, str(fmt_script)])


def main() -> None:
    ensure_tidy_exists()
    df = pd.read_csv(TIDY_PATH)

    required = {"x", "y"}
    missing = required - set(df.columns)
    if missing:
        raise ValueError(f"Tidy file missing {sorted(missing)}. Found: {list(df.columns)}")

    pathlib.Path("figs").mkdir(parents=True, exist_ok=True)

    fig, ax = plt.subplots(figsize=(6.5, 5))
    ax.scatter(df["x"], df["y"], alpha=0.7, s=25)
    ax.set_xlabel("x")
    ax.set_ylabel("y")
    ax.set_title("2D Data Scatter Plot")
    ax.grid(True, linestyle="--", linewidth=0.5, alpha=0.6)

    fig.tight_layout()
    fig.savefig(FIG_PATH, dpi=200)
    print(f"Wrote figure to {FIG_PATH}")


if __name__ == "__main__":
    main()