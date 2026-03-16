from __future__ import annotations

import pathlib
import subprocess
import sys

import pandas as pd
import matplotlib.pyplot as plt


TIDY_PATH = pathlib.Path("workflow/processed/1d-data.tidy.csv")
FIG_PATH = pathlib.Path("figs/fig-1d-data.png")


def ensure_tidy_exists() -> None:
    """
    Visualization expects a tidy file produced by workflow/format_1d_data.py.
    If it's missing, run the formatter using the current Python interpreter.
    """
    if TIDY_PATH.exists():
        return

    fmt_script = pathlib.Path("workflow/format_1d_data.py")
    if not fmt_script.exists():
        raise FileNotFoundError("Missing workflow/format_1d_data.py. Format step must exist before viz.")

    print("Tidy dataset missing. Running formatter to create it.")
    subprocess.check_call([sys.executable, str(fmt_script)])


def choose_log_scale(values: pd.Series) -> bool:
    """
    Use log scale only when values span orders of magnitude.
    This avoids compressing small values or exaggerating large ones.
    """
    v = values.dropna()
    if v.empty:
        return False
    vmin = float(v.min())
    vmax = float(v.max())
    if vmin <= 0:
        return False
    return (vmax / vmin) >= 100  # two orders of magnitude


def main() -> None:
    ensure_tidy_exists()

    df = pd.read_csv(TIDY_PATH)

    # Expected schema from format step: subject, group, value
    required = {"subject", "group", "value"}
    missing = required - set(df.columns)
    if missing:
        raise ValueError(f"Tidy file missing required columns {sorted(missing)}. Found: {list(df.columns)}")

    df["group"] = df["group"].astype(str).str.strip()
    df["value"] = pd.to_numeric(df["value"], errors="coerce")
    df = df.dropna(subset=["group", "value"]).copy()

    # Order groups consistently if possible
    groups = sorted(df["group"].unique())
    preferred = ["control", "treatment", "case", "cases", "test"]
    # If labels contain control or treatment words, put control first
    def group_key(g: str) -> tuple[int, str]:
        gl = g.lower()
        for i, p in enumerate(preferred):
            if p in gl:
                return (i, g)
        return (len(preferred), g)

    groups = sorted(groups, key=group_key)
    df["group"] = pd.Categorical(df["group"], categories=groups, ordered=True)

    # Plot. Jittered scatter + boxplot overlay (non-misleading distribution view)
    pathlib.Path("figs").mkdir(parents=True, exist_ok=True)

    fig, ax = plt.subplots(figsize=(8, 4.5))

    # Jittered points
    x_codes = df["group"].cat.codes.to_numpy()
    y = df["value"].to_numpy()

    # Deterministic jitter (no random seed needed)
    jitter = ((df["subject"].astype(str).str.len().to_numpy() % 7) - 3) * 0.03
    ax.scatter(x_codes + jitter, y, alpha=0.75, s=30)

    # Boxplot per group (shows median and spread)
    data_by_group = [df.loc[df["group"] == g, "value"].to_numpy() for g in groups]
    ax.boxplot(
        data_by_group,
        positions=list(range(len(groups))),
        widths=0.5,
        showfliers=False,
        medianprops={"linewidth": 2},
    )

    ax.set_xticks(list(range(len(groups))))
    ax.set_xticklabels(groups)
    ax.set_xlabel("Group")
    ax.set_ylabel("Value")

    # Log scale only if warranted
    if choose_log_scale(df["value"]):
        ax.set_yscale("log")
        ax.set_ylabel("Value (log scale)")

    ax.grid(True, axis="y", linestyle="--", linewidth=0.5, alpha=0.6)
    ax.set_title("1D Data: Treatment vs Control")

    fig.tight_layout()
    fig.savefig(FIG_PATH, dpi=200)
    print(f"Wrote figure to {FIG_PATH}")


if __name__ == "__main__":
    main()
