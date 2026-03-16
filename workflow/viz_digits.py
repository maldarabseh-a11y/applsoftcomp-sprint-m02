from __future__ import annotations

import pathlib
import subprocess
import sys

import pandas as pd
import matplotlib.pyplot as plt
from sklearn.decomposition import PCA


TIDY_PATH = pathlib.Path("workflow/processed/digits.tidy.csv")
FIG_PATH = pathlib.Path("figs/fig-digits.png")


def ensure_tidy_exists() -> None:
    if TIDY_PATH.exists():
        return
    fmt = pathlib.Path("workflow/format_digits.py")
    if not fmt.exists():
        raise FileNotFoundError("Missing workflow/format_digits.py")
    subprocess.check_call([sys.executable, str(fmt)])


def main() -> None:
    ensure_tidy_exists()
    df = pd.read_csv(TIDY_PATH)

    if "digit" not in df.columns:
        raise ValueError("Expected 'digit' column in tidy digits data.")

    y = df["digit"].astype(int)
    X = df.drop(columns=["digit"]).to_numpy()

    pca = PCA(n_components=2)
    Z = pca.fit_transform(X)

    pathlib.Path("figs").mkdir(parents=True, exist_ok=True)
    fig, ax = plt.subplots(figsize=(7, 5.5))
    sc = ax.scatter(Z[:, 0], Z[:, 1], c=y, s=10, alpha=0.8)
    ax.set_xlabel("PC1")
    ax.set_ylabel("PC2")
    ax.set_title("Digits dataset (PCA projection)")
    cbar = fig.colorbar(sc, ax=ax)
    cbar.set_label("Digit")
    fig.tight_layout()
    fig.savefig(FIG_PATH, dpi=200)
    print(f"Wrote figure to {FIG_PATH}")


if __name__ == "__main__":
    main()