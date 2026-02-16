"""
Plotting functions for the data analysis pipeline.
Refactored from Build 0.
"""

from typing import List
from pathlib import Path
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt


def plot_missingness(miss_df: pd.DataFrame, out_path: Path, top_n: int = 30) -> None:
    """Plot missing data in a horizontal bar chart."""
    plot_df = miss_df.head(top_n).iloc[::-1]
    plt.figure()
    plt.barh(plot_df["column"], plot_df["missing_rate"])
    plt.xlabel("Missing rate")
    plt.title(f"Top {min(top_n, len(miss_df))} columns by missingness")
    plt.tight_layout()
    plt.savefig(out_path, dpi=200)
    plt.close()


def plot_corr_heatmap(corr: pd.DataFrame, out_path: Path) -> None:
    """Create a heatmap of correlations."""
    if corr.empty:
        return
    plt.figure()
    plt.imshow(corr.values, aspect="auto")
    plt.colorbar()
    plt.xticks(range(len(corr.columns)), corr.columns, rotation=90, fontsize=7)
    plt.yticks(range(len(corr.index)), corr.index, fontsize=7)
    plt.title("Correlation heatmap (numeric columns)")
    plt.tight_layout()
    plt.savefig(out_path, dpi=200)
    plt.close()


def plot_histograms(
    df: pd.DataFrame, numeric_cols: List[str], fig_dir: Path, max_cols: int = 12
) -> None:
    """Plot histograms for numeric columns."""
    for c in numeric_cols[:max_cols]:
        series = df[c].dropna()
        if series.empty:
            continue
        plt.figure()
        plt.hist(series, bins=30)
        plt.title(f"Histogram: {c}")
        plt.xlabel(c)
        plt.ylabel("Count")
        plt.tight_layout()
        plt.savefig(fig_dir / f"hist_{c}.png", dpi=200)
        plt.close()


def plot_bar_charts(
    df: pd.DataFrame,
    cat_cols: List[str],
    fig_dir: Path,
    max_cols: int = 12,
    top_k: int = 20,
) -> None:
    """Plot bar charts for categorical columns."""
    for c in cat_cols[:max_cols]:
        series = df[c].astype("string").dropna()
        if series.empty:
            continue
        vc = series.value_counts().head(top_k)
        plt.figure()
        plt.bar(vc.index.astype(str), vc.values)
        plt.title(f"Top {min(top_k, len(vc))} values: {c}")
        plt.xticks(rotation=90, fontsize=7)
        plt.tight_layout()
        plt.savefig(fig_dir / f"bar_{c}.png", dpi=200)
        plt.close()
