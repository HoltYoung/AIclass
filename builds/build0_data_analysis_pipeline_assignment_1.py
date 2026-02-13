"""
Build 0: Data Analysis Pipeline ( ASSIGNMENT 1)

Completed by Holt Young
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Optional, Tuple, List, Dict, Any

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

# -----------------------------
# Utilities
# -----------------------------


def ensure_dirs(reports: Path) -> None:
    """Create output folders."""
    # BLANK 1: create the figures folder
    (reports / "figures").mkdir(parents=True, exist_ok=True)


def read_data(path: Path) -> pd.DataFrame:
    """Read a CSV file into a DataFrame with basic error handling."""
    # BLANK 2: raise FileNotFoundError if path does not exist
    if not path.exists():
        raise FileNotFoundError(f"File not found: {path}")

    # BLANK 3: read the CSV into df
    df = pd.read_csv(path)

    if df.empty:
        raise ValueError("Loaded dataframe is empty.")
    return df


# -----------------------------
# Data profiling
# -----------------------------


def basic_profile(df: pd.DataFrame) -> dict:
    """Return a basic JSON-serializable profile of the dataset."""
    return {
        "n_rows": int(df.shape[0]),
        "n_cols": int(df.shape[1]),
        # BLANK 4: list of column names
        "columns": df.columns.tolist(),
        "dtypes": {c: str(df[c].dtype) for c in df.columns},
        "n_missing_total": int(df.isna().sum().sum()),
        "missing_by_col": df.isna().sum().to_dict(),
        "memory_mb": float(df.memory_usage(deep=True).sum() / (1024**2)),
    }


def split_columns(df: pd.DataFrame) -> Tuple[List[str], List[str]]:
    """Identify and split numeric vs categorical columns into numeric and categorical lists."""
    # BLANK 5: list numeric column names
    numeric_cols = df.select_dtypes(include=["number"]).columns.tolist()

    # Treat everything else as categorical
    cat_cols = [c for c in df.columns if c not in numeric_cols]
    return numeric_cols, cat_cols


# -----------------------------
# Summaries
# -----------------------------


def summarize_numeric(df: pd.DataFrame, numeric_cols: List[str]) -> pd.DataFrame:
    """Compute descriptive statistics for numeric columns."""
    if not numeric_cols:
        return pd.DataFrame(
            columns=[
                "column",
                "count",
                "mean",
                "std",
                "min",
                "p25",
                "median",
                "p75",
                "max",
            ]
        )

    # BLANK 6: Create a transposed describe table with percentiles 0.25, 0.5, 0.75
    summary = df[numeric_cols].describe(percentiles=[0.25, 0.5, 0.75]).T

    summary = summary.rename(columns={"50%": "median", "25%": "p25", "75%": "p75"})
    summary.insert(0, "column", summary.index)
    summary.reset_index(drop=True, inplace=True)
    return summary


def summarize_categorical(
    df: pd.DataFrame, cat_cols: List[str], top_k: int = 10
) -> pd.DataFrame:
    """Compute descriptive statistics for categorical columns."""
    rows = []
    for c in cat_cols:
        series = df[c].astype("string")
        n = int(series.shape[0])
        n_missing = int(series.isna().sum())
        n_unique = int(series.nunique(dropna=True))

        # BLANK 7: top_k value counts (drop missing)
        top = series.dropna().value_counts().head(top_k)

        rows.append(
            {
                "column": c,
                "count": n,
                "missing": n_missing,
                "unique": n_unique,
                "top_values": "; ".join([f"{idx} ({val})" for idx, val in top.items()]),
            }
        )
    return pd.DataFrame(rows)


# -----------------------------
# REQUIRED: Student-built functions
# -----------------------------


def missingness_table(df: pd.DataFrame) -> pd.DataFrame:
    """
    Create a missingness table.
    Returns DataFrame with columns: column, missing_rate, missing_count
    Sorted by missing_rate descending
    """
    missing_rate = df.isna().mean()
    missing_count = df.isna().sum()
    
    result = pd.DataFrame({
        "column": df.columns,
        "missing_rate": missing_rate.values,
        "missing_count": missing_count.values
    })
    
    # Sort by missing_rate descending
    result = result.sort_values("missing_rate", ascending=False).reset_index(drop=True)
    return result


def multiple_linear_regression(
    df: pd.DataFrame, outcome: str, predictors: Optional[List[str]] = None
) -> Dict[str, Any]:
    """
    Fit a multiple linear regression model.
    """
    import statsmodels.api as sm
    
    # Check outcome is numeric
    if df[outcome].dtype.kind not in 'if':
        raise ValueError(f"Outcome must be numeric for linear regression: {outcome}")
    
    # If predictors is None, use all numeric columns except outcome
    if predictors is None:
        numeric_cols = df.select_dtypes(include=["number"]).columns.tolist()
        predictors = [c for c in numeric_cols if c != outcome]
    
    # Drop rows with missing values in outcome or predictors
    cols_needed = [outcome] + predictors
    df_clean = df[cols_needed].dropna()
    
    # Fit the model
    X = df_clean[predictors]
    X = sm.add_constant(X)
    y = df_clean[outcome]
    
    model = sm.OLS(y, X).fit()
    
    # Build JSON-safe result dictionary
    coefficients = {}
    for pred in predictors:
        coefficients[pred] = float(model.params[pred])
    
    result = {
        "outcome": str(outcome),
        "predictors": list(predictors),
        "n_rows_used": int(len(df_clean)),
        "r_squared": float(model.rsquared),
        "adj_r_squared": float(model.rsquared_adj),
        "intercept": float(model.params["const"]),
        "coefficients": coefficients
    }
    
    return result


def correlations(df: pd.DataFrame, numeric_cols: List[str]) -> pd.DataFrame:
    """Compute correlations for numeric columns."""
    if len(numeric_cols) < 2:
        return pd.DataFrame()
    # BLANK 8: compute correlation matrix for numeric columns
    corr = df[numeric_cols].corr()
    return corr


# -----------------------------
# Plots
# -----------------------------


def plot_missingness(miss_df: pd.DataFrame, out_path: Path, top_n: int = 30) -> None:
    """Plot missing data in a horizontal bar chart."""
    plot_df = miss_df.head(top_n).iloc[::-1]
    plt.figure()
    # BLANK 9: create a horizontal bar chart using column names and missing_rate
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


# -----------------------------
# Simple model check
# -----------------------------


def assert_json_safe(obj, context: str = "") -> None:
    """Assert that an object can be serialized to JSON."""
    try:
        json.dumps(obj)
    except TypeError as e:
        raise AssertionError(
            f"Object is not JSON-serializable{': ' + context if context else ''}.\n"
            f"Hint: Convert Pandas / NumPy types to native Python types like "
            f"(str, int, float, list, dict).\n"
            f"Original error: {e}"
        )


def target_check(df: pd.DataFrame, target: str) -> Optional[dict]:
    """Look at a target column and return basic information about it."""
    if target not in df.columns:
        print(f"Column '{target}' not found.")
        return None

    y = df[target]

    results: Dict[str, Any] = {}
    results["target"] = str(target)
    results["dtype"] = str(y.dtype)
    results["missing_rate"] = float(y.isna().mean())
    results["n_unique"] = int(y.nunique(dropna=True))

    if y.dtype.kind in "if":
        results["mean"] = float(y.mean())
        results["std"] = float(y.std())
        results["min"] = float(y.min())
        results["max"] = float(y.max())
    else:
        top = y.astype(str).value_counts().head(5)
        results["top_values"] = {str(k): int(v) for k, v in top.items()}

    assert_json_safe(results, context=f"target_check output for column '{target}'")
    return results


# -----------------------------
# Main pipeline
# -----------------------------


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--data", type=str, required=True, help="Path to CSV file")
    parser.add_argument("--target", type=str, default=None, help="target column")
    parser.add_argument(
        "--outcome", type=str, default=None, help="Optional outcome for regression"
    )
    parser.add_argument(
        "--predictors",
        type=str,
        default=None,
        help="Comma-separated predictors for regression",
    )
    parser.add_argument(
        "--report_dir", type=str, default="reports", help="Output directory"
    )
    args = parser.parse_args()

    report_dir = Path(args.report_dir)
    ensure_dirs(report_dir)

    df = read_data(Path(args.data))
    numeric_cols, cat_cols = split_columns(df)

    profile = basic_profile(df)
    miss_df = missingness_table(df)
    num_summary = summarize_numeric(df, numeric_cols)
    cat_summary = summarize_categorical(df, cat_cols)
    corr = correlations(df, numeric_cols)

    (report_dir / "data_profile.json").write_text(json.dumps(profile, indent=2))
    miss_df.to_csv(report_dir / "missingness_by_column.csv", index=False)
    num_summary.to_csv(report_dir / "summary_numeric.csv", index=False)
    cat_summary.to_csv(report_dir / "summary_categorical.csv", index=False)

    if not corr.empty:
        corr.to_csv(report_dir / "correlations.csv")

    plot_missingness(miss_df, report_dir / "figures" / "missingness.png")
    plot_corr_heatmap(corr, report_dir / "figures" / "corr_heatmap.png")
    plot_histograms(df, numeric_cols, report_dir / "figures")
    plot_bar_charts(df, cat_cols, report_dir / "figures")

    if args.target:
        target_info = target_check(df, args.target)
        (report_dir / "target_overview.json").write_text(
            json.dumps(target_info, indent=2)
        )

    if args.outcome:
        preds: Optional[List[str]] = None
        if args.predictors:
            # BLANK 10: parse comma-separated predictors into a list
            preds = [p.strip() for p in args.predictors.split(",") if p.strip()]

        reg_results = multiple_linear_regression(
            df, outcome=args.outcome, predictors=preds
        )
        assert_json_safe(reg_results, context="multiple_linear_regression output")
        (report_dir / "regression_results.json").write_text(
            json.dumps(reg_results, indent=2)
        )

    print(f"Build0 pipeline complete. Outputs saved to: {report_dir.resolve()}")


if __name__ == "__main__":
    main()
