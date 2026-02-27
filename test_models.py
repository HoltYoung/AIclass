"""
test_models.py — Confirms that all refactored modules in src/ are importable and functional.

Run:  python test_models.py
"""

import pandas as pd
from pathlib import Path

# Test that all functions are importable from src/
from src import (
    ensure_dirs,
    read_data,
    basic_profile,
    split_columns,
    summarize_numeric,
    summarize_categorical,
    missingness_table,
    multiple_linear_regression,
    correlations,
    plot_missingness,
    plot_corr_heatmap,
    plot_histograms,
    plot_bar_charts,
    assert_json_safe,
    target_check,
)


def main():
    print("=== test_models.py ===\n")

    # 1) Read data
    data_path = Path("data/penguins.csv")
    df = read_data(data_path)
    print(f"[PASS] read_data: loaded {len(df)} rows, {len(df.columns)} columns")

    # 2) basic_profile
    profile = basic_profile(df)
    assert isinstance(profile, dict)
    assert "n_rows" in profile and "n_cols" in profile
    print(f"[PASS] basic_profile: {profile['n_rows']} rows, {profile['n_cols']} cols")

    # 3) split_columns
    numeric_cols, cat_cols = split_columns(df)
    assert len(numeric_cols) > 0 and len(cat_cols) > 0
    print(f"[PASS] split_columns: {len(numeric_cols)} numeric, {len(cat_cols)} categorical")

    # 4) summarize_numeric
    num_summary = summarize_numeric(df, numeric_cols)
    assert not num_summary.empty
    print(f"[PASS] summarize_numeric: {len(num_summary)} rows")

    # 5) summarize_categorical
    cat_summary = summarize_categorical(df, cat_cols)
    assert not cat_summary.empty
    print(f"[PASS] summarize_categorical: {len(cat_summary)} rows")

    # 6) missingness_table
    miss_df = missingness_table(df)
    assert "missing_rate" in miss_df.columns and "missing_count" in miss_df.columns
    print(f"[PASS] missingness_table: {len(miss_df)} rows")

    # 7) correlations
    corr = correlations(df, numeric_cols)
    assert not corr.empty
    print(f"[PASS] correlations: {corr.shape[0]}x{corr.shape[1]} matrix")

    # 8) assert_json_safe
    assert_json_safe(profile, context="basic_profile output")
    print("[PASS] assert_json_safe: profile is JSON-serializable")

    # 9) target_check
    target_info = target_check(df, "species")
    assert target_info is not None
    print(f"[PASS] target_check: checked 'species' column")

    # 10) multiple_linear_regression
    reg_results = multiple_linear_regression(df, outcome="body_mass_g")
    assert "r_squared" in reg_results
    print(f"[PASS] multiple_linear_regression: R² = {reg_results['r_squared']:.4f}")

    print("\n=== All module tests passed! ===")


if __name__ == "__main__":
    main()
