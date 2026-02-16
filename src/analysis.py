"""
Analysis functions (missingness, regression, correlations).
Refactored from Build 0.
"""

from typing import Optional, List, Dict, Any
import pandas as pd


def missingness_table(df: pd.DataFrame) -> pd.DataFrame:
    """
    Create a missingness table.
    Returns DataFrame with columns: column, missing_rate, missing_count
    Sorted by missing_rate descending.
    """
    missing_rate = df.isna().mean()
    missing_count = df.isna().sum()

    result = pd.DataFrame({
        "column": df.columns,
        "missing_rate": missing_rate.values,
        "missing_count": missing_count.values
    })

    result = result.sort_values("missing_rate", ascending=False).reset_index(drop=True)
    return result


def multiple_linear_regression(
    df: pd.DataFrame, outcome: str, predictors: Optional[List[str]] = None
) -> Dict[str, Any]:
    """
    Fit a multiple linear regression model.
    Returns a dictionary with model results.
    """
    import statsmodels.api as sm

    if df[outcome].dtype.kind not in "if":
        raise ValueError(
            f"Outcome must be numeric for linear regression: {outcome}"
        )

    if predictors is None:
        numeric_cols = df.select_dtypes(include=["number"]).columns.tolist()
        predictors = [c for c in numeric_cols if c != outcome]

    cols_needed = [outcome] + predictors
    df_clean = df[cols_needed].dropna()

    X = df_clean[predictors]
    X = sm.add_constant(X)
    y = df_clean[outcome]

    model = sm.OLS(y, X).fit()

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
        "coefficients": coefficients,
    }

    return result


def correlations(df: pd.DataFrame, numeric_cols: List[str]) -> pd.DataFrame:
    """Compute correlations for numeric columns."""
    if len(numeric_cols) < 2:
        return pd.DataFrame()
    corr = df[numeric_cols].corr()
    return corr
