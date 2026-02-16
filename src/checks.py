"""
Validation and checking functions for the data analysis pipeline.
Refactored from Build 0.
"""

import json
from typing import Optional, Dict, Any
import pandas as pd


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
