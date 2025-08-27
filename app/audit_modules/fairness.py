"""
Fairness module: Demographic Parity Difference.

Given:
- df: DataFrame
- outcome_column: binary outcome column (1 = positive)
- protected_attributes: list of column names to audit

For each protected attribute:
- Compute positive outcome rate per group.
- Demographic Parity Difference = max_rate - min_rate (lower is better).
- Convert to fairness score in [0,1] as score = 1 - min(1, dpd).

The final module score is 1 - average_dpd (clamped to [0,1]).
Outputs include per-attribute group rates and differences.
"""

from typing import Any, Dict, List

import numpy as np
import pandas as pd


def _binary_to_01(series: pd.Series) -> pd.Series:
    """Normalize a binary-like series to {0,1} ints."""
    if series.dtype == bool:
        return series.astype(int)
    if pd.api.types.is_numeric_dtype(series):
        # Assume already 0/1 numeric; coerce others to NA then fill with mode
        vals = series.astype(float)
        uniq = set(pd.Series(series.dropna().unique()).astype(float).tolist())
        if uniq.issubset({0.0, 1.0}):
            return vals.astype(int)
    # Try common string forms
    lowered = series.astype(str).str.strip().str.lower()
    mapped = lowered.map({"yes": 1, "no": 0, "true": 1, "false": 0})
    if mapped.notna().mean() > 0.9:
        return mapped.fillna(0).astype(int)
    # Fallback: threshold numeric-like
    try:
        numeric = pd.to_numeric(series, errors="coerce")
        med = numeric.median()
        return (numeric >= med).astype(int)
    except Exception:
        # Last resort: majority class as 0, others as 1
        mode = series.mode(dropna=True)
        if len(mode) > 0:
            return (series != mode.iloc[0]).astype(int)
        return pd.Series([0] * len(series))


def calculate_demographic_parity(
    df: pd.DataFrame,
    outcome_column: str,
    protected_attributes: List[str],
) -> Dict[str, Any]:
    """
    Compute Demographic Parity metrics across protected attributes.

    Returns:
        {
          "score": float,
          "attributes": {
              attr: {
                 "group_positive_rates": {group: rate, ...},
                 "parity_difference": float
              }, ...
          }
        }
    """
    if df is None or df.empty:
        raise ValueError("Dataframe must be non-empty for fairness analysis.")
    if outcome_column not in df.columns:
        raise ValueError(f"Outcome column '{outcome_column}' not found in dataframe.")
    missing = [c for c in protected_attributes if c not in df.columns]
    if missing:
        raise ValueError(f"Protected attributes not found in dataframe: {missing}")

    y = _binary_to_01(df[outcome_column])
    results = {}
    dp_diffs = []

    for attr in protected_attributes:
        groups = df[attr].astype(str).fillna("NA_GROUP")
        rates = {}
        for g, idx in groups.groupby(groups).groups.items():
            grp = y.loc[idx]
            if len(grp) == 0:
                rate = 0.0
            else:
                rate = float(grp.mean())
            rates[g] = rate
        if rates:
            max_rate = max(rates.values())
            min_rate = min(rates.values())
            dp_diff = float(max_rate - min_rate)
        else:
            dp_diff = 0.0
        dp_diffs.append(dp_diff)
        results[attr] = {
            "group_positive_rates": rates,
            "parity_difference": dp_diff,
        }

    avg_diff = float(np.mean(dp_diffs)) if dp_diffs else 0.0
    # Score is 1 - avg_diff, clipped to [0,1]
    score = float(np.clip(1.0 - avg_diff, 0.0, 1.0))

    return {"score": score, "attributes": results, "outcome_column": outcome_column}
