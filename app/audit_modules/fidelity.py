"""
Fidelity module: Total Variation Distance (TVD) across numeric marginals.

For each numeric column present in both real and synthetic datasets:
- Bin using robust quantile-based edges (to handle skew).
- Compute discrete distributions and TVD = 0.5 * sum |p - q|.
- Final score = 1 - average_TVD (higher is better fidelity).

Outputs include per-column TVD and bin statistics.
"""

from typing import Any, Dict, List, Tuple

import numpy as np
import pandas as pd


def _quantile_bins(x: pd.Series, n_bins: int = 10) -> np.ndarray:
    """Return sorted unique quantile bin edges including min/max."""
    quantiles = np.linspace(0.0, 1.0, n_bins + 1)
    edges = np.unique(np.nanquantile(x.values, quantiles))
    # Ensure at least two edges
    if len(edges) < 2:
        edges = np.array([np.nanmin(x.values), np.nanmax(x.values)])
    # Degenerate case handling
    if edges[0] == edges[-1]:
        edges[-1] = edges[0] + 1e-9
    return edges


def _hist_probs(x: pd.Series, bins: np.ndarray) -> np.ndarray:
    """Return normalized histogram probabilities for given bins."""
    counts, _ = np.histogram(x.dropna().values, bins=bins)
    total = counts.sum()
    if total == 0:
        # Avoid division by zero; return uniform-ish small distribution
        return np.ones_like(counts, dtype=float) / len(counts)
    return counts.astype(float) / float(total)


def _tvd(p: np.ndarray, q: np.ndarray) -> float:
    return 0.5 * float(np.abs(p - q).sum())


def calculate_total_variation_distance(
    real_df: pd.DataFrame,
    synth_df: pd.DataFrame,
    n_bins: int = 10,
) -> Dict[str, Any]:
    """
    Compute TVD across common numeric columns and produce a fidelity score.

    Returns:
        {
          "score": float,
          "columns": {
            col: {
              "tvd": float,
              "bins": List[List[float]],   # pairs of bin edges
            }, ...
          },
          "columns_used": [ ... ]
        }
    """
    if real_df is None or real_df.empty or synth_df is None or synth_df.empty:
        raise ValueError("Both real and synthetic dataframes must be non-empty.")

    real_num = real_df.select_dtypes(include=[np.number])
    synth_num = synth_df.select_dtypes(include=[np.number])
    common_cols = sorted(set(real_num.columns).intersection(set(synth_num.columns)))
    if not common_cols:
        raise ValueError("No overlapping numeric columns for fidelity analysis.")

    col_results: Dict[str, Any] = {}
    tvds: List[float] = []

    for col in common_cols:
        r = real_num[col].dropna()
        s = synth_num[col].dropna()
        if r.empty or s.empty:
            continue
        bins = _quantile_bins(pd.concat([r, s], axis=0), n_bins=n_bins)
        p = _hist_probs(r, bins)
        q = _hist_probs(s, bins)
        tv = _tvd(p, q)
        tvds.append(tv)

        # Convert bin edges into intervals for readability
        bins_intervals = [[float(bins[i]), float(bins[i + 1])] for i in range(len(bins) - 1)]
        col_results[col] = {"tvd": float(tv), "bins": bins_intervals}

    avg_tvd = float(np.mean(tvds)) if tvds else 1.0  # worst-case if no usable columns
    score = float(np.clip(1.0 - avg_tvd, 0.0, 1.0))

    return {"score": score, "columns": col_results, "columns_used": common_cols}
