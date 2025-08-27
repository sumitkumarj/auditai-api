"""
Privacy module: Membership Inference Approximation via Nearest Neighbor distances.

If real data is provided:
- Standardize numeric columns intersecting in real & synthetic.
- For each real row, compute distance to nearest synthetic row using sklearn NearestNeighbors.
- Compute the fraction of real rows with distance below an adaptive threshold (tau).
- Privacy score = 1 - fraction_below_tau (higher is better privacy).

If real data is not provided:
- Fall back to self-similarity in synthetic data (duplicate/near-duplicate rate).
- Privacy score = 1 - synthetic_selfmatch_rate.

Outputs include distance stats and thresholds for transparency.

Note: This is a practical privacy risk approximation rather than a full Shokri-style MIA.
"""

from typing import Any, Dict, Optional

import numpy as np
import pandas as pd
from sklearn.neighbors import NearestNeighbors
from sklearn.preprocessing import StandardScaler


def _common_numeric(real_df: pd.DataFrame, synth_df: pd.DataFrame) -> pd.DataFrame:
    common_cols = [c for c in synth_df.columns if c in real_df.columns]
    synth_num = synth_df[common_cols].select_dtypes(include=[np.number]).copy()
    real_num = real_df[common_cols].select_dtypes(include=[np.number]).copy()
    if synth_num.empty or real_num.empty:
        raise ValueError("No common numeric columns for privacy analysis.")
    # Align columns/order
    cols = sorted(set(synth_num.columns).intersection(set(real_num.columns)))
    if not cols:
        raise ValueError("No overlapping numeric columns for privacy analysis.")
    return real_num[cols], synth_num[cols]


def _adaptive_threshold(distances: np.ndarray) -> float:
    """
    Choose a robust threshold as Q1 - 1.5*IQR clipped to >= 0 and then add a small epsilon.
    This tries to flag unusually close matches as potential risk.
    """
    q1 = np.percentile(distances, 25)
    q3 = np.percentile(distances, 75)
    iqr = max(q3 - q1, 1e-6)
    tau = max(q1 - 1.5 * iqr, 0.0) + 1e-6
    return float(tau)


def run_membership_inference_attack(
    synthetic_df: pd.DataFrame,
    real_df: Optional[pd.DataFrame] = None,
) -> Dict[str, Any]:
    """
    Execute nearest-neighbor–based membership inference approximation.

    Args:
        synthetic_df: Synthetic dataset as DataFrame.
        real_df: Real dataset as DataFrame (optional but recommended).

    Returns:
        Dict with:
            - score (float in [0,1]): Higher is better privacy.
            - method (str)
            - details: thresholds and statistics.
    """
    if synthetic_df is None or synthetic_df.empty:
        raise ValueError("Synthetic dataset must be non-empty.")

    method_used = ""
    details: Dict[str, Any] = {}

    if real_df is not None and not real_df.empty:
        # Use cross-dataset nearest neighbor linkage risk
        real_num, synth_num = _common_numeric(real_df, synthetic_df)

        # Standardize jointly to ensure comparable scale
        scaler = StandardScaler(with_mean=True, with_std=True)
        combined = pd.concat([real_num, synth_num], axis=0)
        scaler.fit(combined.fillna(combined.mean()))
        real_scaled = scaler.transform(real_num.fillna(real_num.mean()))
        synth_scaled = scaler.transform(synth_num.fillna(synth_num.mean()))

        # Nearest neighbor from real -> synthetic
        nbrs = NearestNeighbors(n_neighbors=1, algorithm="auto", metric="euclidean")
        nbrs.fit(synth_scaled)
        dists, _ = nbrs.kneighbors(real_scaled, return_distance=True)
        dists = dists.reshape(-1)

        tau = _adaptive_threshold(dists)
        frac_below = float(np.mean(dists <= tau))

        # Privacy score: higher when fewer real records are unusually close to synthetic
        score = float(np.clip(1.0 - frac_below, 0.0, 1.0))
        method_used = "real_to_synth_nearest_neighbor"
        details.update(
            {
                "nearest_neighbor": {
                    "count": int(len(dists)),
                    "min": float(np.min(dists)),
                    "q1": float(np.percentile(dists, 25)),
                    "median": float(np.median(dists)),
                    "q3": float(np.percentile(dists, 75)),
                    "max": float(np.max(dists)),
                    "mean": float(np.mean(dists)),
                    "std": float(np.std(dists)),
                    "threshold_tau": float(tau),
                    "fraction_below_tau": float(frac_below),
                },
                "columns_used": list(real_num.columns),
            }
        )
    else:
        # Fallback: self-similarity within synthetic data
        synth_num = synthetic_df.select_dtypes(include=[np.number]).copy()
        if synth_num.empty:
            # If no numeric columns, treat as unknown risk -> conservative mid score
            return {
                "score": 0.5,
                "method": "synthetic_self_similarity_no_numeric",
                "details": {"note": "No numeric columns; defaulting to 0.5 privacy score."},
            }

        scaler = StandardScaler(with_mean=True, with_std=True)
        synth_scaled = scaler.fit_transform(synth_num.fillna(synth_num.mean()))

        nbrs = NearestNeighbors(n_neighbors=2, algorithm="auto", metric="euclidean")
        nbrs.fit(synth_scaled)
        # Distance to the nearest *other* point: take the 2nd neighbor
        dists, idx = nbrs.kneighbors(synth_scaled, return_distance=True)
        nearest_other = dists[:, 1]  # exclude self (0th)
        tau = _adaptive_threshold(nearest_other)
        frac_below = float(np.mean(nearest_other <= tau))
        score = float(np.clip(1.0 - frac_below, 0.0, 1.0))
        method_used = "synthetic_self_similarity"
        details.update(
            {
                "self_nearest_neighbor": {
                    "count": int(len(nearest_other)),
                    "min": float(np.min(nearest_other)),
                    "q1": float(np.percentile(nearest_other, 25)),
                    "median": float(np.median(nearest_other)),
                    "q3": float(np.percentile(nearest_other, 75)),
                    "max": float(np.max(nearest_other)),
                    "mean": float(np.mean(nearest_other)),
                    "std": float(np.std(nearest_other)),
                    "threshold_tau": float(tau),
                    "fraction_below_tau": float(frac_below),
                },
                "columns_used": list(synth_num.columns),
            }
        )

    return {"score": score, "method": method_used, **details}
