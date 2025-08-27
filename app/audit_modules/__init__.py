"""
Audit modules package exposing Privacy, Fairness, and Fidelity computations.
"""

from .privacy import run_membership_inference_attack
from .fairness import calculate_demographic_parity
from .fidelity import calculate_total_variation_distance

__all__ = [
    "run_membership_inference_attack",
    "calculate_demographic_parity",
    "calculate_total_variation_distance",
]
