"""
Decoupled model theory: beta expansion with independent delta.
"""

from typing import Dict, Tuple

import numpy as np


def calculate_coefficients_beta_expansion(
    N: int,
    L: int,
    b: float,
    c: float,
    k0: float,
    k_cc_star: float,
    k_cd_star: float,
    k_dd_star: float
) -> Dict[str, float]:
    """
    Compute beta-expansion coefficients (delta-independent constants).
    
    rho(beta; delta) = 1/N + A(delta)*beta + B(delta)*beta^2
    where:
        A(delta) = A10 + A01*delta
        B(delta) = B1 + B01*delta + B02*delta^2
    """
    A10 = -((N - 1) * c * L) / (2.0 * N)
    
    struct_term = ((N + 1) / 3.0) * (k_cc_star - 2.0 * k_cd_star + k_dd_star)
    struct_term += N * (k_cd_star - k_dd_star)
    A01 = -((N - 1) / (2.0 * N**2 * k0)) * struct_term
    
    B1 = (c**2 * L**2 * (N - 2) * (N - 1)) / (12.0 * N)
    
    num_B01 = L * (N - 1) * (N + 1) * (
        4.0 * b * (k_cd_star - k_cc_star)
        + c * (
            4.0 * k_dd_star
            + 2.0 * k_cd_star * (N - 2)
            + k_cc_star * N
            - 3.0 * k_dd_star * N
        )
    )
    B01 = num_B01 / (24.0 * k0 * N**2)
    
    common = k_cc_star - 2.0 * k_cd_star + k_dd_star
    common += (k_cc_star + k_cd_star - 2.0 * k_dd_star) * N
    
    term_B02 = (
        2.0 * (N - 1) * common**2
        - 3.0 * N * (
            4.0 * k_cd_star**2
            - 2.0 * k_cd_star * k_dd_star
            - k_dd_star**2
            - 2.0 * k_cd_star**2 * N
            - 2.0 * k_cd_star * k_dd_star * N
            + 5.0 * k_dd_star**2 * N
        )
        - 3.0 * k_cc_star**2 * (1.0 + N)
        + 2.0 * k_cc_star * k_cd_star * (1.0 + N)
        + 2.0 * k0 * common
    )
    B02 = ((N - 1) / (72.0 * k0**2 * N**3)) * term_B02
    
    return {
        "A10": A10, "A01": A01,
        "B1": B1, "B01": B01, "B02": B02,
        "baseline": 1.0 / N,
    }


def theoretical_prediction(
    beta: float,
    delta: float,
    coeffs: Dict[str, float]
) -> Tuple[float, float]:
    """Compute first- and second-order predictions."""
    rho0 = coeffs["baseline"]
    A_delta = coeffs["A10"] + coeffs["A01"] * delta
    B_delta = coeffs["B1"] + coeffs["B01"] * delta + coeffs["B02"] * delta**2
    
    rho_1st = rho0 + A_delta * beta
    rho_2nd = rho0 + A_delta * beta + B_delta * beta**2
    
    return rho_1st, rho_2nd
