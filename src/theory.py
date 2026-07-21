"""
Shared theoretical calculations: full second-order expansion.
"""

from typing import Tuple, Dict

import numpy as np


def compute_coefficients(
    N: int,
    L: int,
    b: float,
    c: float,
    k0: float,
    a: float,
    k_cc_star: float,
    k_cd_star: float,
    k_dd_star: float
) -> Tuple[float, float]:
    """
    Compute first-order (A) and second-order (B) coefficients.
    
    Fixation probability expansion:
        psi = 1/N + A * beta + B * beta^2
    
    Args:
        N, L, b, c, k0, a: Physical parameters.
        k_cc_star, k_cd_star, k_dd_star: Link breaking perturbations.
    
    Returns:
        Tuple (A, B).
    """
    # Linear coefficient A
    term1 = -((N - 1) * c * L) / (2.0 * N)
    struct_term = ((N + 1) / 3.0) * (k_cc_star - 2.0 * k_cd_star + k_dd_star)
    struct_term += N * (k_cd_star - k_dd_star)
    term2 = -((N - 1) / (2.0 * N**2 * k0)) * struct_term * a
    A = term1 + term2

    # Quadratic coefficient B
    B1 = (c**2 * L**2 * (N - 2) * (N - 1)) / (12.0 * N)
    
    num_B2 = L * (N - 1) * (N + 1) * (
        4.0 * b * (k_cd_star - k_cc_star)
        + c * (
            4.0 * k_dd_star
            + 2.0 * k_cd_star * (N - 2)
            + k_cc_star * N
            - 3.0 * k_dd_star * N
        )
    )
    B2 = (num_B2 / (24.0 * k0 * N**2)) * a
    
    common = k_cc_star - 2.0 * k_cd_star + k_dd_star
    common += (k_cc_star + k_cd_star - 2.0 * k_dd_star) * N
    term_in_B3 = (
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
    B3 = ((N - 1) / (72.0 * k0**2 * N**3)) * term_in_B3 * (a**2)

    B = B1 + B2 + B3
    return A, B


def generate_smooth_theory(
    beta_min: float,
    beta_max: float,
    n_points: int = 2000,
    **coeff_kwargs
) -> Tuple[np.ndarray, np.ndarray, np.ndarray, float, float, float]:
    """
    Generate dense theoretical curves on log-spaced beta grid.
    
    Returns:
        beta_dense, psi_1st, psi_2nd, A, B, beta_cross.
    """
    beta_dense = np.logspace(
        np.log10(beta_min), np.log10(beta_max), n_points
    )
    A, B = compute_coefficients(**coeff_kwargs)
    
    N = coeff_kwargs['N']
    psi_1st = 1.0 / N + A * beta_dense
    psi_2nd = 1.0 / N + A * beta_dense + B * beta_dense**2
    
    beta_cross = -A / B if B != 0 else np.nan
    
    return beta_dense, psi_1st, psi_2nd, A, B, beta_cross
