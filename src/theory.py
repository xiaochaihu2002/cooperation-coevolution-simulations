"""
Shared theoretical calculations for the second-order expansion of the
fixation probability in the coupled fast-rewiring model.

Along the coupled path delta = a*beta the fixation probability of a single
cooperator reads

    psi_C(beta) = 1/N + A*beta + B*beta^2 + o(beta^2),

where A and B are the coefficients collected in Eq. (Psi) of the manuscript:

    A = -(N-1)cL/(2N)
        - (N-1)a/(2N^2 k0) * [(N+1)/3 (k_CC* - 2k_CD* + k_DD*)
                              + N (k_CD* - k_DD*)],

    B = c^2 L^2 (N-2)(N-1)/(12N)
        + a L (N-1)(N+1)/(24 k0 N^2)
          * [4b(k_CD* - k_CC*)
             + c(4k_DD* + 2k_CD*(N-2) + k_CC*N - 3k_DD*N)]
        + a^2 (N-1)/(360 k0^2 N^3) * Phi(k_CC*, k_CD*, k_DD*),

with the quadratic form Phi given in Eq. (Psi).
"""

from typing import Tuple

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
    """Return the first-order (A) and second-order (B) coefficients.

    Args:
        N: Population size.
        L: Average degree of the interaction network.
        b: Benefit produced by a cooperating neighbour.
        c: Cost paid by a cooperator.
        k0: Baseline link-breaking probability.
        a: Coupling constant delta = a*beta.
        k_cc_star, k_cd_star, k_dd_star: Perturbation slopes of the
            CC, CD and DD link-breaking probabilities.

    Returns:
        Tuple (A, B) such that psi_C = 1/N + A*beta + B*beta^2.
    """
    # Linear coefficient A.
    A = -((N - 1) * c * L) / (2.0 * N)
    struct_term = ((N + 1) / 3.0) * (
        k_cc_star - 2.0 * k_cd_star + k_dd_star
    )
    struct_term += N * (k_cd_star - k_dd_star)
    A += -((N - 1) * a / (2.0 * N**2 * k0)) * struct_term

    # Quadratic coefficient B = B1 + B2 + B3.
    B1 = (c**2 * L**2 * (N - 2) * (N - 1)) / (12.0 * N)

    B2 = (
        a * L * (N - 1) * (N + 1) / (24.0 * k0 * N**2)
        * (
            4.0 * b * (k_cd_star - k_cc_star)
            + c * (
                4.0 * k_dd_star
                + 2.0 * k_cd_star * (N - 2)
                + k_cc_star * N
                - 3.0 * k_dd_star * N
            )
        )
    )

    phi = (
        (N**3 + 46.0 * N**2 + 41.0 * N - 4.0) * k_cc_star**2
        + (11.0 * N**3 - 49.0 * N**2 - 44.0 * N + 16.0)
        * k_cc_star * k_cd_star
        + (-13.0 * N**3 + 17.0 * N**2 + 22.0 * N - 8.0)
        * k_cc_star * k_dd_star
        + (4.0 * N**3 + 4.0 * N**2 - 16.0 * N - 16.0) * k_cd_star**2
        + (-19.0 * N**3 + 101.0 * N**2 - 44.0 * N + 16.0)
        * k_cd_star * k_dd_star
        + (16.0 * N**3 - 119.0 * N**2 + 41.0 * N - 4.0) * k_dd_star**2
    )
    B3 = (a**2 * (N - 1) * phi) / (360.0 * N**3 * k0**2)

    return A, B1 + B2 + B3


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
