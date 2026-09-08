"""Second-order expansion of the decoupled-rationality model.

When the rewiring rationality delta is independent of the strategy-updating
intensity beta, the bivariate expansion of the fixation probability around
(beta, delta) = (0, 0) reads

    psi_C(beta; delta) = 1/N + A01*delta + B02*delta^2
                             + (A10 + B01*delta)*beta + B1*beta^2,

with

    A10 = d psi/d beta,
    A01 = d psi/d delta,
    B1  = (1/2) d^2 psi/d beta^2,
    B01 = d^2 psi/d beta d delta,
    B02 = (1/2) d^2 psi/d delta^2,

all derivatives being evaluated at (0, 0) and expressed through the
perturbation slopes k_XY* of the link-breaking probabilities
k_XY(delta) = k0 + k_XY* delta.
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
    """Return the coefficients of the decoupled beta expansion.

    Args:
        N: Population size.
        L: Average degree.
        b: Benefit of cooperation.
        c: Cost of cooperation.
        k0: Baseline link-breaking probability.
        k_cc_star, k_cd_star, k_dd_star: Perturbation slopes.

    Returns:
        Dict with keys A10, A01, B1, B01, B02 and baseline = 1/N.
    """
    X = k_cc_star - 2.0 * k_cd_star + k_dd_star
    Y = k_cd_star - k_dd_star

    # First-order derivatives.
    A10 = -((N - 1) * c * L) / (2.0 * N)
    A01 = -((N - 1) / (2.0 * N**2 * k0)) * (
        (N + 1) / 3.0 * X + N * Y
    )

    # Second-order derivatives.
    B1 = (c**2 * L**2 * (N - 2) * (N - 1)) / (12.0 * N)

    B01 = (
        L * (N - 1) * (N + 1) / (24.0 * k0 * N**2)
        * (
            c * (N * X + 4.0 * (N - 1) * Y)
            - 4.0 * b * (X + Y)
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
    B02 = ((N - 1) * phi) / (360.0 * N**3 * k0**2)

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
    """Evaluate the first- and second-order predictions at (beta, delta)."""
    rho0 = coeffs["baseline"]
    const = coeffs["A01"] * delta + coeffs["B02"] * delta**2
    lin = coeffs["A10"] + coeffs["B01"] * delta
    quad = coeffs["B1"]

    rho_1st = rho0 + const + lin * beta
    rho_2nd = rho0 + const + lin * beta + quad * beta**2
    return rho_1st, rho_2nd


def critical_intensity(delta: float, coeffs: Dict[str, float]) -> float:
    """Return the positive crossing of psi_C(beta) with 1/N, or NaN.

    The crossing solves
        const + lin*beta + B1*beta^2 = 0,
    with const = A01*delta + B02*delta^2 and lin = A10 + B01*delta.
    """
    const = coeffs["A01"] * delta + coeffs["B02"] * delta**2
    lin = coeffs["A10"] + coeffs["B01"] * delta
    quad = coeffs["B1"]

    disc = lin**2 - 4.0 * quad * const
    if disc < 0.0 or quad == 0.0:
        return np.nan
    roots = np.array([
        (-lin - np.sqrt(disc)) / (2.0 * quad),
        (-lin + np.sqrt(disc)) / (2.0 * quad),
    ])
    positive = roots[roots > 0.0]
    return float(positive.max()) if positive.size else np.nan
