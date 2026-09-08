"""Configuration for the validation of the second-order expansion."""

from dataclasses import dataclass, field
from typing import Tuple

import numpy as np

from src.config import BaseConfig


@dataclass
class ValidationConfig(BaseConfig):
    """Monte Carlo validation of psi_C(beta) = 1/N + A beta + B beta^2.

    Two cases with k_CD* = 6 and k_CD* = -2 are simulated at fixed
    N = 100, b = 4, c = 1, L = 4, k0 = 0.5, a = 0.5, k_CC* = -3 and
    k_DD* = -10 (Fig. 2 of the manuscript).
    """

    a: float = 0.5

    N: int = 100
    b: float = 4.0
    c: float = 1.0
    L: int = 4
    k0: float = 0.5
    k_cc_star: float = -3.0
    k_dd_star: float = -10.0
    k_cd_star_values: Tuple[float, float] = (6.0, -2.0)

    beta_range: np.ndarray = field(
        default_factory=lambda: np.logspace(-4, -2, 10)
    )
    mc_trials: int = 1_000_000
    max_steps: int = 1_000_000_000
    n_jobs: int = -1

    filename: str = "fig_validation.csv"
