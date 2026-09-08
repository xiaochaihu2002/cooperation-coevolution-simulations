"""Robustness of the critical intensity in the decoupled model (Fig. 6).

Three panels sweep the benefit b, the rewiring rationality delta and the
structural perturbation k_CC*, respectively, while all remaining parameters
are fixed.  Each curve is the second-order analytical prediction of the
decoupled model, psi_C(beta; delta), and crosses the neutral baseline 1/N
at a finite beta*.
"""

from pathlib import Path
from typing import Dict, List, Tuple

import numpy as np
import matplotlib.pyplot as plt

from src.plot_utils import setup_academic_style
from src.fig_beta_scan_decoupled.theory import (
    calculate_coefficients_beta_expansion,
    theoretical_prediction,
    critical_intensity,
)


BASE = {
    "N": 100, "L": 4, "b": 4.0, "c": 2.0, "k0": 0.5,
    "k_cc_star": -3.0, "k_cd_star": -2.0, "k_dd_star": -3.0,
    "delta": 0.005,
}
BETA_RANGE = np.logspace(-5.0, -2.0, 200)


def _curve(
    b: float = BASE["b"],
    delta: float = BASE["delta"],
    k_cc_star: float = BASE["k_cc_star"],
) -> Tuple[np.ndarray, np.ndarray, float]:
    """Return (beta, psi_C(beta)) and the crossing beta* for one case."""
    params = {
        "N": BASE["N"], "L": BASE["L"], "b": b, "c": BASE["c"],
        "k0": BASE["k0"], "k_cc_star": k_cc_star,
        "k_cd_star": BASE["k_cd_star"], "k_dd_star": BASE["k_dd_star"],
    }
    coeffs = calculate_coefficients_beta_expansion(**params)
    rho = np.array([
        theoretical_prediction(x, delta, coeffs)[1] for x in BETA_RANGE
    ])
    beta_star = critical_intensity(delta, coeffs)
    return BETA_RANGE, rho, beta_star


def _draw_panel(
    ax,
    title: str,
    curves: List[Dict[str, float]],
    base: float,
) -> None:
    """Draw one panel with a viridis palette and the 1/N baseline."""
    cmap = plt.get_cmap("viridis")
    n_curves = len(curves)
    for i, item in enumerate(curves):
        color = cmap(0.10 + 0.78 * i / max(n_curves - 1, 1))
        curve_kwargs = dict(item)
        label = curve_kwargs.pop("label")
        beta, rho, beta_star = _curve(**curve_kwargs)
        ax.plot(beta, rho, color=color, linestyle="-", linewidth=3.0,
                alpha=0.85, label=label, zorder=2)
        if np.isfinite(beta_star):
            ax.scatter([beta_star], [base], s=35, color=color, zorder=5,
                       edgecolors="white", linewidths=0.8)

    ax.axhline(base, color="gray", linestyle="--", linewidth=2,
               alpha=0.6, zorder=1, label=r"$1/N$")
    ax.set_xscale("log")
    ax.set_title(title, fontsize=16, pad=10)
    ax.set_xlabel(r"Selection intensity, $\beta$", fontsize=16)
    ax.legend(loc="lower left", frameon=False, fontsize=12,
              handlelength=2.0)


def main() -> None:
    """Generate the three-panel parameter robustness figure."""
    setup_academic_style()

    base = 1.0 / BASE["N"]
    curves_b = [
        {"b": b, "label": rf"$b = {b:.0f}$"}
        for b in (2.0, 4.0, 8.0, 10.0, 50.0, 100.0)
    ]
    curves_delta = [
        {"delta": d, "label": rf"$\delta = {d}$"}
        for d in (0.0005, 0.001, 0.002, 0.005, 0.01, 0.02)
    ]
    curves_kcc = [
        {"k_cc_star": k, "label": rf"$k_{{CC}}^* = {k}$"}
        for k in (-7.0, -5.0, -3.0, -1.0, 1.0, 3.0)
    ]

    fig, axes = plt.subplots(1, 3, figsize=(21, 6.5), dpi=300)
    _draw_panel(axes[0], r"Varying benefit $b$", curves_b, base)
    _draw_panel(axes[1], r"Varying $\delta$", curves_delta, base)
    _draw_panel(axes[2], r"Varying structural parameter $k_{CC}^*$",
                curves_kcc, base)
    axes[0].set_ylabel(r"Fixation probability, $\psi_C$", fontsize=18)

    plt.tight_layout()
    root = Path(__file__).resolve().parents[2]
    out = root / "figures"
    out.mkdir(parents=True, exist_ok=True)
    plt.savefig(out / "fig_parameter_robustness.svg", dpi=300,
                bbox_inches="tight")
    plt.savefig(out / "fig_parameter_robustness.jpg", dpi=300,
                bbox_inches="tight")
    plt.close(fig)


if __name__ == "__main__":
    main()
