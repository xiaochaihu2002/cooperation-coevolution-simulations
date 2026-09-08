"""Critical selection intensity against k_CD in the decoupled model.

The input quantities are the actual link-breaking probabilities
k_XY = k0 + k_XY* delta of the decoupled model, with delta = 0.005 held
fixed.  For k_DD = 0.650 and k_CC in {0.350, 0.390, 0.430}, the upward
crossing beta*(k_CD) of the second-order expansion is drawn over
k_CD in [0.650, 0.670] (Fig. 5 of the manuscript).
"""

from pathlib import Path

import numpy as np
import matplotlib.pyplot as plt
from matplotlib.ticker import MultipleLocator

from src.plot_utils import setup_academic_style
from src.fig_beta_scan_decoupled.theory import (
    calculate_coefficients_beta_expansion,
)


def compute_beta_star(
    k_cd: np.ndarray,
    k_cc: float,
    k_dd: float,
    n: int = 100,
    b: float = 4.0,
    c: float = 1.0,
    L: int = 4,
    k0: float = 0.5,
    delta: float = 0.005
) -> np.ndarray:
    """Return the upward crossing beta*(k_CD) of the decoupled expansion.

    The probabilities k_XY are converted to the perturbation slopes
    k_XY* = (k_XY - k0)/delta before the coefficients are evaluated.  NaN is
    returned wherever no positive crossing exists.
    """
    k_cd = np.asarray(k_cd, dtype=float)
    kcc_s = (k_cc - k0) / delta
    kcd_s = (k_cd - k0) / delta
    kdd_s = (k_dd - k0) / delta

    coeffs = calculate_coefficients_beta_expansion(
        n, L, b, c, k0, kcc_s, kcd_s, kdd_s,
    )
    const = coeffs["A01"] * delta + coeffs["B02"] * delta**2
    lin = coeffs["A10"] + coeffs["B01"] * delta
    disc = lin**2 - 4.0 * coeffs["B1"] * const

    out = np.full_like(k_cd, np.nan)
    ok = disc >= 0.0
    if not np.any(ok):
        return out

    r1 = (-lin[ok] - np.sqrt(disc[ok])) / (2.0 * coeffs["B1"])
    r2 = (-lin[ok] + np.sqrt(disc[ok])) / (2.0 * coeffs["B1"])
    upward = np.where(const[ok] < 0.0, np.minimum(r1, r2),
                      np.maximum(r1, r2))
    out[ok] = np.where(upward > 0.0, upward, np.nan)
    return out


def main() -> None:
    """Generate the decoupled k_CD scan figure."""
    setup_academic_style()

    k_cd = np.linspace(0.650, 0.670, 500)
    k_dd = 0.650
    param_sets = [
        {"k_cc": 0.350, "color": "#000080", "marker": "o",
         "label": r"$k_{CC} = 0.350$"},
        {"k_cc": 0.390, "color": "#006400", "marker": "s",
         "label": r"$k_{CC} = 0.390$"},
        {"k_cc": 0.430, "color": "#B22222", "marker": "^",
         "label": r"$k_{CC} = 0.430$"},
    ]

    fig, ax = plt.subplots(figsize=(8, 6), dpi=300)

    y_max_global = -np.inf
    y_min_global = np.inf
    for p in param_sets:
        beta_star = compute_beta_star(k_cd, p["k_cc"], k_dd,
                                      n=100, b=4.0, c=1.0, L=4,
                                      k0=0.5, delta=0.005)
        valid = ~np.isnan(beta_star)
        x_valid = k_cd[valid]
        y_valid = beta_star[valid]
        if not len(x_valid):
            continue

        y_max_global = max(y_max_global, float(y_valid.max()))
        y_min_global = min(y_min_global, float(y_valid.min()))

        ax.plot(x_valid, y_valid, color=p["color"], linewidth=2.5,
                alpha=0.9, zorder=3)
        ax.plot(x_valid, y_valid, color=p["color"], marker=p["marker"],
                markersize=7, markevery=15, linestyle="None",
                markerfacecolor="white", markeredgewidth=2,
                label=p["label"], zorder=4)

    margin = (y_max_global - y_min_global) * 0.15
    ax.set_xlim(0.650, 0.670)
    ax.set_ylim(y_min_global - margin, y_max_global + margin * 1.5)
    ax.xaxis.set_major_locator(MultipleLocator(0.0025))
    ax.yaxis.set_major_locator(MultipleLocator(0.0025))
    ax.xaxis.set_major_formatter("{x:.4f}")

    ax.set_xlabel(r"Breaking probability between cooperators and "
                  r"defectors, $k_{CD}$", fontsize=18, labelpad=8)
    ax.set_ylabel(r"The critical selection intensity, $\beta^*$",
                  fontsize=18, labelpad=8)
    ax.tick_params(axis="both", which="major", length=7, labelsize=16)
    ax.tick_params(axis="x", which="minor", length=4, direction="in",
                   top=True)
    ax.legend(loc="upper left", fontsize=16)

    plt.tight_layout()
    root = Path(__file__).resolve().parents[2]
    out = root / "figures"
    out.mkdir(parents=True, exist_ok=True)
    plt.savefig(out / "fig_decoupled_validation.svg", dpi=300,
                bbox_inches="tight")
    plt.savefig(out / "fig_decoupled_validation.jpg", dpi=300,
                bbox_inches="tight")
    plt.close(fig)


if __name__ == "__main__":
    main()
