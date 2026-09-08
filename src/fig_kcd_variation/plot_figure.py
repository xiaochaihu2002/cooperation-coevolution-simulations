"""Critical selection intensity against the CD breaking probability.

The second-order prediction beta*(k_CD*) = -A(k_CD*)/B(k_CD*) is evaluated
with the coefficients of Eq. (Psi).  The curves reproduce Fig. 4 of the
manuscript: N = 100, b = 4, c = 1, L = 4, k0 = 0.5, a = 0.5, with
k_DD* = 1 fixed and k_CC* = -10, -8, -7.  The shaded intervals mark the
domain on which beta* is positive and strictly increasing in k_CD*.
"""

from pathlib import Path

import numpy as np
import matplotlib.pyplot as plt
from matplotlib.ticker import MultipleLocator

from src.theory import compute_coefficients
from src.plot_utils import setup_academic_style


def compute_beta_star(
    k_cd_star: np.ndarray,
    k_cc_star: float,
    k_dd_star: float,
    n: int = 100,
    b: float = 4.0,
    c: float = 1.0,
    a: float = 0.5,
    L: int = 4,
    k0: float = 0.5
) -> np.ndarray:
    """Return beta*(k_CD*) = -A/B of the second-order expansion.

    The result is undefined where B = 0 (pole of the approximation) and is
    returned as NaN there.
    """
    A, B = compute_coefficients(
        n, L, b, c, k0, a, k_cc_star,
        np.asarray(k_cd_star, dtype=float), k_dd_star,
    )
    with np.errstate(divide="ignore", invalid="ignore"):
        beta2 = np.where(B != 0.0, -A / B, np.nan)
    return np.where(np.isfinite(beta2), beta2, np.nan)


def main() -> None:
    """Generate the beta*(k_CD*) figure."""
    setup_academic_style()

    k_cd_star = np.linspace(-10.0, 10.0, 2000)
    param_sets = [
        {"k_cc_star": -10.0, "color": "#B22222", "linestyle": "-"},
        {"k_cc_star": -8.0, "color": "#000080", "linestyle": "--"},
        {"k_cc_star": -7.0, "color": "#006400", "linestyle": "-."},
    ]

    fig, ax = plt.subplots(figsize=(8, 6), dpi=300)
    ax.axhline(0.0, color="black", linestyle="-", linewidth=2, zorder=2)

    y_bottom = -0.001
    y_top = 0.0
    for p in param_sets:
        beta_star = compute_beta_star(
            k_cd_star, p["k_cc_star"], k_dd_star=1.0,
            n=100, b=4.0, c=1.0, a=0.5, L=4, k0=0.5,
        )
        finite = beta_star[np.isfinite(beta_star)]
        if finite.size:
            y_top = max(y_top, float(finite.max()))

        disp = np.clip(beta_star, y_bottom, None)
        ax.plot(k_cd_star, disp, color=p["color"], linewidth=3,
                linestyle=p["linestyle"],
                label=rf'$k_{{CC}}^* = {p["k_cc_star"]:.0f}$', zorder=4)

        A, _ = compute_coefficients(
            100, 4, 4.0, 1.0, 0.5, 0.5, p["k_cc_star"],
            k_cd_star, 1.0,
        )
        gradient = np.gradient(beta_star)
        ascending = (beta_star > 0.0) & (A < 0.0) & (gradient > 0.0)
        ax.fill_between(k_cd_star, disp, 0.0, where=ascending,
                        color=p["color"], alpha=0.3, edgecolor="none",
                        zorder=3)

    ax.fill_between(k_cd_star, y_bottom, 0.0, color="gray", alpha=0.1,
                    hatch="//////", edgecolor="gray", zorder=1)

    ax.set_xlabel(r"Breaking probability between cooperators and "
                  r"defectors, $k_{CD}^*$", fontsize=18, labelpad=8)
    ax.set_ylabel(r"The critical selection intensity, $\beta^*$",
                  fontsize=18, labelpad=8)
    ax.set_xlim(-10.0, 10.0)
    ax.set_ylim(y_bottom, y_top * 1.05)
    ax.xaxis.set_major_locator(MultipleLocator(2.5))
    ax.yaxis.set_major_locator(MultipleLocator(0.002))
    ax.tick_params(axis="both", which="major", length=7, labelsize=18)
    ax.tick_params(axis="x", which="minor", length=4, direction="in",
                   top=True)
    ax.legend(loc="upper left", fontsize=16)

    plt.tight_layout()
    root = Path(__file__).resolve().parents[2]
    out = root / "figures"
    out.mkdir(parents=True, exist_ok=True)
    plt.savefig(out / "fig_kcd_variation.svg", dpi=300,
                bbox_inches="tight")
    plt.savefig(out / "fig_kcd_variation.jpg", dpi=300,
                bbox_inches="tight")
    plt.close(fig)


if __name__ == "__main__":
    main()
