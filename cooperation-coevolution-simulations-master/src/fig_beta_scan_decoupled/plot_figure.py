"""Plot the decoupled beta scan (Fig. 3).

The second-order analytical curve psi_C(beta; delta) is overlaid on the
Monte Carlo estimates.  The crossing with the neutral baseline 1/N is
marked at the analytical critical intensity.
"""

from pathlib import Path

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from src.plot_utils import setup_academic_style
from .config import DecoupledConfig
from .theory import (
    calculate_coefficients_beta_expansion,
    theoretical_prediction,
    critical_intensity,
)


def main() -> None:
    """Generate the decoupled beta scan figure."""
    root = Path(__file__).resolve().parents[2]
    csv_path = root / "data" / "fig_beta_scan_decoupled.csv"
    if csv_path.exists():
        df = pd.read_csv(csv_path)
        beta_min = float(df["beta"].min())
        beta_max = float(df["beta"].max())
        delta = float(df["delta"].iloc[0])
    else:
        cfg = DecoupledConfig()
        beta_min, beta_max = float(np.min(cfg.beta_range)), float(
            np.max(cfg.beta_range))
        delta = cfg.delta
        df = None

    params = dict(N=100, L=4, b=4.0, c=1.0, k0=0.5,
                  k_cc_star=-3.0, k_cd_star=-2.0, k_dd_star=-10.0)
    coeffs = calculate_coefficients_beta_expansion(**params)

    beta_dense = np.logspace(np.log10(beta_min), np.log10(beta_max), 2000)
    rho_1st = np.array([
        theoretical_prediction(b, delta, coeffs)[0] for b in beta_dense
    ])
    rho_2nd = np.array([
        theoretical_prediction(b, delta, coeffs)[1] for b in beta_dense
    ])
    base = 1.0 / params["N"]
    beta_star = critical_intensity(delta, coeffs)

    setup_academic_style()
    fig, ax = plt.subplots(figsize=(8, 6), dpi=300)
    ax.set_xscale("log")

    ax.plot(beta_dense, rho_2nd, color="#B22222", linewidth=3,
            label=r"Theoretical ($2^{\mathrm{nd}}$ Order)", zorder=2)

    if df is not None:
        ax.errorbar(df["beta"], df["sim_prob"], yerr=df["sim_error"],
                    fmt="o", color="black", ecolor="black", elinewidth=2,
                    capsize=4, markersize=7, markerfacecolor="white",
                    markeredgewidth=2, label="Simulation Results", zorder=3)

    ax.axhline(base, color="gray", linestyle=":", linewidth=2,
               label=rf"Neutral ($1/N = {base:.4f}$)", zorder=1)

    if np.isfinite(beta_star):
        ax.scatter([beta_star], [base], s=70, color="#008B45", zorder=5,
                   edgecolors="white", linewidths=1.2)
        ax.annotate(
            rf"$\beta^* \approx {beta_star:.2e}$",
            xy=(beta_star, base), xytext=(0.36, 0.18),
            textcoords="axes fraction", fontsize=15, color="#008B45",
            arrowprops=dict(arrowstyle="->", color="#008B45", lw=1.5),
        )

    ax.set_xlabel(r"Selection Intensity, $\beta$", fontsize=18, labelpad=8)
    ax.set_ylabel(r"Fixation Probability, $\psi_C$", fontsize=18,
                  labelpad=8)
    ax.tick_params(axis="both", which="major", length=7, labelsize=16)
    ax.tick_params(axis="x", which="minor", length=4, direction="in",
                   top=True)

    y_vals = np.concatenate([
        rho_1st, rho_2nd,
        df[["sim_prob"]].to_numpy().ravel()
        if df is not None else np.array([]),
    ])
    y_min, y_max = np.min(y_vals), np.max(y_vals)
    pad = 0.05 * (y_max - y_min)
    ax.set_ylim(max(0.0, y_min - pad), y_max + pad)

    ax.legend(loc="upper left", fontsize=15)
    plt.tight_layout()

    out = root / "figures"
    out.mkdir(parents=True, exist_ok=True)
    plt.savefig(out / "fig_beta_scan_decoupled.svg", dpi=300,
                bbox_inches="tight")
    plt.savefig(out / "fig_beta_scan_decoupled.jpg", dpi=300,
                bbox_inches="tight")
    plt.close(fig)


if __name__ == "__main__":
    main()
