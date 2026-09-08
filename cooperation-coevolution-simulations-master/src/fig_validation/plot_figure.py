"""Plot the validation of the second-order expansion (Fig. 2).

Two stacked panels show the fixation probability against the selection
intensity for k_CD* = 6 and k_CD* = -2.  Each panel contains the first- and
second-order theoretical curves, the neutral baseline 1/N and markers at the
analytical critical intensity.  When the Monte Carlo table produced by
src.fig_validation.run_simulation is present, simulation estimates with
error bars and the simulated critical intensity are overlaid as well.
"""

from pathlib import Path
from typing import Optional

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from src.theory import compute_coefficients
from src.plot_utils import setup_academic_style


PARAMETERS = dict(N=100, L=4, b=4.0, c=1.0, k0=0.5, a=0.5,
                  k_cc_star=-3.0, k_dd_star=-10.0)
BETA_GRID = np.logspace(-4.0, -2.0, 10)


def _simulated_crossing(beta: np.ndarray, prob: np.ndarray) -> float:
    """Return the first beta where the simulated probability crosses 1/N."""
    base = 1.0 / PARAMETERS["N"]
    diff = prob - base
    for i in range(len(beta) - 1):
        if diff[i] < 0.0 <= diff[i + 1]:
            frac = -diff[i] / (diff[i + 1] - diff[i])
            return float(beta[i] + frac * (beta[i + 1] - beta[i]))
    return np.nan


def _panel(ax, k_cd_star: float, letter: str,
           sim: Optional[pd.DataFrame]) -> None:
    """Draw one panel of the validation figure."""
    params = dict(PARAMETERS, k_cd_star=k_cd_star)
    A, B = compute_coefficients(**params)
    base = 1.0 / params["N"]

    if sim is not None:
        beta = np.logspace(np.log10(sim["beta"].min()),
                           np.log10(sim["beta"].max()), 2000)
    else:
        beta = BETA_GRID
    dense = np.logspace(np.log10(beta.min()), np.log10(beta.max()), 2000)
    psi_1st = base + A * dense
    psi_2nd = base + A * dense + B * dense**2

    ax.set_xscale("log")
    ax.plot(dense, psi_2nd, color="#B22222", linewidth=3,
            label=r"Theoretical ($2^{\mathrm{nd}}$ Order)", zorder=2)
    ax.plot(dense, psi_1st, color="#000080", linewidth=3,
            linestyle="--",
            label=r"Theoretical ($1^{\mathrm{st}}$ Order)", zorder=2)
    ax.axhline(base, color="gray", linestyle=":", linewidth=2,
               label=rf"Neutral ($1/N = {base:.4f}$)", zorder=1)

    if sim is not None:
        ax.errorbar(sim["beta"], sim["sim_prob"], yerr=sim["sim_error"],
                    fmt="o", color="black", ecolor="black", elinewidth=2,
                    capsize=4, markersize=7, markerfacecolor="white",
                    markeredgewidth=2, label="Simulation", zorder=3)

        beta_sim = _simulated_crossing(
            sim["beta"].to_numpy(), sim["sim_prob"].to_numpy())
        if np.isfinite(beta_sim):
            ax.scatter([beta_sim], [base], s=65, color="#E8860B",
                       zorder=5, edgecolors="white", linewidths=1.2)
            ax.annotate(
                rf"$\beta^{{\prime *}} \approx {beta_sim:.2g}$",
                xy=(beta_sim, base), xytext=(0.62, 0.22),
                textcoords="axes fraction", fontsize=15, color="#E8860B",
                arrowprops=dict(arrowstyle="->", color="#E8860B", lw=1.5),
            )

    beta_star = -A / B
    if beta_star > 0.0:
        ax.scatter([beta_star], [base], s=65, color="#008B45", zorder=5,
                   edgecolors="white", linewidths=1.2)
        ax.annotate(
            rf"$\beta^* \approx {beta_star:.2g}$",
            xy=(beta_star, base), xytext=(0.30, 0.68),
            textcoords="axes fraction", fontsize=15, color="#008B45",
            arrowprops=dict(arrowstyle="->", color="#008B45", lw=1.5),
        )

    ax.text(0.97, 0.92, rf"$k_{{CD}}^* = {k_cd_star:.0f}$",
            transform=ax.transAxes, ha="right", va="top", fontsize=17)
    ax.text(0.02, 0.08, f"({letter})", transform=ax.transAxes,
            ha="left", va="bottom", fontsize=17)

    y_max = max(psi_2nd.max(), psi_1st.max(), base)
    y_min = min(psi_1st.min(), psi_2nd.min(), base)
    if sim is not None:
        y_max = max(y_max, float(sim["sim_prob"].max()))
        y_min = min(y_min, float(sim["sim_prob"].min()))
    pad = 0.08 * (y_max - y_min)
    ax.set_ylim(y_min - pad, y_max + pad)


def main() -> None:
    """Generate the validation figure."""
    root = Path(__file__).resolve().parents[2]
    csv_path = root / "data" / "fig_validation.csv"
    df = None
    if csv_path.exists():
        df = pd.read_csv(csv_path)
    else:
        print("Simulation table not found; plotting the analytical curves "
              "only. Run python -m src.fig_validation.run_simulation first "
              "to include the Monte Carlo markers.")

    setup_academic_style()
    fig, axes = plt.subplots(2, 1, figsize=(8, 10), dpi=300, sharex=True)

    for ax, (k_cd_star, letter) in zip(axes, [(6.0, "a"), (-2.0, "b")]):
        sim = None
        if df is not None:
            case = df[df["k_cd_star"] == k_cd_star]
            if len(case):
                sim = case
        _panel(ax, k_cd_star, letter, sim)

    axes[-1].set_xlabel(r"Selection Intensity, $\beta$",
                        fontsize=18, labelpad=8)
    axes[0].set_ylabel(r"Fixation Probability, $\psi_C$",
                       fontsize=18, labelpad=8)
    axes[0].legend(loc="upper left", fontsize=15, ncol=2)

    plt.tight_layout()
    out = root / "figures"
    out.mkdir(parents=True, exist_ok=True)
    plt.savefig(out / "fig_validation.svg", dpi=300, bbox_inches="tight")
    plt.savefig(out / "fig_validation.jpg", dpi=300, bbox_inches="tight")
    plt.close(fig)


if __name__ == "__main__":
    main()
