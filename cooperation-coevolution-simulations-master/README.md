# Cooperation on Adaptive Networks: Critical Selection Intensity

Simulation and visualization code for the critical selection intensity
$\beta^*$ in evolutionary game dynamics on adaptive networks.

## Installation

```bash
pip install -r requirements.txt
```

## Overview

In the fast-rewiring limit the fixation probability of a single cooperator is
approximated by the second-order expansion

$$\psi_C(\beta) = \frac{1}{N} + A\beta + B\beta^2 + o(\beta^2),$$

where the coefficients $A$ and $B$ follow from the binary Taylor expansion of
the fixation probability in the coupled model with rewiring rationality
$\delta = a\beta$.  The critical selection intensity $\beta^* = -A/B$ is the
threshold at which the fixation probability crosses the neutral baseline
$1/N$: cooperation cannot be favored below $\beta^*$ and can be favored above
it.

The repository contains the shared theory module, Monte Carlo kernels,
per-figure packages, and verification scripts.

## Project layout

```text
src/
    theory.py                  coupled second-order coefficients (Eq. Psi)
    simulation_core.py         shared network and Monte Carlo primitives
    fig_validation/            validation against Monte Carlo simulations
    fig_kcd_variation/         beta*(k_CD*) in the coupled model
    fig_beta_scan_decoupled/   beta scan with decoupled rationalities
    fig_decoupled_validation/  beta*(k_CD) with decoupled rationalities
    fig_parameter_robustness/  parameter sweeps of the decoupled model
scripts/
    verify_correction.py       theory and figure checks
    residual_analysis.py       residual analysis of the expansion
```

## Usage

Run the Monte Carlo simulations and generate the figures from the project
root:

```bash
# validation of the second-order expansion (Fig. 2)
python -m src.fig_validation.run_simulation
python -m src.fig_validation.plot_figure

# critical selection intensity vs. k_CD* (Fig. 4)
python -m src.fig_kcd_variation.plot_figure

# decoupled rationalities: beta scan (Fig. 3)
python -m src.fig_beta_scan_decoupled.run_simulation
python -m src.fig_beta_scan_decoupled.plot_figure

# decoupled rationalities: k_CD scan (Fig. 5)
python -m src.fig_decoupled_validation.plot_figure

# parameter robustness of the decoupled model (Fig. 6)
python -m src.fig_parameter_robustness.plot_figure
```

Simulation tables are written to `data/` and figures to `figures/`.

Verify the theoretical coefficients and the figure predictions:

```bash
python scripts/verify_correction.py
```

The Monte Carlo results in the manuscript were obtained with $10^6$
independent runs per data point; the default `mc_trials` setting reproduces
that resolution.  All analytical figures are generated directly from the
second-order expansion and do not require the simulations.

If the original dataset is needed, please contact "3273417757@qq.com" or
"lyx2025@bupt.edu.cn".
