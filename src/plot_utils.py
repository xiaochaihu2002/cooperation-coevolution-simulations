"""
Shared visualization utilities.
"""

import matplotlib.pyplot as plt


def setup_academic_style() -> None:
    """Configure matplotlib for publication-quality figures."""
    plt.style.use('default')
    plt.rcParams.update({
        'font.family': 'serif',
        'font.serif': ['Times New Roman'],
        'mathtext.fontset': 'stix',
        'font.size': 16,
        'axes.linewidth': 3,
        'xtick.major.width': 2,
        'ytick.major.width': 2,
        'xtick.minor.width': 1.5,
        'xtick.direction': 'in',
        'ytick.direction': 'in',
        'xtick.top': True,
        'ytick.right': True,
        'legend.frameon': False,
        'axes.unicode_minus': False,
    })
