"""
Shared Monte Carlo simulation primitives.
"""

import random
from typing import Set, List

import numpy as np
import networkx as nx


def initialize_network(N: int, L: int) -> nx.Graph:
    """Create a connected L-regular random graph."""
    while True:
        G = nx.random_regular_graph(L, N)
        if nx.is_connected(G):
            return G


def get_payoff(
    idx: int,
    strategies: np.ndarray,
    adj: List[Set[int]],
    b: float,
    c: float
) -> float:
    """Calculate normalized payoff for individual idx."""
    neighbors = list(adj[idx])
    if not neighbors:
        return 0.0
    
    n_coop = sum(strategies[n] for n in neighbors)
    payoff = b * n_coop
    if strategies[idx] == 1:
        payoff -= c * len(neighbors)
    
    return payoff / len(neighbors)


def fermi_probability(payoff_diff: float, beta: float) -> float:
    """
    Fermi update probability.
    
    Args:
        payoff_diff: pi_v - pi_u (neighbor minus focal).
        beta: Selection intensity.
    """
    if beta == 0:
        return 0.5
    
    exponent = -beta * payoff_diff
    if exponent > 50:
        return 0.0
    if exponent < -50:
        return 1.0
    return 1.0 / (1.0 + np.exp(exponent))


def rewire(
    u: int,
    v: int,
    adj: List[Set[int]],
    N: int
) -> None:
    """
    Rewire edge (u, v): remove it and add new edge from one endpoint.
    """
    if len(adj[u]) <= 1 or len(adj[v]) <= 1:
        return
    
    adj[u].remove(v)
    adj[v].remove(u)
    
    source = u if np.random.random() < 0.5 else v
    targets = list(range(N))
    np.random.shuffle(targets)
    
    for t in targets:
        if t != source and t not in adj[source]:
            adj[source].add(t)
            adj[t].add(source)
            return
    
    # Restore original edge if rewiring fails
    adj[u].add(v)
    adj[v].add(u)
