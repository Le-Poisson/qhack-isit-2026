"""Classical baselines for channel allocation.

  - brute_force : exhaustive over all K^N assignments (only tractable for small
    N; gives the true optimum -> used to validate the QUBO and QAOA quality).
  - greedy_dfs  : DSATUR-style greedy colouring with weighted ordering. Fast,
    scalable, the heuristic the quantum approach is benchmarked against.

Both return (assignment, objective) where objective is the total co-channel
interference from `qubo.interference_objective`.
"""

from __future__ import annotations

import itertools

import numpy as np

from src.graphs import InterferenceGraph
from src.qubo import interference_objective


def brute_force(ig: InterferenceGraph) -> tuple[dict[int, int], float]:
    """Exhaustive search over all K^N channel assignments. Returns the optimum."""
    g = ig.graph
    K = ig.n_channels
    nodes = list(g.nodes)
    best_assign: dict[int, int] = {}
    best_obj = float("inf")
    for combo in itertools.product(range(K), repeat=len(nodes)):
        assign = dict(zip(nodes, combo))
        obj = interference_objective(ig, assign)
        if obj < best_obj - 1e-12:
            best_obj = obj
            best_assign = assign
    return best_assign, float(best_obj)


def greedy_dfs(ig: InterferenceGraph) -> tuple[dict[int, int], float]:
    """DSATUR-style greedy weighted colouring.

    Order cells by descending weighted degree (most-constrained-first). For each
    cell, pick the channel that minimises the incremental co-channel interference
    with already-coloured neighbours. This is the classic heuristic baseline.
    """
    g = ig.graph
    K = ig.n_channels
    nodes = list(g.nodes)
    # weighted degree = sum of edge weights incident to the node
    wdeg = {n: sum(float(d["weight"]) for _, _, d in g.edges(n, data=True)) for n in nodes}
    order = sorted(nodes, key=lambda n: wdeg[n], reverse=True)

    assign: dict[int, int] = {}
    for v in order:
        nb_chans = [(assign[u], float(g[v][u]["weight"])) for u in g.neighbors(v) if u in assign]
        best_c, best_cost = 0, float("inf")
        for c in range(K):
            cost = sum(wgt for ch, wgt in nb_chans if ch == c)
            if cost < best_cost - 1e-12:
                best_cost, best_c = cost, c
        assign[v] = best_c
    return assign, interference_objective(ig, assign)
