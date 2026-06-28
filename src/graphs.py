"""Interference-graph generation for dense small-cell channel allocation.

A small-cell network is modelled as a weighted graph:
  - nodes  = small cells (with a 2-D position)
  - edges  = pairs of cells that interfere if they reuse the same channel
  - weight = interference coupling strength (here: decreasing in distance)

Edge weight follows a simple distance-based coupling model so that "minimize
total co-channel interference" is a well-defined weighted-graph-coloring
objective. The generator is deterministic given a seed so cold/warm-start
comparisons are reproducible.

Two size presets back the demo:
  - small  : 6-10 cells, 3-4 channels  -> runs live in the pitch
  - midsize: 15-25 cells, 3-5 channels -> scalability point for the chart
"""

from __future__ import annotations

from dataclasses import dataclass

import networkx as nx
import numpy as np


def interference_weight(distance: float, *, d_ref: float = 0.3, pathloss_exp: float = 2.0) -> float:
    """Interference coupling between two cells `distance` apart.

    Monotonically decreasing in distance (closer cells interfere more), so
    assigning them the same colour is penalised harder. A distance-dependent
    path-loss-style model: w = (d_ref / max(distance, d_ref)) ** pathloss_exp,
    so cells at or below the reference distance get w=1 (maximal coupling) and
    coupling falls off as 1/d^pathloss_exp beyond that.
    """
    d = max(float(distance), d_ref)
    return float((d_ref / d) ** pathloss_exp)


@dataclass
class InterferenceGraph:
    """A generated interference graph plus its generation parameters."""

    graph: nx.Graph
    n_cells: int
    n_channels: int
    radius: float
    seed: int

    @property
    def positions(self) -> dict[int, tuple[float, float]]:
        return {i: tuple(self.graph.nodes[i]["pos"]) for i in self.graph.nodes}

    @property
    def weights(self) -> dict[tuple[int, int], float]:
        return {(u, v): float(d["weight"]) for u, v, d in self.graph.edges(data=True)}


def generate_interference_graph(
    n_cells: int,
    n_channels: int,
    *,
    radius: float = 1.0,
    seed: int = 0,
    pathloss_exp: float = 2.0,
) -> InterferenceGraph:
    """Generate a random-geometric interference graph for `n_cells` small cells.

    Cells are scattered uniformly in the unit square scaled by `radius`. An edge
    connects two cells whose distance is below `radius * 0.9` (i.e. they are
    close enough to interfere); the edge weight is `interference_weight(distance)`.

    Args:
        n_cells: number of small cells (graph nodes).
        n_channels: number of available channels (graph colours). Stored on the
            object so downstream QUBO builders don't need it passed separately.
        radius: side length of the square region cells live in.
        seed: RNG seed for reproducibility.
        pathloss_exp: path-loss exponent for the coupling model.
    """
    rng = np.random.default_rng(seed)
    pos = rng.uniform(0.0, radius, size=(n_cells, 2))

    g = nx.Graph()
    for i in range(n_cells):
        g.add_node(i, pos=tuple(pos[i]))

    cutoff = radius * 0.9
    for i in range(n_cells):
        for j in range(i + 1, n_cells):
            d = float(np.linalg.norm(pos[i] - pos[j]))
            if d < cutoff:
                w = interference_weight(d, pathloss_exp=pathloss_exp)
                g.add_edge(i, j, weight=w, distance=d)

    return InterferenceGraph(graph=g, n_cells=n_cells, n_channels=n_channels,
                             radius=radius, seed=seed)


def perturb_graph(
    ig: InterferenceGraph, *, max_shift: float = 0.15, seed: int = 1,
) -> InterferenceGraph:
    """Create the *next mobility snapshot*: nudge each cell position slightly,
    recompute edges/weights. Adjacent snapshots differ by a small perturbation —
    the structural feature warm-start QAOA exploits.
    """
    rng = np.random.default_rng(seed)
    pos = np.array([ig.graph.nodes[i]["pos"] for i in range(ig.n_cells)])
    pos = pos + rng.uniform(-max_shift, max_shift, size=pos.shape)
    pos = np.clip(pos, 0.0, ig.radius)

    g = nx.Graph()
    for i in range(ig.n_cells):
        g.add_node(i, pos=tuple(pos[i]))
    cutoff = ig.radius * 0.9
    for i in range(ig.n_cells):
        for j in range(i + 1, ig.n_cells):
            d = float(np.linalg.norm(pos[i] - pos[j]))
            if d < cutoff:
                w = interference_weight(d)
                g.add_edge(i, j, weight=w, distance=d)
    return InterferenceGraph(graph=g, n_cells=ig.n_cells, n_channels=ig.n_channels,
                             radius=ig.radius, seed=seed)


# ---------------------------------------------------------------------------
# Presets used across the demo / scalability study.
#
# `MIDSIZE` is sized to stay within the Aer statevector sampler's reliable
# width for this stack (~28 qubits for the binary encoding, ~24 for one-hot).
# 14 cells x 2 bits = 28 qubits runs; 16 cells x 2 bits = 32 qubits hits an
# Aer SamplerV2 postprocess edge case. We state this as an honest scale limit
# in the pitch rather than hide it. The (binary) encoding is what makes the
# mid-size point runnable at all: one-hot at 14 cells x 4 channels = 56 qubits
# is far past the simulator.
# ---------------------------------------------------------------------------

SMALL = {"n_cells": 8, "n_channels": 3, "radius": 1.0}
MIDSIZE = {"n_cells": 14, "n_channels": 4, "radius": 1.4}


def small_graph(seed: int = 0) -> InterferenceGraph:
    return generate_interference_graph(seed=seed, **SMALL)


def midsize_graph(seed: int = 0) -> InterferenceGraph:
    return generate_interference_graph(seed=seed, **MIDSIZE)


def mobility_sequence(base: InterferenceGraph, n_snapshots: int, *, seed: int = 1) -> list[InterferenceGraph]:
    """Return [base, snapshot_1, ..., snapshot_{n_snapshots-1}], each a perturbation
    of the previous one."""
    seq = [base]
    cur = base
    for k in range(n_snapshots - 1):
        cur = perturb_graph(cur, seed=seed + k)
        seq.append(cur)
    return seq
