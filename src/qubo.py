"""Channel-allocation QUBO in two encodings: one-hot and binary.

This module is the technical heart of the project. It builds the weighted
graph-colouring QUBO that `solve_qaoa.py` optimises, in two variable encodings,
and decodes a solved bitstring back to a cell->channel assignment.

One-hot encoding
----------------
    variable x[v, c] in {0,1} for each cell v and channel c   -> N*K binary vars
    cost        sum over edges (u,v) of w_uv * sum_c x[u,c]*x[v,c]
                  (co-channel coupling: penalised when u and v share a channel)
    constraint   each cell uses exactly one channel:
                  lambda * sum_v ( sum_c x[v,c] - 1 )^2        (linear-in-x^2 penalty)
    -> many qubits, trivial "one colour per node" penalty, shallow cost term.

Binary encoding
---------------
    K_eff = ceil(log2(K)) bits per cell encode its channel index  -> N*ceil(log2 K) vars
    cost        sum over edges (u,v) of w_uv * [ch(u) == ch(v)]
                  where ch(v) = sum_b 2^b * bit[v, b]
                  i.e. an equality indicator expanded into products of bits ->
                  the cost Hamiltonian is denser and deeper.
    constraint   channel index must be < K (the ceil(log2 K) bit space covers
                  2^K_eff >= K values, so indices K..2^K_eff-1 are infeasible).
                  penalty: lambda * sum_v 1[ ch(v) >= K ]  (expanded as bit products)
    -> far fewer qubits, but denser/higher-weight cost Hamiltonian, smaller
       feasible subspace. This is the qubit-vs-depth-vs-feasibility trade-off
       that the encoding study reports.

Both builders return a `QuadraticProgram` whose binary variables the QAOA solver
treats as qubits. A consistent `decode_*` turns a solved bitstring into a
cell->channel dict and checks feasibility, so solvers can swap encodings
without changing call sites.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
from qiskit_optimization import QuadraticProgram
from qiskit_optimization.problems import Variable

from src.graphs import InterferenceGraph


# ---------------------------------------------------------------------------
# Shared objective helpers
# ---------------------------------------------------------------------------


def _channels(n_channels: int) -> list[int]:
    return list(range(n_channels))


# ---------------------------------------------------------------------------
# One-hot encoding
# ---------------------------------------------------------------------------


def build_onehot_qubo(ig: InterferenceGraph, *, penalty: float) -> QuadraticProgram:
    """Weighted graph-colouring QUBO, one-hot encoding.

    Variables: x[v, c] named ``x_{v}_{c}``, one per (cell, channel) pair.
    Objective: minimise co-channel interference + penalty for violating
    "each cell uses exactly one channel".
    """
    g = ig.graph
    K = ig.n_channels
    qp = QuadraticProgram(name=f"onehot_N{ig.n_cells}_K{K}")
    for v in g.nodes:
        for c in range(K):
            qp.binary_var(name=f"x_{v}_{c}")

    # --- cost term: co-channel coupling on each interfering edge ---
    linear: dict[str, float] = {}
    quadratic: dict[tuple[str, str], float] = {}
    for u, v, data in g.edges(data=True):
        w = float(data["weight"])
        for c in range(K):
            xu, xv = f"x_{u}_{c}", f"x_{v}_{c}"
            quadratic[(xu, xv)] = quadratic.get((xu, xv), 0.0) + w

    # --- constraint penalty: (sum_c x[v,c] - 1)^2 = (sum x)^2 - 2 sum x + 1 ---
    for v in g.nodes:
        xs = [f"x_{v}_{c}" for c in range(K)]
        for c in xs:
            linear[c] = linear.get(c, 0.0) - 2.0 * penalty
        for a in range(len(xs)):
            for b in range(len(xs)):
                key = (xs[a], xs[b])
                quadratic[key] = quadratic.get(key, 0.0) + penalty

    qp.minimize(linear=linear, quadratic=quadratic)
    return qp


def decode_onehot(qp: QuadraticProgram, x: np.ndarray) -> dict[int, int]:
    """Decode a one-hot bitstring into {cell: channel}. If a cell's row is not
    exactly one-hot, the argmax channel is returned and the assignment is marked
    infeasible by `is_feasible_onehot`."""
    K = _count_channels_onehot(qp)
    n_cells = qp.get_num_vars() // K
    assign: dict[int, int] = {}
    for v in range(n_cells):
        block = x[v * K:(v + 1) * K]
        assign[v] = int(np.argmax(block))
    return assign


def is_feasible_onehot(qp: QuadraticProgram, x: np.ndarray) -> bool:
    K = _count_channels_onehot(qp)
    n_cells = qp.get_num_vars() // K
    for v in range(n_cells):
        block = x[v * K:(v + 1) * K]
        if int(round(float(np.sum(block)))) != 1:
            return False
    return True


def _count_channels_onehot(qp: QuadraticProgram) -> int:
    # K is the number of colours; recover from variable naming x_{v}_{c}.
    names = qp.variables
    return 1 + max(int(var.name.split("_")[2]) for var in names)


# ---------------------------------------------------------------------------
# Binary encoding
# ---------------------------------------------------------------------------


def _keff(n_channels: int) -> int:
    return int(np.ceil(np.log2(max(n_channels, 2))))


def build_binary_qubo(ig: InterferenceGraph, *, penalty: float) -> QuadraticProgram:
    """Weighted graph-colouring QUBO, binary encoding of the channel index.

    Variables: bit[v, b] named ``b_{v}_{b}`` for b in 0..K_eff-1.
    Cell v's channel index is ch(v) = sum_b 2^b bit[v,b].

    The co-channel cost `w_uv * 1[ch(u)==ch(v)]` and the infeasibility penalty
    `1[ch(v) >= K]` are expanded into multilinear polynomials of the bits, which
    is what makes this encoding's cost Hamiltonian denser/deeper than one-hot.
    """
    g = ig.graph
    K = ig.n_channels
    Ke = _keff(K)
    qp = QuadraticProgram(name=f"binary_N{ig.n_cells}_K{K}")
    for v in g.nodes:
        for b in range(Ke):
            qp.binary_var(name=f"b_{v}_{b}")

    linear: dict[str, float] = {}
    quadratic: dict[tuple[str, str], float] = {}

    # --- co-channel cost: w_uv * 1[ch(u) == ch(v)] ---
    # 1[ch(u)==ch(v)] = product over bits b of 1[bit[u,b]==bit[v,b]]
    # 1[a==b] = 1 - a - b + 2ab  for a,b in {0,1}. Multiplying across b gives a
    # multilinear expansion in the bits; we accumulate its coefficients.
    for u, v, data in g.edges(data=True):
        w = float(data["weight"])
        prod = _equality_indicator_expansion(u, v, Ke)  # dict of term -> coeff
        _accumulate(prod, linear, quadratic, scale=w)

    # --- infeasibility penalty: 1[ch(v) >= K] for each cell ---
    for v in g.nodes:
        ind = _geq_indicator_expansion(v, Ke, K)  # 1[ch(v) >= K]
        _accumulate(ind, linear, quadratic, scale=penalty)

    qp.minimize(linear=linear, quadratic=quadratic)
    return qp


def _equality_indicator_expansion(u: int, v: int, ke: int) -> dict[frozenset[str], float]:
    """Return multilinear expansion of 1[ch(u) == ch(v)] in the bits.

    1[ch(u)==ch(v)] = Prod_b (1 - bit[u,b] - bit[v,b] + 2 bit[u,b] bit[v,b]).
    Output maps a frozenset of variable names (the monomial) to its coefficient.
    """
    base: dict[frozenset[str], float] = {frozenset(): 1.0}
    for b in range(ke):
        bu, bv = f"b_{u}_{b}", f"b_{v}_{b}"
        factor: dict[frozenset[str], float] = {
            frozenset(): 1.0,
            frozenset([bu]): -1.0,
            frozenset([bv]): -1.0,
            frozenset([bu, bv]): 2.0,
        }
        base = _poly_mul(base, factor)
    return base


def _geq_indicator_expansion(v: int, ke: int, K: int) -> dict[frozenset[str], float]:
    """Return multilinear expansion of 1[ch(v) >= K], where ch(v) = sum 2^b bit[v,b].

    Enumerate all bit patterns whose integer value is in [K, 2^ke) and add the
    corresponding minterm (product of literals) for each. Each minterm is the
    AND of bits set to 1 and (1-bit) for bits set to 0.
    """
    terms: dict[frozenset[str], float] = {}
    bits = [f"b_{v}_{b}" for b in range(ke)]
    for val in range(K, 2 ** ke):
        # minterm for val: product over b of (bit if (val>>b)&1 else (1-bit))
        minterm: dict[frozenset[str], float] = {frozenset(): 1.0}
        for b in range(ke):
            bitname = bits[b]
            if (val >> b) & 1:
                factor = {frozenset([bitname]): 1.0}
            else:
                factor = {frozenset(): 1.0, frozenset([bitname]): -1.0}
            minterm = _poly_mul(minterm, factor)
        for mono, coeff in minterm.items():
            terms[mono] = terms.get(mono, 0.0) + coeff
    return terms


def _poly_mul(a: dict[frozenset[str], float], b: dict[frozenset[str], float]) -> dict[frozenset[str], float]:
    out: dict[frozenset[str], float] = {}
    for ma, ca in a.items():
        for mb, cb in b.items():
            mono = frozenset(ma | mb)
            out[mono] = out.get(mono, 0.0) + ca * cb
    return out


def _accumulate(terms: dict[frozenset[str], float],
                linear: dict[str, float],
                quadratic: dict[tuple[str, str], float],
                *, scale: float) -> None:
    """Fold a multilinear polynomial (term -> coeff) into the QUBO's linear/quadratic
    dicts. A monomial of size 0 -> constant (skip, doesn't affect argmin offset);
    size 1 -> linear; size 2 -> quadratic (symmetric)."""
    for mono, coeff in terms.items():
        c = coeff * scale
        if c == 0.0:
            continue
        if len(mono) == 1:
            name = next(iter(mono))
            linear[name] = linear.get(name, 0.0) + c
        elif len(mono) == 2:
            it = iter(mono)
            a, b = next(it), next(it)
            key = (a, b)
            quadratic[key] = quadratic.get(key, 0.0) + c


def decode_binary(qp: QuadraticProgram, x: np.ndarray, n_channels: int) -> dict[int, int]:
    """Decode a binary bitstring into {cell: channel}. Clips infeasible indices
    (>= K) down to K-1 so a downstream objective can still be computed; use
    `is_feasible_binary` to flag infeasibility."""
    Ke = _keff(n_channels)
    n_cells = qp.get_num_vars() // Ke
    assign: dict[int, int] = {}
    for v in range(n_cells):
        block = x[v * Ke:(v + 1) * Ke]
        idx = sum(int(round(float(b))) << i for i, b in enumerate(block))
        assign[v] = min(idx, n_channels - 1)
    return assign


def is_feasible_binary(qp: QuadraticProgram, x: np.ndarray, n_channels: int) -> bool:
    Ke = _keff(n_channels)
    n_cells = qp.get_num_vars() // Ke
    for v in range(n_cells):
        block = x[v * Ke:(v + 1) * Ke]
        idx = sum(int(round(float(b))) << i for i, b in enumerate(block))
        if idx >= n_channels:
            return False
    return True


# ---------------------------------------------------------------------------
# Unified entry point
# ---------------------------------------------------------------------------

@dataclass
class Encoding:
    name: str  # "onehot" | "binary"
    qp: QuadraticProgram
    n_qubits: int
    n_channels: int

    def decode(self, x: np.ndarray) -> dict[int, int]:
        if self.name == "onehot":
            return decode_onehot(self.qp, x)
        return decode_binary(self.qp, x, self.n_channels)

    def is_feasible(self, x: np.ndarray) -> bool:
        if self.name == "onehot":
            return is_feasible_onehot(self.qp, x)
        return is_feasible_binary(self.qp, x, self.n_channels)


def build_encoding(ig: InterferenceGraph, *, encoding: str, penalty: float) -> Encoding:
    """Build the channel-allocation QUBO in the requested encoding.

    Args:
        ig: interference graph (carries n_cells, n_channels).
        encoding: "onehot" or "binary".
        penalty: lambda scaling the constraint-violation term.
    """
    if encoding == "onehot":
        qp = build_onehot_qubo(ig, penalty=penalty)
        n_q = qp.get_num_vars()
    elif encoding == "binary":
        qp = build_binary_qubo(ig, penalty=penalty)
        n_q = qp.get_num_vars()
    else:
        raise ValueError(f"unknown encoding {encoding!r}")
    return Encoding(name=encoding, qp=qp, n_qubits=n_q, n_channels=ig.n_channels)


def interference_objective(ig: InterferenceGraph, assignment: dict[int, int]) -> float:
    """Total co-channel interference of an assignment (the thing we minimise)."""
    g = ig.graph
    total = 0.0
    for u, v, data in g.edges(data=True):
        if assignment.get(u) == assignment.get(v):
            total += float(data["weight"])
    return total
