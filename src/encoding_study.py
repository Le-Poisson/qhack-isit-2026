"""One-hot vs binary encoding trade-off study.

For fixed interference graphs, report the three axes of the trade-off the pitch
claims:

  - qubits            : N*K (one-hot) vs N*ceil(log2 K) (binary)
  - circuit depth/gates: transpile the QAOA cost-layer ansatz for each encoding's
                        Ising Hamiltonian and report depth + gate count. Binary's
                        cost Hamiltonian is denser (co-channel equality expands to
                        products of bits) -> deeper cost layer.
  - feasible rate     : run QAOA (cold, fixed budget) for each encoding on the
                        same graphs and report how often the decoded bitstring is
                        a valid assignment (one-hot: exactly-one-channel; binary:
                        channel index < K). Binary's feasible subspace is smaller.

The output is a table the demo notebook renders as the pitch's encoding chart.
"""

from __future__ import annotations

import warnings
from dataclasses import dataclass

import numpy as np
from qiskit.circuit.library import QAOAAnsatz

from src.graphs import InterferenceGraph
from src.qubo import build_encoding
from src.solve_qaoa import solve_cold

warnings.filterwarnings("ignore")


@dataclass
class EncodingRow:
    graph_id: str
    n_cells: int
    n_channels: int
    encoding: str
    n_qubits: int
    depth: int
    n_gates: int
    feasible_rate: float


def _ansatz_metrics(qp, *, reps: int = 1) -> tuple[int, int]:
    """Decompose the QAOA ansatz for this QUBO's Ising Hamiltonian and return
    (depth, gate_count). The cost Hamiltonian's density drives the depth here.

    We decompose rather than transpile-against-a-target: transpiling needs a
    device target, and one-hot at large N*K can exceed a default target's qubit
    count and fail spuriously. Decomposition still reflects the cost-Hamiltonian
    density difference (the thing we want to measure) via CNOT-exponential
    expansion of each Pauli term.
    """
    op, _offset = qp.to_ising()
    ansatz = QAOAAnsatz(cost_operator=op, reps=reps)
    param_vals = np.array([0.1] * ansatz.num_parameters)
    bound = ansatz.assign_parameters(param_vals) if ansatz.num_parameters else ansatz
    dec = bound.decompose(reps=3)
    depth = dec.depth()
    gates = sum(dec.count_ops().values())
    return depth, gates


def study_graph(
    ig: InterferenceGraph,
    *,
    graph_id: str,
    penalty: float,
    reps: int = 1,
    n_runs: int = 5,
    maxiter: int = 80,
    seed: int = 42,
    feasibility_qubit_cap: int = 26,
) -> list[EncodingRow]:
    """Produce one EncodingRow per encoding for the given graph.

    The qubit/depth/gate metrics come from decomposing the QAOA ansatz and work
    for any width. The feasibility-rate probe runs QAOA on the Aer sampler and
    is reported as NaN when the encoding's qubit count exceeds
    `feasibility_qubit_cap` — Aer's shot-based statevector sampler becomes
    unreliable past ~26 qubits in this stack, an honest scale limitation we
    state in the pitch rather than hide.
    """
    rows: list[EncodingRow] = []
    for enc_name in ("onehot", "binary"):
        enc = build_encoding(ig, encoding=enc_name, penalty=penalty)
        depth, gates = _ansatz_metrics(enc.qp, reps=reps)

        if enc.n_qubits <= feasibility_qubit_cap:
            feasible = 0
            for k in range(n_runs):
                res, _ = solve_cold(ig, encoding=enc_name, penalty=penalty, reps=reps,
                                    maxiter=maxiter, shots=2048, seed=seed + k)
                if res.feasible:
                    feasible += 1
            feas_rate = feasible / n_runs
        else:
            feas_rate = float("nan")
        rows.append(EncodingRow(
            graph_id=graph_id,
            n_cells=ig.n_cells,
            n_channels=ig.n_channels,
            encoding=enc_name,
            n_qubits=enc.n_qubits,
            depth=depth,
            n_gates=gates,
            feasible_rate=feas_rate,
        ))
    return rows


def study(*, n_cells: int, n_channels: int, seed: int = 0,
          penalty: float = 6.0, reps: int = 1, n_runs: int = 5,
          feasibility_qubit_cap: int = 26) -> list[EncodingRow]:
    """Convenience: study one (N, K) configuration. Returns [onehot_row, binary_row]."""
    from src.graphs import generate_interference_graph
    ig = generate_interference_graph(n_cells, n_channels, seed=seed)
    gid = f"N{n_cells}_K{n_channels}"
    return study_graph(ig, graph_id=gid, penalty=penalty, reps=reps, n_runs=n_runs,
                       feasibility_qubit_cap=feasibility_qubit_cap)


def print_table(rows: list[EncodingRow]) -> None:
    print(f"{'graph':>9} {'enc':>7} {'qubits':>7} {'depth':>7} {'gates':>7} {'feas%':>6}")
    for r in rows:
        feas = "  --" if r.feasible_rate != r.feasible_rate else f"{r.feasible_rate*100:>5.0f}"  # NaN check
        print(f"{r.graph_id:>9} {r.encoding:>7} {r.n_qubits:>7} {r.depth:>7} "
              f"{r.n_gates:>7} {feas:>6}")


def main() -> None:
    import argparse
    p = argparse.ArgumentParser(description="One-hot vs binary encoding trade-off")
    p.add_argument("--cells", type=int, default=8)
    p.add_argument("--channels", type=int, default=3)
    p.add_argument("--reps", type=int, default=1)
    p.add_argument("--runs", type=int, default=5)
    args = p.parse_args()
    rows = study(n_cells=args.cells, n_channels=args.channels, reps=args.reps, n_runs=args.runs)
    print_table(rows)


if __name__ == "__main__":
    main()
