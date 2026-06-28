"""Hybrid QAOA solvers for channel allocation: cold-start and warm-start.

Both run on an Aer statevector simulator via a V2 sampler. The cold-start solver
re-optimises all variational parameters from scratch each call. The warm-start
solver relaxes the QUBO to a continuous problem, solves it classically, and
seeds the QAOA ansatz state + mixer from that relaxation so the variational
search starts near the previous solution's basin.

Stack integration (root-caused once, reused everywhere):
  - qiskit_algorithms.QAOA needs a V2 sampler  -> qiskit_aer.primitives.SamplerV2
  - Aer's SamplerV2 does NOT transpile, so pass a pass_manager to QAOA(transpiler=)
    or Aer raises "unknown instruction: QAOA"
  - warm-start's pre_solver must be continuous -> SlsqpOptimizer (SciPy)

We call `QAOA.compute_minimum_eigenvalue` directly and decode `best_measurement`
ourselves, rather than wrapping in `MinimumEigenOptimizer`, because the latter's
sample-indexing packaging crashes on this Qiskit/qiskit-optimization version
combo (IndexError in _interpret_samples). The optimization itself is fine; only
the result packaging is fragile, so we bypass it.

An optimizer callback records the per-iteration objective so callers can report
optimizer-iteration counts (the warm-start-vs-cold-start metric).
"""

from __future__ import annotations

import warnings
from dataclasses import dataclass, field
from typing import Callable

import numpy as np
from qiskit_aer.primitives import SamplerV2 as AerSamplerV2
from qiskit_algorithms import QAOA
from qiskit_algorithms.optimizers import COBYLA
from qiskit_optimization.algorithms import SlsqpOptimizer, WarmStartQAOAOptimizer
from qiskit.transpiler.preset_passmanagers import generate_preset_pass_manager

from src.graphs import InterferenceGraph
from src.qubo import Encoding, build_encoding, interference_objective

warnings.filterwarnings("ignore")


def _make_sampler(*, shots: int, seed: int) -> AerSamplerV2:
    return AerSamplerV2(
        default_shots=shots,
        seed=seed,
        options={"backend_options": {"method": "statevector", "seed_simulator": seed}},
    )


def _make_transpiler():
    return generate_preset_pass_manager(
        optimization_level=1,
        basis_gates=["u", "cx", "rz", "sx", "x", "h"],
    )


def _iteration_callback(record: list[float]) -> Callable | None:
    """Return a QAOA callback that appends each optimizer-step objective to `record`."""
    def cb(eval_count, params, mean, metadata):  # noqa: ANN001
        record.append(float(mean))
    return cb


@dataclass
class SolveResult:
    assignment: dict[int, int]
    objective: float          # true co-channel interference of the decoded assignment
    feasible: bool
    qaoa_fval: float          # raw QUBO objective at the sampled bitstring (cost + penalty)
    n_iterations: int         # number of optimizer steps recorded (== len(trace))
    trace: list[float] = field(default_factory=list)


def _decode_best_measurement(qp, eigen_result, enc: Encoding, ig: InterferenceGraph) -> SolveResult:
    """Decode QAOA's best_measurement bitstring into a SolveResult.

    Qiskit measurement bitstrings are little-endian (rightmost char = qubit 0),
    matching `QuadraticProgram.to_ising()` variable ordering. We pad/trim to the
    QP's variable count defensively.
    """
    bs = eigen_result.best_measurement["bitstring"]
    n = qp.get_num_vars()
    bits = [int(c) for c in reversed(bs)]
    if len(bits) < n:
        bits += [0] * (n - len(bits))
    x = np.array(bits[:n], dtype=float)

    assignment = enc.decode(x)
    return SolveResult(
        assignment=assignment,
        objective=interference_objective(ig, assignment),
        feasible=enc.is_feasible(x),
        qaoa_fval=float(eigen_result.best_measurement.get("value", float("nan"))),
        n_iterations=0,  # filled by caller via trace
    )


def solve_cold(
    ig: InterferenceGraph,
    *,
    encoding: str = "onehot",
    penalty: float = 6.0,
    reps: int = 1,
    maxiter: int = 100,
    shots: int = 4096,
    seed: int = 42,
) -> tuple[SolveResult, Encoding]:
    """Cold-start QAOA on the channel-allocation QUBO.

    Returns (result, encoding). Re-optimises all variational parameters from
    scratch — the baseline warm-start is compared against.
    """
    enc = build_encoding(ig, encoding=encoding, penalty=penalty)
    qp = enc.qp
    sampler = _make_sampler(shots=shots, seed=seed)
    pm = _make_transpiler()

    trace: list[float] = []
    qaoa = QAOA(
        sampler=sampler,
        optimizer=COBYLA(maxiter=maxiter, tol=1e-4),
        reps=reps,
        transpiler=pm,
        callback=_iteration_callback(trace),
    )
    op, _offset = qp.to_ising()
    eigen = qaoa.compute_minimum_eigenvalue(operator=op)

    out = _decode_best_measurement(qp, eigen, enc, ig)
    out.n_iterations = len(trace)
    out.trace = trace
    return out, enc


def solve_warm(
    ig: InterferenceGraph,
    *,
    encoding: str = "onehot",
    penalty: float = 6.0,
    reps: int = 1,
    maxiter: int = 100,
    shots: int = 4096,
    seed: int = 42,
    epsilon: float = 0.25,
) -> tuple[SolveResult, Encoding]:
    """Warm-start QAOA. SlsqpOptimizer solves the continuous relaxation; the
    warm-start factory turns that relaxation into a biased initial state + a
    warm-start mixer, and `epsilon` controls how close the initial state sits
    to the relaxation.

    We replicate `WarmStartQAOAOptimizer.solve`'s seeding (relax -> pre-solve ->
    create_initial_state + create_mixer) but then call
    `compute_minimum_eigenvalue` ourselves instead of the library's
    `MinimumEigenOptimizer` packaging, which crashes on this version combo.

    For the dynamic regime, the relaxation of snapshot t is close to that of
    snapshot t-1 (graphs differ by a small perturbation), so the warm-start
    basin seeds the new QAOA search near the previous optimum — the structural
    feature that makes warm-start converge in fewer iterations.
    """
    enc = build_encoding(ig, encoding=encoding, penalty=penalty)
    qp = enc.qp
    sampler = _make_sampler(shots=shots, seed=seed)
    pm = _make_transpiler()

    trace: list[float] = []
    qaoa = QAOA(
        sampler=sampler,
        optimizer=COBYLA(maxiter=maxiter, tol=1e-4),
        reps=reps,
        transpiler=pm,
        callback=_iteration_callback(trace),
    )
    pre = SlsqpOptimizer()
    # Build the optimizer only to access its warm-start factory and relaxation
    # helper; we never call .solve() (its packaging crashes).
    ws = WarmStartQAOAOptimizer(
        pre_solver=pre,
        relax_for_pre_solver=True,
        qaoa=qaoa,
        epsilon=epsilon,
        num_initial_solutions=1,
    )

    relaxed = ws._relax_problem(qp)
    pre_result = pre.solve(relaxed)
    init_vars = ws._warm_start_factory.create_initial_variables(pre_result.x)
    qaoa.initial_state = ws._warm_start_factory.create_initial_state(init_vars)
    qaoa.mixer = ws._warm_start_factory.create_mixer(init_vars)

    op, _offset = qp.to_ising()
    eigen = qaoa.compute_minimum_eigenvalue(operator=op)

    out = _decode_best_measurement(qp, eigen, enc, ig)
    out.n_iterations = len(trace)
    out.trace = trace
    return out, enc
