# AGENTS.md

本文件是给 AI agent 看的项目速查文档，不绑定特定 AI 工具。接手任何 agent 时，把本文件喂给它即可。

---

## 项目概述

用暖启动 QAOA 给密集小蜂窝网络做信道分配。干扰图 → 图着色 QUBO（one-hot / binary 双编码）→ 冷/暖 QAOA → 移动性动态对比 → 编码权衡研究。Qiskit + Aer statevector 模拟器，仅模拟器，无真机。

---

## 关键 API 入口

```python
from src.graphs import generate_interference_graph, small_graph, mobility_sequence
from src.qubo import build_encoding, interference_objective
from src.baselines import brute_force, greedy_dfs
from src.solve_qaoa import solve_cold, solve_warm
from src.dynamics import run_dynamics
from src.encoding_study import study
```

**主函数签名：**

```python
# 构建 QUBO
enc = build_encoding(ig, encoding="onehot" | "binary", penalty=6.0) → Encoding
# enc.qp, enc.n_qubits, enc.decode(x), enc.is_feasible(x)

# 冷/暖启动 QAOA
result, enc = solve_cold(ig, encoding=..., penalty=6.0, reps=1, maxiter=100, seed=42)
result, enc = solve_warm(ig, encoding=..., penalty=6.0, reps=1, maxiter=100, seed=42, epsilon=0.25)
# result.assignment, result.objective, result.feasible, result.n_iterations, result.trace

# 动态对比
report = run_dynamics(base=ig, n_snapshots=5, encoding="onehot", maxiter=25, ...)
# report.iter_table(), report.mean_cold_gap, report.mean_warm_gap, report.warm_better_count
```

`SolveResult` 字段：
- `.assignment` — `{cell: channel}` 解码后的分配
- `.objective` — 解码分配的**真实**共信道干扰（评估质量用这个）
- `.qaoa_fval` — QUBO 原始值（含 penalty，跟 `.objective` 不是一回事）
- `.n_iterations` — 优化器实际迭代次数
- `.feasible` — 解码分配是否合法

---

## 关键约束（集成陷阱）

以下几条是踩过一遍才摸出来的，**修改求解器代码时务必逐条确认**。

### 1. 用 V2 sampler，不是 V1

用 `qiskit_aer.primitives.SamplerV2`，不是 V1 的 `Sampler`，也不是 `BackendEstimatorV2`。构造方式见 [src/solve_qaoa.py](src/solve_qaoa.py) 的 `_make_sampler`：`SamplerV2(default_shots=..., seed=..., options={'backend_options': {'method':'statevector','seed_simulator':...}})`。它内部自己建 AerSimulator，**切勿传入 `backend=` 参数**。

### 2. QAOA 必须传 transpiler

Aer 的 `SamplerV2` 跑线路时**不转译**，QAOA 的不透明 ansatz 门直接传递给 Aer 编译器会导致 `AerError: unknown instruction: QAOA`。修法：`generate_preset_pass_manager(optimization_level=1, ...)` 然后 `QAOA(sampler=..., transpiler=pm, ...)`。`QAOA.__init__` 收 `transpiler=` 和 `transpiler_options=`。

### 2b. transpiler 不能用 backend target

`_make_transpiler` 用 `basis_gates=[...]` 构建，**不用** `backend=AerSimulator()`。Aer 默认后端 target 只有 31 量子比特，one-hot 编码 32+ qubit 时报 `TranspilerError: PhysicalQubit(31) not in Target`。basis-gate 方式宽度无限，不影响模拟。

### 3. 暖启动 pre_solver 必须是连续求解器

`WarmStartQAOAOptimizer(pre_solver=..., relax_for_pre_solver=True, ...)` 会把 QUBO 松弛成连续变量，`MinimumEigenOptimizer` 解不了连续变量（报 "Continuous variables are not supported"）。用 `from qiskit_optimization.algorithms import SlsqpOptimizer`（基于 SciPy，不需要 CPLEX/Gurobi）。

### 4. 绕过 `MinimumEigenOptimizer` 包装

这个 Qiskit/qiskit-optimization 版本组合下，`MinimumEigenOptimizer._interpret_samples` 在采样索引那步会崩 `IndexError`。优化本身没问题，只是结果包装脆弱，所以我们直接调 `QAOA.compute_minimum_eigenvalue` 然后自己解 `best_measurement` 的 bitstring（见 `_decode_best_measurement`）。**不应为追求使用官方 API 而把它换回 `MinimumEigenOptimizer`**。

### 5. `GraphColoring` 不存在

`from qiskit_optimization.applications import GraphColoring` 在 qiskit-optimization 0.7 里不存在（只有 Maxcut、StableSet、VertexCover、Tsp 等）。图着色 QUBO 在 [src/qubo.py](src/qubo.py) 里手动构建。

### 6. bitstring 是小端序

Qiskit 测量串最右字符 = 0 号量子比特，跟 `QuadraticProgram.to_ising()` 的变量顺序对得上。`_decode_best_measurement` 里 `reversed(bs)` 是有意为之，不应修改为正向读取。

### 7. 量子比特上限约 26

可行性探测的硬限制（`feasibility_qubit_cap=26`）。binary 编码下 midsize 14 基站 = 28 量子比特还能跑，超出此规模后 shot-based sampler 的结果不再可靠。量子比特/深度/门数指标只分解 ansatz 不跑采样，对任意宽度都能算。

### 8. `QAOAAnsatz` 参数名改了

Qiskit 2.x 里 `QAOAAnsatz(operator=op)` 会 TypeError，用 **`QAOAAnsatz(cost_operator=op)`**。`encoding_study.py` 里 `ansatz = QAOAAnsatz(cost_operator=op, reps=reps)` 正确。

### 9. `np.sum(generator)` 已弃用

`is_feasible_binary` / `decode_binary` 里计算 bit 合成时，用 **Python 内置 `sum()`** 而非 `np.sum(generator)`，否则 NumPy 会抛 TypeError（"calling np.sum(generator) is deprecated"）。

---

## 接口契约

**`Encoding` 统一接口（src/qubo.py）：**

加新编码时实现这个接口，下游求解器不用改调用点：

```python
@dataclass
class Encoding:
    name: str            # "onehot" | "binary" | 你的新编码名
    qp: QuadraticProgram
    n_qubits: int
    n_channels: int

    def decode(self, x: np.ndarray) -> dict[int, int]:
        """bitstring → {cell: channel}"""

    def is_feasible(self, x: np.ndarray) -> bool:
        """检查解是否合法"""
```

**扩展步骤：**

- **加新编码**：在 `src/qubo.py` 加 `build_<name>_qubo`、`decode_<name>`、`is_feasible_<name>`，在 `build_encoding` 的分支里加进去，在 `src/encoding_study.py` 的 `("onehot", "binary")` 元组里加上
- **加新基线**：在 `src/baselines.py` 加函数，签名 `-> tuple[dict[int,int], float]`，用 `interference_objective` 算目标值
- **加新求解器**：照 `solve_cold` / `solve_warm` 的样子写，返回 `(SolveResult, Encoding)`，`.objective`（真实干扰）和 `.n_iterations` 必须对齐
- **加新图预设**：在 `src/graphs.py` 加常量 + 构造函数，大图先算量子比特数，超过 26 后可行性探测将不可靠

---

## 项目约定

- **术语**：中文语句中用"基站"指代网络单元，用"节点"指代图顶点（不用"小区"）
- **编码名**：使用 `binary`（不用"二进制"）
- **中文标点**：正文逗号 `，`、括号 `（）`、冒号 `：` 用全角；代码块和行内代码里保留半角
- **风格**：准确、清晰、稍带书面感，避免网络口语，也不要写成学术论文
