# Quantum-Boosted Channel Allocation（ISIT 2026 QHack）

用暖启动 QAOA 给密集小蜂窝网络做信道分配。这是给项目协作者和后续接手的 agent 看的文档：重点讲 [src/](src/) 里的代码怎么用、架构怎么读、扩展时应避免的常见陷阱。

## 这个项目在干什么

密集小蜂窝网络要给每个基站分配信道，目标是让相互干扰的基站尽量不用同一个信道。这本质上是 **加权图着色** 问题：

- 节点 = 基站，边 = 两个基站会互相干扰，边权 = 干扰耦合强度（离得越近越强）
- 给每个节点涂一种颜色（信道），目标是让同色节点之间的总边权最小
- 图着色是 NP-hard，而且这问题在真实网络里 **反复出现**：用户和基站会移动，干扰图每个快照都在变，同一族 QUBO 要一遍遍重新求解

我们的做法：把这个图着色建成 QUBO，用 Qiskit 的混合 QAOA（QAOA 线路 + 经典 COBYLA 优化参数）在 Aer statevector 模拟器上跑。针对 "反复求解" 这个特征做两件事：

1. **暖启动 QAOA 做动态 QUBO 的低成本增量重优化**：把上一个快照的解松弛成连续振幅，用来初始化 ansatz 态和 mixer，让这一快照的参数搜索从上一个解的 "盆地" 里起步。在固定迭代预算下（每次迭代才是 NISQ 硬件上最贵的部分），暖启动比冷启动更接近最优。
2. **one-hot vs binary 编码选择规则**：one-hot 用 `N·K` 个量子比特，约束项浅；binary 用 `N·⌈log₂K⌉` 个量子比特，但 cost Hamiltonian 更稠密更深、可行子空间更小。我们基准对比，给出 "什么场景用哪种编码" 的规则。

**诚实边界**：暖启动 QAOA 不是我们发明的，是 Egger, Mareček & Woerner 的工作。我们的贡献是 **把它用在动态/反复求解的 QUBO 场景** + **编码选择研究**，不是宣称发明了暖启动。熟悉该问题的研究者清楚暖启动 QAOA 的既有工作，因此我们如实引用。

## 技术栈与版本

环境是 Python 3.12 + `uv`，依赖锁定在 [uv.lock](uv.lock) 里。关键版本：

| 包 | 版本 | 用途 |
|---|---|---|
| qiskit | 2.4.2 | 线路、transpiler、ansatz |
| qiskit-aer | 0.17.2 | statevector 模拟器 + V2 sampler |
| qiskit-algorithms | 0.4.0 | QAOA、WarmStartQAOAOptimizer、COBYLA |
| qiskit-optimization | 0.7.0 | QuadraticProgram、SlsqpOptimizer |
| networkx | ≥3.6 | 干扰图 |
| numpy / scipy | | 数值、连续松弛 |
| matplotlib / pandas / ipykernel | | demo 可视化 |

装环境：

```bash
uv venv --python 3.12
uv sync            # 按 uv.lock 装依赖
```

之后用 `uv run <命令>` 跑东西，不用手动 activate。

## 快速开始

跑 demo notebook（端到端流水线 + 5 项验证）：

```bash
uv run jupyter lab notebooks/demo.ipynb
```

或者直接跑两条 CLI：

```bash
# 冷启动 vs 暖启动，跑一条移动性快照序列（项目的主图表）
uv run python -m src.dynamics --cells 8 --channels 3 --snapshots 5 --encoding onehot --maxiter 25

# one-hot vs binary 编码权衡表
uv run python -m src.encoding_study --cells 8 --channels 3 --runs 5
```

`--maxiter 25` 是故意的：暖启动的优势在 **紧预算** 下才显出来，预算给够了冷启动也能收敛到差不多。详见 [src/dynamics.py](src/dynamics.py) 顶部的注释。

## 代码结构与数据流

```
src/
├── graphs.py          # 干扰图生成 + 移动性扰动 + small/midsize 预设
├── qubo.py            # one-hot / binary 两种编码的图着色 QUBO + 解码 + 可行性
├── baselines.py       # 暴力法 + DSATUR 贪心（对照基准）
├── solve_qaoa.py      # 冷启动 / 暖启动 QAOA（Aer statevector）
├── dynamics.py        # 移动性循环：冷 vs 暖，固定预算下对比（主图表）
└── encoding_study.py  # 编码权衡：量子比特 / 深度 / 门数 / 可行率
```

数据流：

```
src/graphs              干扰图生成 + 移动性扰动
       │
       ▼
src/qubo                图着色 QUBO（one-hot / binary 双编码）
       │
       ├─── src/baselines            经典基线：暴力法 · DSATUR 贪心
       │
       └─── src/solve_qaoa           冷启动 QAOA / 暖启动 QAOA
                 │
                 ├─── src/dynamics         动态对比：冷 vs 暖 @ 固定预算
                 │
                 └─── src/encoding_study   编码权衡：qubits · depth · gates · feas%
```

每个模块顶部的 docstring 都把这模块做什么、为什么这么做讲清楚了，建议修改前通读一遍。

## 各模块怎么用

### src/graphs.py — 干扰图

生成随机几何干扰图：基站撒在正方形里，距离够近就连边，边权按路径损耗衰减。

```python
from src.graphs import generate_interference_graph, small_graph, midsize_graph, mobility_sequence

ig = small_graph(seed=0)          # 8 基站 3 信道，用于 demo
mid = midsize_graph(seed=0)       # 14 基站 4 信道，用于可扩展性演示
ig = generate_interference_graph(n_cells=10, n_channels=4, seed=0)  # 自定义

seq = mobility_sequence(ig, n_snapshots=5, seed=1)  # [快照 0, 快照 1, ...]，每个是上一个的微扰
```

`InterferenceGraph` 拿得到的东西：`.graph`（nx.Graph，节点带 `pos`、边带 `weight`/`distance`）、`.n_cells`、`.n_channels`、`.positions`、`.weights`。`.n_channels` 是存在图对象上的，下游 QUBO 构造不用再传一次。

**两个预设是调过的**：`SMALL = {8 cells, 3 channels}`，`MIDSIZE = {14 cells, 4 channels}`。MIDSIZE 选 14 是因为 binary 编码下 `14×2=28` 量子比特还能跑，`16×2=32` 会撞 Aer SamplerV2 的一个后处理边界。这是当前的规模上限，超出会导致求解失败。

### src/qubo.py — QUBO 构造（项目技术核心）

把图着色建成 QUBO，两种编码：

```python
from src.qubo import build_encoding, interference_objective

enc = build_encoding(ig, encoding="onehot", penalty=6.0)  # 或 "binary"
# enc.qp        -> qiskit QuadraticProgram
# enc.n_qubits  -> one-hot: N*K；binary: N*ceil(log2 K)
# enc.decode(x) -> bitstring 解码成 {cell: channel}
# enc.is_feasible(x) -> 是否合法（one-hot：每格恰好一色；binary：索引 < K）

obj = interference_objective(ig, assignment)  # 某分配的真实共信道干扰（要最小化的目标）
```

`penalty` 是约束违反项的系数 λ。6.0 在小图上验过；图大了要调高（比如 midsize 用 10.0，见 demo cell 6），不然 QAOA 容易翻到不可行解。

两套编码的差别都在 [src/qubo.py](src/qubo.py) 顶部 docstring 里写明白了：
- **one-hot**：`x[v,c]` 每对（基站，信道）一个变量 → `N·K` 量子比特，约束 `(Σc x[v,c] - 1)²` 很浅
- **binary**：每基站 `⌈log₂K⌉` 个比特编码信道索引 → 量子比特少很多，但 `1[ch(u)==ch(v)]` 要展开成比特的多线性多项式，cost Hamiltonian 又稠又深

`Encoding` 是统一入口：不管哪种编码，`.decode()` / `.is_feasible()` 接口一样，所以下游求解器可以随便换编码，不用改调用点。**加新编码时实现这个接口就行**（见下面扩展指南）。

### src/baselines.py — 经典对照

```python
from src.baselines import brute_force, greedy_dfs

assign, obj = brute_force(ig)   # 穷举 K^N，只对小图能用，给真实最优
assign, obj = greedy_dfs(ig)    # DSATUR 式贪心，快、可扩展，是量子方法需要超越的基线
```

`brute_force` 在 [src/dynamics.py](src/dynamics.py) 里只在 `n_cells <= 12` 时才调，大图直接跳过（标 NaN）。

### src/solve_qaoa.py — 冷启动 / 暖启动 QAOA

```python
from src.solve_qaoa import solve_cold, solve_warm

rc, enc = solve_cold(ig, encoding="onehot", penalty=6.0, reps=1, maxiter=100, seed=42)
rw, enc = solve_warm(ig, encoding="onehot", penalty=6.0, reps=1, maxiter=100, seed=42, epsilon=0.25)
# 返回 (SolveResult, Encoding)
```

`SolveResult` 字段：
- `.assignment` — `{cell: channel}` 解码后的分配
- `.objective` — 解码分配的 **真实** 共信道干扰（这才是评估质量的数）
- `.feasible` — 解码分配是否合法
- `.qaoa_fval` — 采样到那个 bitstring 时 QUBO 原始值（含 penalty，跟 `.objective` 不是一回事）
- `.n_iterations` — 优化器实际跑了几步（暖 vs 冷的核心指标）
- `.trace` — 每步的目标值，画收敛曲线用

`solve_warm` 的 `epsilon` 控制初始态离松弛解有多近（0~1，小 = 贴得近）。暖启动的优势体现在 **固定 `maxiter`** 下：预算给紧了（`maxiter=25`），冷启动还没收敛，暖启动因为从上个解的盆地起步，已经接近最优了。预算给足（`maxiter=120+`）两者差不多，这符合预期。

### src/dynamics.py — 移动性动态对比（主图表）

```python
from src.dynamics import run_dynamics

rep = run_dynamics(base=ig, n_snapshots=5, encoding="onehot",
                   penalty=6.0, maxiter=25, seed=42, perturb_seed=1)
# rep.iter_table()         -> 每快照一行 dict
# rep.mean_cold_gap / .mean_warm_gap   -> 平均到最优的 gap
# rep.warm_better_count    -> 暖启动更优的快照数
# rep.total_cold_iters / .total_warm_iters
```

快照 0 两个路径都冷启动（没有上个解可暖）。快照 1 之后：冷启动从头优化，暖启动从上一快照的松弛解起步。可辩护的结论是 **固定预算下的近似比**，不是 "暖启动迭代更少"（虽然我们也报迭代数，但那不是主卖点）。

### src/encoding_study.py — 编码权衡

```python
from src.encoding_study import study, study_graph, print_table

rows = study(n_cells=8, n_channels=3, reps=1, n_runs=5)   # 返回 [onehot_row, binary_row]
print_table(rows)
```

每行 `EncodingRow` 报四轴：`n_qubits`（量子比特）、`depth`/`n_gates`（分解 QAOA ansatz 得到的深度/门数）、`feasible_rate`（冷启动 QAOA 跑 n 次里合法解的比例）。

可行性探测有个 `feasibility_qubit_cap=26` 的上限：超过约 26 量子比特，Aer 的 shot-based sampler 就不可靠了，所以可行性那一栏报 NaN。**这是当前的规模限制，扩展时不应删除或掩盖。** 量子比特/深度/门数那几栏对任意宽度都算得出（只分解 ansatz，不跑采样），所以大图也有数据。

## 扩展指南

扩展新编码、新基线、新求解器、新图预设的具体步骤见 [AGENTS.md](AGENTS.md) 的"接口契约"部分。核心原则：

- **加新编码**：在 `src/qubo.py` 实现 `Encoding` 接口（`.decode()` / `.is_feasible()`），下游求解器不用改
- **加新求解器**：返回 `(SolveResult, Encoding)`，`.objective`（真实干扰）和 `.n_iterations` 必须对齐
- **加新基线/图预设**：签名和约束见 [AGENTS.md](AGENTS.md)

## 验证检查表

demo notebook（[notebooks/demo.ipynb](notebooks/demo.ipynb)）按这 5 项验证跑了一遍，改了代码后重跑确认没退化：

| # | 验证 | 在 demo 的 | 复现 |
|---|---|---|---|
| 1 | QUBO 正确性：两种编码的 QUBO argmin 都等于暴力法最优 | §2 | demo cell 2 |
| 2 | 冷启动质量：QAOA 打平/超过贪心，接近暴力最优 | §3 | demo cell 3 |
| 3 | 暖启动价值：固定预算下，暖启动在多数快照上更接近最优 | §4 | `uv run python -m src.dynamics --maxiter 25` |
| 4 | 编码权衡：binary 省一半量子比特，代价是更稠更深的 cost Hamiltonian | §5 | `uv run python -m src.encoding_study` |
| 5 | 可扩展性：midsize 图在 binary 编码下能跑（暴力法不可行） | §6 | demo cell 6 |

memory 里记的一条历史验证：两种编码的 argmin 都对上暴力法最优 `1.8057`（小图）。

## 诚实的局限

- **仅模拟器**，未接入真机，受平台规则限制（仅 Qiskit + Aer）
- **纯 CPU 运行**，Aer sampler 未启用 GPU 加速——协作者可考虑通过 `options={'method':'statevector'}` + cuStateVec 后端启用 GPU，突破 statevector 模拟的内存和规模瓶颈
- **约 26 量子比特是可行性探测的硬上限**，再大 Aer shot-based sampler 不可靠
- **暂无噪声模型**，QAOA 深度受模拟器成本限制（`reps=1` 为主）
- 暖启动 vs 冷启动的结论是 **固定预算下近似更优**，不是 "更少迭代必然更优"——不应在材料中夸大其词

## 参考

- Egger, Mareček & Woerner, *Warm-starting quantum optimization*（暖启动 QAOA 原始工作，见仓库内同名 PDF）
- 路演材料：[Napkin_Pitch.md](Napkin_Pitch.md)、[Napkin_Pitch_Speech.md](Napkin_Pitch_Speech.md)
- 项目想法池：[Project_Ideas.md](Project_Ideas.md) / [Project_Ideas_zh.md](Project_Ideas_zh.md)
- 赛程：[Detailed_Schedule.md](Detailed_Schedule.md)
- 答疑记录：[QA_log_zh.md](QA_log_zh.md)
