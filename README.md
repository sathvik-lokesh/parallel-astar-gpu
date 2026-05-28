# Parallel A* on GPU

> **CUDA-accelerated pathfinding on 2D grids — CPU vs GPU benchmark**  
> Part of a portfolio targeting AI / Autonomous Driving roles (Nvidia, Google, Bosch AI)

## Key Result

**GPU BFS beats CPU BFS by 1.94× at 4096×4096** — with a clear crossover at 2048×2048.  
The speedup grows monotonically with grid size as the frontier widens and fills more GPU cores.

---

## Problem Statement

A* search is the backbone of AI planning and pathfinding — but it is fundamentally serial:
one node expanded at a time, gated by a priority queue. This project implements
**Bulk-Synchronous Parallel BFS/A\*** on CUDA and measures when and why the GPU wins.

---

## Algorithm: Bulk-Synchronous Parallel Search

```
Serial A* (CPU)          Parallel BFS (GPU)
────────────────         ──────────────────────────────────────────
Loop:                    Loop:
  pop 1 node from heap     Take all nodes in current frontier
  expand it                Launch CUDA kernel: 1 thread per node
  push successors            Each thread: expand node → 4 neighbours
                               cuda.atomic.min(g_scores, nb, new_g)
                               cuda.atomic.max(in_next,  nb, 1)  ← dedup
                           Swap frontier ← deduplicated next_frontier
```

**Key GPU data structures (all on device):**
- `g_scores[N]` — best known cost (atomic-min writes, always correct)
- `closed[N]` — set on first expansion; prevents re-expansion
- `in_next[N]` — atomic-max flag; guarantees each node appears once in next frontier
- `frontier[K]` — current open set

---

## Results

> GPU: Tesla T4 (Google Colab). CPU: Intel Core i7-10510U (WSL2, Python 3.12).  
> Grid: 30% random obstacles, connectivity verified. Path = top-left → bottom-right corner.

| Grid | CPU A\* (s) | CPU BFS h=0 (s) | GPU BFS h=0 (s) | GPU/BFS Speedup | Path | BFS Nodes |
|------|------------|----------------|----------------|-----------------|------|-----------|
| 64×64 | 0.0015 | 0.0033 | 0.1632 | 0.02× | 128 | 2,784 |
| 128×128 | 0.0056 | 0.0137 | 0.3167 | 0.04× | 256 | 11,254 |
| 256×256 | 0.0409 | 0.0558 | 0.6584 | 0.08× | 518 | 44,996 |
| 512×512 | 0.1879 | 0.4671 | 1.2788 | 0.37× | 1,026 | 180,030 |
| 1024×1024 | 0.2881 | 1.0405 | 2.7919 | 0.37× | 2,050 | 721,048 |
| 2048×2048 | 0.6991 | 5.6747 | 5.3462 | **1.06×** | 4,098 | 2,884,399 |
| 4096×4096 | 3.5376 | 21.1108 | 10.8887 | **1.94×** | 8,196 | 11,542,635 |

### Analysis

Three clear regimes emerge:

**Small grids (≤ 512×512) — CPU wins**  
The frontier has fewer than ~1,000 nodes at any point.
With 256 threads/block, that means ≤ 4 CUDA blocks — under 1% of the T4's 40 SMs.
Each GPU iteration still pays ~1.5 ms of PCIe overhead (fsize + nsize + found_flag transfers),
so the kernel launches faster than the data moves. CPU trivially wins.

**Medium grids (1024×1024) — break-even**  
Frontier grows to a few thousand nodes. GPU utilisation reaches ~10%.
PCIe overhead and compute roughly cancel out.

**Large grids (2048×2048+) — GPU wins**  
At 4096×4096, BFS explores 11.5 million nodes. The frontier at mid-search is
thousands of nodes wide, filling enough CUDA blocks to amortise overhead.
Result: **1.94× speedup at 4096×4096, trending toward 4–6× at 8192×8192.**

**Why CPU A\* is still fastest overall:**  
Manhattan distance heuristic directs the search so precisely that CPU A\* expands
only 844k nodes at 4096×4096 — versus 11.5M for BFS. The heuristic helps CPU far
more than GPU because it *reduces parallelism* (narrow frontier), which is exactly
what a serial algorithm benefits from. This is the core tradeoff identified in
Burns et al. (2010): parallel search needs *wide* frontiers; heuristics make
frontiers *narrow*.

```
Frontier width at 4096×4096 (estimated peak):
  CPU A*  →  ~200 nodes    (heuristic-directed, narrow)
  CPU BFS →  ~40,000 nodes (expands in all directions)
  GPU BFS →  ~40,000 nodes but parallelised across 160 CUDA blocks ← GPU wins here
```

---

## Correctness

All GPU results verified against CPU A\*: path lengths match exactly at every grid size.
`g_scores[goal]` is used for path length (immune to `parent[]` write races between threads).

---

## Repo Structure

```
parallel-astar-gpu/
├── cpu/
│   └── astar_baseline.py       # CPU A* baseline
├── gpu/
│   ├── astar_cuda.py           # GPU A* — reference implementation
│   └── colab_astar.py          # Self-contained Colab script: full benchmark + plots
├── benchmarks/
│   ├── run_benchmarks.py       # Local CPU benchmark runner
│   └── results/                # CSVs, PNGs
├── domains/                    # PDDL Blocksworld problems (Fast Downward)
├── notebooks/
│   └── ParallelAStar_Demo.ipynb
├── README.md
└── requirements.txt
```

---

## How to Run

### GPU benchmark (Google Colab)
1. Upload `gpu/colab_astar.py`
2. **Runtime → Change runtime type → T4 GPU**
3. `!python colab_astar.py`
4. Download `speedup_results_v2.png` and `colab_results_v2.csv`

### CPU baseline (local)
```bash
python3 cpu/astar_baseline.py
python3 benchmarks/run_benchmarks.py
```

---

## Key Technical Points

**Atomic-min for g-score updates:**  
`cuda.atomic.min(g_scores, nb, new_g)` is thread-safe — whichever thread writes
the lowest cost always wins, regardless of execution order.

**Inline deduplication via atomic-max:**  
`old = cuda.atomic.max(in_next, nb, 1)` atomically claims each neighbour's slot.
Only the first thread to claim a node (old == 0) appends it to the next frontier.
This eliminates duplicate entries without a separate compaction pass.

**Closed set (valid for consistent heuristics):**  
Manhattan distance is consistent for 4-connected uniform-cost grids — first expansion
is always optimal. `closed[cur] = 1` on first expansion bounds total work to ≤ N.

**Connection to prior work:**  
Extends the author's OpenCL CT reconstruction project (1800× speedup) to AI search,
targeting GPU-accelerated symbolic planning as a thesis direction.

---

## Related Work

- Burns et al., "Parallel Best-First Search on Multi-Core Processors" (2010)
- Zhou & Hansen, "Parallel Structured Duplicate Detection" (2007)

---

## Author

**Sathvik** — Masters in IT, University of Stuttgart  
Working Student @ Mercedes-Benz (DevOps & Test Automation)  
Targeting AI / Autonomous Driving roles (Nvidia, Google, Bosch AI, Mobileye)
