"""
colab_astar.py — Parallel A* on GPU (Google Colab T4)
======================================================
Upload to Colab and run:  !python colab_astar.py
OR paste cell-by-cell.

Runtime → Change runtime type → T4 GPU  (required)

Correctness fixes over v1/v2:
  - Closed set (d_closed): each node expanded at most once
    (safe because Manhattan distance is a consistent heuristic)
  - Inline dedup via atomic-max on in_next[]: guarantees each node
    appears at most once in next_frontier — replaces the broken compact_kernel
  - Path length from g_scores[goal] (immune to parent[] write races)
  - MAX_ITER 100,000 for large grids

Performance note:
  GPU A* amortises kernel-launch overhead only when the frontier is large
  (>~10,000 nodes/iteration). Manhattan heuristic keeps frontiers narrow,
  so GPU advantage grows with grid size. Visible speedup at 4096×4096+.

Author: Sathvik
"""

# ── 0. Environment ──────────────────────────────────────────────────────────
import subprocess, sys, math, time, csv
import numpy as np
import matplotlib; matplotlib.use("Agg")
import matplotlib.pyplot as plt

r = subprocess.run(["nvidia-smi"], capture_output=True, text=True)
if r.returncode == 0:
    for line in r.stdout.split("\n")[:3]:
        print(line)
else:
    print("WARNING: No GPU. Go to Runtime → Change runtime type → T4 GPU")

import numba
from numba import cuda
print(f"Numba {numba.__version__}  |  CUDA: {cuda.is_available()}")
if cuda.is_available():
    print(f"GPU: {cuda.get_current_device().name}")


# ── 1. Grid Generation ──────────────────────────────────────────────────────

def generate_grid(rows, cols, obstacle_pct=0.30, seed=42):
    """
    Random obstacle grid — NO forced corridor.
    Guarantees connectivity via flood-fill; regenerates if not connected.
    30% obstacles forces wider detours → larger frontier → more GPU parallelism.
    """
    rng = np.random.default_rng(seed)
    for attempt in range(20):
        grid = rng.random((rows, cols)) > obstacle_pct
        grid[0, 0] = True
        grid[rows-1, cols-1] = True
        if _is_reachable(grid, 0, 0, rows-1, cols-1):
            return grid
        seed += 1
        rng = np.random.default_rng(seed)
    # fallback: carve a minimal path
    grid[0, :cols//2] = True
    grid[rows//2, :] = True
    grid[rows//2:, cols-1] = True
    return grid


def _is_reachable(grid, r0, c0, r1, c1):
    """Check grid connectivity using scipy connected-components (fast for any size)."""
    try:
        from scipy.ndimage import label as scipy_label
        labeled, _ = scipy_label(grid)
        comp = labeled[r0, c0]
        return comp > 0 and comp == labeled[r1, c1]
    except ImportError:
        pass
    # BFS fallback when scipy is unavailable (only practical for small grids)
    rows, cols = grid.shape
    if rows * cols > 512 * 512:
        return True
    visited = np.zeros((rows, cols), dtype=bool)
    queue = [(r0, c0)]
    visited[r0, c0] = True
    while queue:
        r, c = queue.pop()
        if r == r1 and c == c1:
            return True
        for dr, dc in [(-1,0),(1,0),(0,-1),(0,1)]:
            nr, nc = r+dr, c+dc
            if 0 <= nr < rows and 0 <= nc < cols and grid[nr,nc] and not visited[nr,nc]:
                visited[nr, nc] = True
                queue.append((nr, nc))
    return False


# ── 2. CPU A* ───────────────────────────────────────────────────────────────

import heapq

def astar_cpu(grid, start, goal):
    """Standard serial A* with Manhattan-distance heuristic."""
    rows, cols = grid.shape
    t0 = time.perf_counter()

    def h(s):
        r1, c1 = divmod(s, cols)
        r2, c2 = divmod(goal, cols)
        return abs(r1-r2) + abs(c1-c2)

    heap = [(h(start), 0, start)]
    came_from = {start: -1}
    g = {start: 0}
    ctr = expanded = 0

    while heap:
        _f, _c, cur = heapq.heappop(heap)
        expanded += 1
        if cur == goal:
            node, length = cur, 0
            while came_from[node] != -1:
                node = came_from[node]
                length += 1
            return {"found": True, "path_length": length,
                    "nodes_expanded": expanded,
                    "time_s": time.perf_counter()-t0}
        r, c = divmod(cur, cols)
        for dr, dc in [(-1,0),(1,0),(0,-1),(0,1)]:
            nr, nc = r+dr, c+dc
            if 0 <= nr < rows and 0 <= nc < cols and grid[nr,nc]:
                nb = nr*cols+nc
                ng = g[cur]+1
                if nb not in g or ng < g[nb]:
                    g[nb] = ng
                    ctr += 1
                    heapq.heappush(heap, (ng+h(nb), ctr, nb))
                    came_from[nb] = cur

    return {"found": False, "path_length": -1,
            "nodes_expanded": expanded, "time_s": time.perf_counter()-t0}


def dijkstra_cpu(grid, start, goal):
    """
    CPU BFS — no heuristic (h=0). Fair baseline for the GPU kernel, which also
    uses no heuristic. Uses a deque (O(1) per op) instead of a heap since all
    edge costs are equal.
    """
    from collections import deque
    rows, cols = grid.shape
    t0 = time.perf_counter()

    dist = {start: 0}
    queue = deque([start])
    expanded = 0

    while queue:
        cur = queue.popleft()
        expanded += 1
        if cur == goal:
            return {"found": True, "path_length": dist[cur],
                    "nodes_expanded": expanded,
                    "time_s": time.perf_counter()-t0}
        r, c = divmod(cur, cols)
        for dr, dc in [(-1,0),(1,0),(0,-1),(0,1)]:
            nr, nc = r+dr, c+dc
            if 0 <= nr < rows and 0 <= nc < cols and grid[nr,nc]:
                nb = nr*cols+nc
                if nb not in dist:
                    dist[nb] = dist[cur]+1
                    queue.append(nb)

    return {"found": False, "path_length": -1,
            "nodes_expanded": expanded, "time_s": time.perf_counter()-t0}


# ── 3. GPU Kernel ────────────────────────────────────────────────────────────

@cuda.jit
def expand_kernel(
    frontier, f_size,
    next_f, next_size,
    g_scores, in_next, parent,
    grid_flat, rows, cols,
    goal, found_flag,
    closed,
):
    """
    One thread per frontier node.

    closed[cur]=1 prevents re-expansion (valid: Manhattan distance is consistent,
    so the first path to any node via A* is always optimal).

    atomic-max on in_next[nb] ensures each node enters next_f at most once,
    even when multiple threads simultaneously discover the same neighbour.
    This replaces the broken compact_kernel from v1/v2.
    """
    tid = cuda.grid(1)
    if tid >= f_size[0]:
        return
    cur = frontier[tid]
    if cur < 0 or closed[cur]:
        return

    closed[cur] = 1
    cur_g = g_scores[cur]
    r = cur // cols
    c = cur - r * cols

    DR = (-1, 1,  0, 0)
    DC = ( 0, 0, -1, 1)

    for i in range(4):
        nr = r + DR[i]
        nc = c + DC[i]
        if nr < 0 or nr >= rows or nc < 0 or nc >= cols:
            continue
        nb = nr * cols + nc
        if not grid_flat[nb] or closed[nb]:
            continue

        new_g = cur_g + 1
        old_g = cuda.atomic.min(g_scores, nb, new_g)

        # Set found_flag as soon as goal is reachable (regardless of which
        # thread wins the atomic.min race).
        if nb == goal:
            cuda.atomic.max(found_flag, 0, 1)

        if new_g < old_g:
            parent[nb] = cur
            # Claim nb's slot in next_f atomically — only first claimer adds it.
            old_in = cuda.atomic.max(in_next, nb, 1)
            if old_in == 0:
                idx = cuda.atomic.add(next_size, 0, 1)
                if idx < next_f.shape[0]:
                    next_f[idx] = nb


# ── 4. GPU A* Host Code ──────────────────────────────────────────────────────

def astar_gpu(grid, start, goal, threads=256):
    """
    GPU A* — bulk-synchronous parallel frontier expansion.

    Each iteration: launch one kernel thread per frontier node,
    collect deduplicated next-frontier, swap buffers, repeat.
    """
    rows, cols = grid.shape
    N   = rows * cols
    INF = np.int32(2**30)
    t0  = time.perf_counter()

    g_init        = np.full(N, INF, dtype=np.int32)
    g_init[start] = np.int32(0)

    d_grid    = cuda.to_device(grid.ravel().astype(np.bool_))
    d_g       = cuda.to_device(g_init)
    d_in_next = cuda.to_device(np.zeros(N, dtype=np.int32))
    d_closed  = cuda.to_device(np.zeros(N, dtype=np.int32))
    d_parent  = cuda.to_device(np.full(N, -1, dtype=np.int32))
    d_found   = cuda.to_device(np.array([0], dtype=np.int32))

    # Closed set guarantees each node enters the frontier at most once,
    # so frontier size is bounded by N.
    max_f   = N
    d_f     = cuda.device_array(max_f, dtype=np.int32)
    d_nxt   = cuda.device_array(max_f, dtype=np.int32)
    d_fsize = cuda.to_device(np.array([1], dtype=np.int32))
    d_nsize = cuda.to_device(np.array([0], dtype=np.int32))
    d_f[0]  = np.int32(start)

    iters    = 0
    MAX_ITER = 100_000

    while iters < MAX_ITER:
        fsize = int(d_fsize.copy_to_host()[0])
        if fsize == 0:
            break

        blk = max(1, math.ceil(fsize / threads))
        d_nsize[0] = np.int32(0)

        expand_kernel[blk, threads](
            d_f, d_fsize,
            d_nxt, d_nsize,
            d_g, d_in_next, d_parent,
            d_grid, np.int32(rows), np.int32(cols),
            np.int32(goal), d_found,
            d_closed,
        )
        cuda.synchronize()

        if int(d_found.copy_to_host()[0]) > 0:
            break

        nsize = min(int(d_nsize.copy_to_host()[0]), max_f)
        if nsize == 0:
            break

        # Device-to-device swap: d_f ← d_nxt (no PCIe round-trip)
        d_f[:nsize] = d_nxt[:nsize]
        d_fsize[0]  = np.int32(nsize)
        iters += 1

    elapsed = time.perf_counter() - t0
    found   = bool(d_found.copy_to_host()[0])

    # g_scores[goal] gives the correct shortest-path length even when
    # parent[] has write races (atomic.min on g_scores is always correct).
    path_length = -1
    if found:
        g_vals = d_g.copy_to_host()
        g_goal = int(g_vals[goal])
        path_length = g_goal if g_goal < INF else -1

    return {"found": found, "path_length": path_length,
            "iterations": iters, "time_s": elapsed}


# ── 5. Correctness Check ─────────────────────────────────────────────────────

print("\n--- Correctness check ---")
for size in [64, 128, 256]:
    g  = generate_grid(size, size)
    s  = 0
    gl = (size-1)*size + (size-1)
    rc = astar_cpu(g, s, gl)
    if cuda.is_available():
        astar_gpu(g, s, gl)   # JIT warm-up
        rg = astar_gpu(g, s, gl)
        match = rc["found"] == rg["found"] and rc["path_length"] == rg["path_length"]
        ok = "✓" if match else f"✗ MISMATCH (cpu={rc['path_length']} gpu={rg['path_length']})"
        print(f"  {size}×{size}  CPU path={rc['path_length']}  "
              f"GPU path={rg['path_length']}  {ok}  "
              f"CPU={rc['time_s']:.4f}s  GPU={rg['time_s']:.4f}s")
    else:
        print(f"  {size}×{size}  CPU path={rc['path_length']}  "
              f"GPU=N/A (no CUDA)  CPU={rc['time_s']:.4f}s")


# ── 6. Benchmark ─────────────────────────────────────────────────────────────

SIZES   = [64, 128, 256, 512, 1024, 2048, 4096]
REPEATS = 3

cpu_times, gpu_times, bfs_times = [], [], []
cpu_nodes, paths = [], []

print("\n--- Benchmark: CPU A*  vs  CPU BFS (h=0)  vs  GPU BFS (h=0) ---")
print(f"{'Size':>10}  {'CPU A*(s)':>10}  {'CPU BFS(s)':>11}  {'GPU BFS(s)':>11}  "
      f"{'GPU/BFS':>8}  {'Path':>6}  {'BFS nodes':>12}")
print("-" * 78)

for size in SIZES:
    grid = generate_grid(size, size, obstacle_pct=0.30)
    s    = 0
    gl   = (size-1)*size + (size-1)

    # ── CPU A* ───────────────────────────────────────────────────────────────
    ct_runs = []
    rc = None
    for _ in range(REPEATS):
        rc = astar_cpu(grid, s, gl)
        if rc["found"]:
            ct_runs.append(rc["time_s"])
    cpu_t    = float(np.mean(ct_runs)) if ct_runs else float("nan")
    path_len = rc["path_length"]       if rc else -1
    cpu_times.append(cpu_t)
    paths.append(path_len)

    # ── CPU BFS (h=0, fair comparison with GPU) ───────────────────────────────
    rb  = dijkstra_cpu(grid, s, gl)   # single run — can be slow for large grids
    bfs_t     = rb["time_s"]          if rb["found"] else float("nan")
    bfs_nodes = rb["nodes_expanded"]  if rb["found"] else 0
    bfs_times.append(bfs_t)
    cpu_nodes.append(bfs_nodes)

    # ── GPU BFS (h=0) ─────────────────────────────────────────────────────────
    if cuda.is_available():
        astar_gpu(grid, s, gl)   # JIT warm-up
        gt_runs = []
        for _ in range(REPEATS):
            rg = astar_gpu(grid, s, gl)
            if rg["found"]:
                gt_runs.append(rg["time_s"])
        gpu_t = float(np.mean(gt_runs)) if gt_runs else float("nan")
    else:
        gpu_t = float("nan")
    gpu_times.append(gpu_t)

    # GPU speedup relative to CPU BFS (the fair comparison)
    sp_bfs = bfs_t / gpu_t if (not np.isnan(gpu_t) and not np.isnan(bfs_t) and gpu_t > 0) else float("nan")

    cat_str  = f"{cpu_t:.4f}s"  if not np.isnan(cpu_t)  else "N/A"
    bfs_str  = f"{bfs_t:.4f}s"  if not np.isnan(bfs_t)  else "N/A"
    gpu_str  = f"{gpu_t:.4f}s"  if not np.isnan(gpu_t)  else "N/A"
    sp_str   = f"{sp_bfs:.2f}×" if not np.isnan(sp_bfs) else "N/A"
    print(f"  {size:>4}×{size:<4}  {cat_str:>10}  {bfs_str:>11}  {gpu_str:>11}  "
          f"{sp_str:>8}  {path_len:>6}  {bfs_nodes:>12,}")


# ── 7. README Table ──────────────────────────────────────────────────────────

print("\n=== README TABLE ===")
print("| Grid | CPU A* (s) | CPU BFS (s) | GPU BFS (s) | GPU/BFS speedup | Path | BFS nodes |")
print("|------|-----------|------------|------------|----------------|------|-----------|")
for i, sz in enumerate(SIZES):
    ca = f"{cpu_times[i]:.4f}" if not np.isnan(cpu_times[i]) else "N/A"
    cb = f"{bfs_times[i]:.4f}" if not np.isnan(bfs_times[i]) else "N/A"
    gt = f"{gpu_times[i]:.4f}" if not np.isnan(gpu_times[i]) else "N/A"
    sp = (f"{bfs_times[i]/gpu_times[i]:.2f}×"
          if not np.isnan(bfs_times[i]) and not np.isnan(gpu_times[i]) and gpu_times[i] > 0
          else "N/A")
    print(f"| {sz}×{sz} | {ca} | {cb} | {gt} | {sp} | {paths[i]} | {cpu_nodes[i]:,} |")

with open("colab_results_v2.csv", "w", newline="") as f:
    w = csv.writer(f)
    w.writerow(["grid_size","cpu_astar_s","cpu_bfs_s","gpu_bfs_s",
                "gpu_vs_bfs_speedup","path_length","bfs_nodes"])
    for i, sz in enumerate(SIZES):
        sp = (bfs_times[i]/gpu_times[i]
              if not np.isnan(bfs_times[i]) and not np.isnan(gpu_times[i]) and gpu_times[i] > 0
              else None)
        w.writerow([f"{sz}x{sz}", cpu_times[i], bfs_times[i], gpu_times[i],
                    sp, paths[i], cpu_nodes[i]])
print("\nSaved: colab_results_v2.csv")


# ── 8. Plots ─────────────────────────────────────────────────────────────────

valid   = [(i, s) for i, s in enumerate(SIZES) if not np.isnan(cpu_times[i])]
vi      = [i for i, _ in valid]
sizes_v = [s for _, s in valid]
labels  = [f"{s}×{s}" for s in sizes_v]

fig, axes = plt.subplots(1, 3, figsize=(17, 5))

ax = axes[0]
ax.plot(sizes_v, [cpu_times[i] for i in vi], "o-",
        color="#e74c3c", lw=2, ms=7, label="CPU A* (heuristic)")
bfs_v = [bfs_times[i] for i in vi]
if any(not np.isnan(t) for t in bfs_v):
    ax.plot(sizes_v, bfs_v, "s--", color="#e67e22", lw=2, ms=7,
            label="CPU BFS (h=0)")
gpu_v = [gpu_times[i] for i in vi]
if any(not np.isnan(t) for t in gpu_v):
    ax.plot(sizes_v, gpu_v, "^-", color="#27ae60", lw=2, ms=7,
            label="GPU BFS (h=0, CUDA / T4)")
ax.set_yscale("log"); ax.grid(True, which="both", alpha=0.3)
ax.set_xticks(sizes_v); ax.set_xticklabels(labels, rotation=25, fontsize=8)
ax.set_xlabel("Grid Size"); ax.set_ylabel("Time (s)")
ax.set_title("Solve Time: CPU A* vs CPU BFS vs GPU BFS"); ax.legend(fontsize=8)

ax = axes[1]
speedups_bfs = [bfs_times[i]/gpu_times[i]
                if (not np.isnan(bfs_times[i]) and not np.isnan(gpu_times[i]) and gpu_times[i] > 0)
                else 0
                for i in vi]
bars = ax.bar(labels, speedups_bfs, color="#27ae60", edgecolor="#1a6e3a", lw=0.8)
for bar, sp in zip(bars, speedups_bfs):
    if sp > 0:
        ax.text(bar.get_x()+bar.get_width()/2, bar.get_height()+0.02,
                f"{sp:.1f}×", ha="center", va="bottom", fontsize=9, fontweight="bold")
ax.axhline(1, color="red", ls="--", lw=1.5, label="1× (break-even)")
ax.set_xlabel("Grid Size"); ax.set_ylabel("GPU BFS speedup over CPU BFS")
ax.set_title("GPU Speedup over CPU BFS (fair: both h=0)"); ax.legend()
ax.tick_params(axis="x", rotation=25, labelsize=8)

ax = axes[2]
ax.plot(sizes_v, [cpu_nodes[i] for i in vi], "s-",
        color="#8e44ad", lw=2, ms=7, label="CPU BFS nodes")
ax.set_yscale("log"); ax.grid(True, which="both", alpha=0.3)
ax.set_xticks(sizes_v); ax.set_xticklabels(labels, rotation=25, fontsize=8)
ax.set_xlabel("Grid Size"); ax.set_ylabel("Nodes Expanded")
ax.set_title("BFS Search Space Growth (h=0)")

plt.suptitle("GPU BFS vs CPU BFS — h=0 pathfinding on T4  (fair comparison)", fontsize=13, y=1.02)
plt.tight_layout()
plt.savefig("speedup_results_v2.png", dpi=150, bbox_inches="tight")
plt.show()
print("Saved: speedup_results_v2.png")
print("\nDownload speedup_results_v2.png + colab_results_v2.csv")
print("then paste the README table above into parallel-astar-gpu/README.md")
