"""
astar_cuda.py — GPU Parallel A* via CUDA (Numba).

Mirrors colab_astar.py for local reference.
Run on Google Colab with T4 GPU — not locally (no GPU available in WSL).

Algorithm: Bulk-Synchronous Parallel A*
  Each iteration expands all frontier nodes in parallel (one CUDA thread per node).
  Atomic-min on g_scores[] ensures thread-safe best-path tracking.
  Closed set prevents re-expansion (valid: Manhattan distance is consistent).
  Atomic-max on in_next[] deduplicates next_frontier inline — no separate pass needed.

Usage (on Colab):
    from gpu.astar_cuda import astar_gpu, generate_grid
    grid = generate_grid(512, 512)
    result = astar_gpu(grid, start=0, goal=512*512-1)
"""

import math
import time
import numpy as np

try:
    from numba import cuda
    CUDA_AVAILABLE = cuda.is_available()
except ImportError:
    CUDA_AVAILABLE = False
    print("Warning: Numba not installed. Run on Google Colab for GPU support.")

import sys
import os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))
from cpu.astar_baseline import generate_grid, grid_start, grid_goal


# ---------------------------------------------------------------------------
# CUDA Kernel
# ---------------------------------------------------------------------------

if CUDA_AVAILABLE:
    from numba import cuda as _cuda

    @_cuda.jit
    def expand_frontier_kernel(
        frontier, frontier_size,
        g_scores, parent,
        grid_flat, rows, cols,
        goal,
        next_frontier, next_size,
        found_flag,
        closed,   # int32[N]: 1 = already expanded, never re-expand
        in_next,  # int32[N]: 1 = already queued in next_frontier
    ):
        """
        Expand one frontier node per thread.

        closed[]: prevents re-expansion. Manhattan distance is a consistent
        heuristic for 4-connected uniform-cost grids, so the first path to any
        node via A* is always optimal — closing on first expansion is safe.

        atomic-max on in_next[nb]: ensures each node appears at most once in
        next_frontier, even when multiple threads simultaneously discover the
        same neighbour through different paths with the same cost.
        """
        tid = _cuda.grid(1)
        if tid >= frontier_size[0]:
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
            old_g = _cuda.atomic.min(g_scores, nb, new_g)

            # Set found_flag as soon as goal is reachable — whichever thread
            # first discovers goal as a neighbour wins the race.
            if nb == goal:
                _cuda.atomic.max(found_flag, 0, 1)

            if new_g < old_g:
                parent[nb] = cur
                # Claim this node's slot in next_frontier atomically.
                # Only the first thread to claim it actually appends it.
                old_in = _cuda.atomic.max(in_next, nb, 1)
                if old_in == 0:
                    idx = _cuda.atomic.add(next_size, 0, 1)
                    if idx < next_frontier.shape[0]:
                        next_frontier[idx] = nb


# ---------------------------------------------------------------------------
# Host function: GPU A*
# ---------------------------------------------------------------------------

def astar_gpu(grid: np.ndarray, start: int, goal: int,
              threads_per_block: int = 256) -> dict:
    """
    GPU A* — bulk-synchronous parallel frontier expansion.

    Args:
        grid:              Boolean numpy array (rows x cols), True=free cell
        start:             Start cell index (row * cols + col)
        goal:              Goal cell index
        threads_per_block: CUDA thread block size (default 256)

    Returns:
        dict with keys: found, path_length, iterations, time_s
    """
    if not CUDA_AVAILABLE:
        raise RuntimeError("CUDA not available. Run on Google Colab with T4 GPU.")

    rows, cols = grid.shape
    N = rows * cols
    INF = np.int32(2**30)

    if start == goal:
        return {"found": True, "path_length": 0, "iterations": 0, "time_s": 0.0}

    t0 = time.perf_counter()

    g_init        = np.full(N, INF, dtype=np.int32)
    g_init[start] = np.int32(0)

    d_grid    = cuda.to_device(grid.ravel().astype(np.bool_))
    d_g_scores = cuda.to_device(g_init)
    d_parent   = cuda.to_device(np.full(N, -1, dtype=np.int32))
    d_closed   = cuda.to_device(np.zeros(N, dtype=np.int32))
    d_in_next  = cuda.to_device(np.zeros(N, dtype=np.int32))
    d_found_flag = cuda.to_device(np.array([0], dtype=np.int32))

    # With closed set, each node enters the frontier at most once → bounded by N.
    max_frontier = N
    d_frontier      = cuda.device_array(max_frontier, dtype=np.int32)
    d_next          = cuda.device_array(max_frontier, dtype=np.int32)
    d_frontier_size = cuda.to_device(np.array([1], dtype=np.int32))
    d_next_size     = cuda.to_device(np.array([0], dtype=np.int32))
    d_frontier[0]   = np.int32(start)

    iterations = 0
    MAX_ITER   = 100_000

    while iterations < MAX_ITER:
        fsize = int(d_frontier_size.copy_to_host()[0])
        if fsize == 0:
            break

        blocks = max(1, math.ceil(fsize / threads_per_block))
        d_next_size[0] = np.int32(0)

        expand_frontier_kernel[blocks, threads_per_block](
            d_frontier, d_frontier_size,
            d_g_scores, d_parent,
            d_grid, np.int32(rows), np.int32(cols), np.int32(goal),
            d_next, d_next_size,
            d_found_flag,
            d_closed, d_in_next,
        )
        cuda.synchronize()

        if int(d_found_flag.copy_to_host()[0]) > 0:
            break

        nsize = min(int(d_next_size.copy_to_host()[0]), max_frontier)
        if nsize == 0:
            break

        # Device-to-device swap (no PCIe round-trip through host)
        d_frontier[:nsize] = d_next[:nsize]
        d_frontier_size[0] = np.int32(nsize)

        iterations += 1

    elapsed = time.perf_counter() - t0
    found = bool(d_found_flag.copy_to_host()[0])

    # g_scores[goal] gives the correct shortest-path distance even when
    # parent[] has write races between threads (atomic.min on g_scores is always correct).
    path_length = -1
    if found:
        g_vals = d_g_scores.copy_to_host()
        g_goal = int(g_vals[goal])
        path_length = g_goal if g_goal < INF else -1

    return {
        "found":       found,
        "path_length": path_length,
        "iterations":  iterations,
        "time_s":      elapsed,
    }


# ---------------------------------------------------------------------------
# Standalone benchmark (run on Colab)
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    if not CUDA_AVAILABLE:
        print("No CUDA available. Upload colab_astar.py to Google Colab (T4 GPU).")
        print("Running CPU baseline instead...\n")
        from cpu.astar_baseline import astar_grid
        for size in [64, 128, 256, 512, 1024]:
            grid = generate_grid(size, size)
            r = astar_grid(grid, grid_start(size, size), grid_goal(size, size))
            print(f"  {size}×{size}  CPU: {r['time_s']:.4f}s  "
                  f"expanded={r['nodes_expanded']:,}")
    else:
        print(f"GPU: {cuda.get_current_device().name}")
        print("\nRunning GPU A* benchmark...\n")
        print(f"{'Size':>10}  {'GPU time':>10}  {'Path':>6}  {'Iter':>6}")
        print("-" * 40)
        for size in [64, 128, 256, 512, 1024]:
            grid = generate_grid(size, size)
            astar_gpu(grid, grid_start(size, size), grid_goal(size, size))  # JIT warm-up
            r = astar_gpu(grid, grid_start(size, size), grid_goal(size, size))
            print(f"  {size}×{size}    {r['time_s']:>9.4f}s  "
                  f"{r['path_length']:>6}  {r['iterations']:>6}")
