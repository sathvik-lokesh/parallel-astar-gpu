"""
astar_baseline.py — CPU A* baseline for grid pathfinding.

Implements A* on 2D grids with random obstacles.
Grid sizes scale from 64x64 to 1024x1024 — giving a clean, predictable
benchmark that exposes how CPU performance degrades as search space grows.

State:   integer (row * cols + col)
Heuristic: Manhattan distance (admissible, fast)

This is the CPU version to be ported to GPU CUDA (on Colab).
"""

import heapq
import time
import random
import numpy as np
from typing import Optional


# ---------------------------------------------------------------------------
# Grid generation
# ---------------------------------------------------------------------------

def generate_grid(rows: int, cols: int, obstacle_pct: float = 0.20,
                  seed: int = 42) -> np.ndarray:
    """
    Generate a 2D grid with random obstacles.

    Args:
        rows, cols: Grid dimensions
        obstacle_pct: Fraction of cells that are walls (0 = free, 1 = wall)
        seed: RNG seed for reproducibility

    Returns:
        numpy bool array (True = free, False = wall)
    """
    rng = np.random.default_rng(seed)
    grid = rng.random((rows, cols)) > obstacle_pct

    # Always keep start (0,0) and goal (rows-1, cols-1) clear
    grid[0, 0] = True
    grid[rows - 1, cols - 1] = True

    # Carve a guaranteed path so problem is always solvable
    # Simple corridor: right along top row, then down last column
    grid[0, :] = True
    grid[:, cols - 1] = True

    return grid


def grid_start(rows: int, cols: int) -> int:
    """Start state: top-left cell encoded as integer."""
    return 0  # row=0, col=0


def grid_goal(rows: int, cols: int) -> int:
    """Goal state: bottom-right cell encoded as integer."""
    return (rows - 1) * cols + (cols - 1)


# ---------------------------------------------------------------------------
# Successors & heuristic (CPU)
# ---------------------------------------------------------------------------

def get_successors_grid(state: int, grid: np.ndarray, cols: int) -> list[int]:
    """
    4-connected neighbors (up/down/left/right) that are free cells.

    Args:
        state: Current cell index
        grid: Boolean grid (True = free)
        cols: Number of columns

    Returns:
        List of reachable neighbor cell indices
    """
    rows = grid.shape[0]
    r, c = divmod(state, cols)
    neighbors = []

    for dr, dc in [(-1, 0), (1, 0), (0, -1), (0, 1)]:
        nr, nc = r + dr, c + dc
        if 0 <= nr < rows and 0 <= nc < cols and grid[nr, nc]:
            neighbors.append(nr * cols + nc)

    return neighbors


def manhattan(state: int, goal: int, cols: int) -> int:
    """
    Manhattan distance heuristic (admissible for 4-connected grid).

    Args:
        state: Current cell index
        goal: Goal cell index
        cols: Number of columns

    Returns:
        Manhattan distance
    """
    r1, c1 = divmod(state, cols)
    r2, c2 = divmod(goal, cols)
    return abs(r1 - r2) + abs(c1 - c2)


# ---------------------------------------------------------------------------
# A* Search (CPU — pure Python)
# ---------------------------------------------------------------------------

def astar_grid(grid: np.ndarray, start: int, goal: int) -> dict:
    """
    A* search on a 2D grid.

    Args:
        grid: Boolean free/wall grid
        start: Start cell index
        goal: Goal cell index

    Returns:
        Dict: found, path_length, nodes_expanded, nodes_generated, time_s, path
    """
    rows, cols = grid.shape
    start_time = time.perf_counter()

    if start == goal:
        return {"found": True, "path_length": 0, "nodes_expanded": 0,
                "nodes_generated": 1, "time_s": 0.0, "path": [start]}

    counter = 0
    open_heap = []
    h0 = manhattan(start, goal, cols)
    heapq.heappush(open_heap, (h0, counter, start))

    came_from = {start: -1}
    g_score = {start: 0}

    nodes_expanded = 0
    nodes_generated = 1

    while open_heap:
        _f, _, current = heapq.heappop(open_heap)
        nodes_expanded += 1

        if current == goal:
            # Reconstruct path
            path = []
            node = current
            while node != -1:
                path.append(node)
                node = came_from[node]
            path.reverse()
            elapsed = time.perf_counter() - start_time
            return {
                "found": True,
                "path_length": len(path) - 1,
                "nodes_expanded": nodes_expanded,
                "nodes_generated": nodes_generated,
                "time_s": elapsed,
                "path": path,
            }

        current_g = g_score[current]

        for nb in get_successors_grid(current, grid, cols):
            tentative_g = current_g + 1
            if nb not in g_score or tentative_g < g_score[nb]:
                g_score[nb] = tentative_g
                f = tentative_g + manhattan(nb, goal, cols)
                came_from[nb] = current
                counter += 1
                heapq.heappush(open_heap, (f, counter, nb))
                nodes_generated += 1

    elapsed = time.perf_counter() - start_time
    return {
        "found": False, "path_length": -1,
        "nodes_expanded": nodes_expanded, "nodes_generated": nodes_generated,
        "time_s": elapsed, "path": [],
    }


# ---------------------------------------------------------------------------
# Fast Downward wrapper (Blocksworld — symbolic planning comparison)
# ---------------------------------------------------------------------------

import subprocess
import re
import tempfile
import os

FD_PATH = "/home/sathv/fast_downward/fast-downward.py"


def run_fast_downward(domain_path: str, problem_path: str, timeout: int = 120) -> dict:
    """
    Run Fast Downward (A* + FF heuristic) on a PDDL problem.

    Args:
        domain_path: Path to PDDL domain file
        problem_path: Path to PDDL problem file
        timeout: Max seconds

    Returns:
        Dict: found, plan_length, time_s, raw_output
    """
    start = time.perf_counter()

    with tempfile.TemporaryDirectory() as tmpdir:
        plan_path = os.path.join(tmpdir, "plan")
        cmd = [
            "python3", FD_PATH,
            "--overall-time-limit", str(timeout),
            "--plan-file", plan_path,
            domain_path, problem_path,
            "--search", "astar(ff())",
        ]

        try:
            result = subprocess.run(cmd, capture_output=True, text=True,
                                    timeout=timeout + 5, cwd=tmpdir)
            elapsed = time.perf_counter() - start
            output = result.stdout + result.stderr

            plan_length = -1
            plan_found = False

            if os.path.exists(plan_path):
                with open(plan_path) as f:
                    lines = [l.strip() for l in f if not l.startswith(";") and l.strip()]
                    plan_length = len(lines)
                    plan_found = True
            elif "Solution found" in output:
                plan_found = True
                m = re.search(r"Plan length: (\d+)", output)
                if m:
                    plan_length = int(m.group(1))

            return {"found": plan_found, "plan_length": plan_length,
                    "time_s": elapsed, "raw_output": output[-2000:]}

        except subprocess.TimeoutExpired:
            return {"found": False, "plan_length": -1,
                    "time_s": time.perf_counter() - start, "raw_output": "TIMEOUT"}
        except Exception as e:
            return {"found": False, "plan_length": -1,
                    "time_s": time.perf_counter() - start, "raw_output": str(e)}


# ---------------------------------------------------------------------------
# Self-test
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    print("=== CPU A* Grid Pathfinding Baseline ===\n")
    print(f"{'Size':>10} {'Nodes Exp':>12} {'Nodes Gen':>12} {'Path':>6} {'Time(s)':>10}")
    print("-" * 55)

    for size in [64, 128, 256, 512, 1024]:
        grid = generate_grid(size, size, obstacle_pct=0.20, seed=42)
        start = grid_start(size, size)
        goal  = grid_goal(size, size)

        result = astar_grid(grid, start, goal)

        if result["found"]:
            print(f"{size:>4}x{size:<4}  "
                  f"{result['nodes_expanded']:>12,}  "
                  f"{result['nodes_generated']:>12,}  "
                  f"{result['path_length']:>6}  "
                  f"{result['time_s']:>10.4f}s")
        else:
            print(f"{size:>4}x{size:<4}  NO PATH FOUND")
