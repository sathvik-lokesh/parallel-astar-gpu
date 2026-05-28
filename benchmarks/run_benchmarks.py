"""
run_benchmarks.py — CPU A* vs Fast Downward benchmarks.

Two benchmark suites:
  1. Grid pathfinding — CPU A* across 64x64 to 1024x1024 grids
     (This is the domain used for GPU comparison on Colab)
  2. Blocksworld PDDL — CPU A* vs Fast Downward on symbolic planning tasks

Usage:
    python3 benchmarks/run_benchmarks.py
"""

import sys
import os
import csv
import json
import time

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from cpu.astar_baseline import (
    astar_grid,
    generate_grid,
    grid_start,
    grid_goal,
    run_fast_downward,
)

DOMAIN_PATH  = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "domains", "blocksworld_domain.pddl"))
PROBLEMS_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "domains"))
RESULTS_DIR  = os.path.abspath(os.path.join(os.path.dirname(__file__), "results"))

GRID_SIZES    = [64, 128, 256, 512, 1024]
BLOCK_SIZES   = [4, 6, 8, 10, 12]
OBSTACLE_PCT  = 0.20
REPEATS       = 3


# ---------------------------------------------------------------------------
# Suite 1: Grid A* (CPU)
# ---------------------------------------------------------------------------

def benchmark_grid_astar(size: int, repeats: int = REPEATS) -> dict:
    """Benchmark CPU A* on a size×size grid, average over repeats."""
    grid  = generate_grid(size, size, OBSTACLE_PCT, seed=42)
    start = grid_start(size, size)
    goal  = grid_goal(size, size)

    times = []
    result = None
    for _ in range(repeats):
        result = astar_grid(grid, start, goal)
        if result["found"]:
            times.append(result["time_s"])

    return {
        "solver":          "CPU A*",
        "domain":          "grid",
        "size":            size,
        "found":           result["found"],
        "path_length":     result["path_length"],
        "nodes_expanded":  result["nodes_expanded"],
        "nodes_generated": result["nodes_generated"],
        "time_mean_s":     float(np.mean(times))   if times else -1,
        "time_std_s":      float(np.std(times))    if times else  0,
        "time_min_s":      float(np.min(times))    if times else -1,
    }


# ---------------------------------------------------------------------------
# Suite 2: Blocksworld PDDL (Fast Downward)
# ---------------------------------------------------------------------------

def benchmark_fast_downward(n_blocks: int) -> dict:
    """Benchmark Fast Downward on a Blocksworld PDDL problem."""
    problem_path = os.path.join(PROBLEMS_DIR, f"blocksworld_p{n_blocks:02d}.pddl")

    if not os.path.exists(problem_path):
        return {
            "solver": "Fast Downward", "domain": "blocksworld", "size": n_blocks,
            "found": False, "plan_length": -1, "nodes_expanded": -1,
            "nodes_generated": -1, "time_mean_s": -1, "time_std_s": 0,
            "time_min_s": -1, "error": "Problem file not found",
        }

    times = []
    result = None
    for _ in range(REPEATS):
        result = run_fast_downward(DOMAIN_PATH, problem_path)
        if result["found"]:
            times.append(result["time_s"])

    return {
        "solver":          "Fast Downward",
        "domain":          "blocksworld",
        "size":            n_blocks,
        "found":           result["found"],
        "plan_length":     result["plan_length"],
        "nodes_expanded":  -1,
        "nodes_generated": -1,
        "time_mean_s":     float(np.mean(times)) if times else -1,
        "time_std_s":      float(np.std(times))  if times else  0,
        "time_min_s":      float(np.min(times))  if times else -1,
    }


# ---------------------------------------------------------------------------
# Plotting
# ---------------------------------------------------------------------------

def plot_grid_results(rows: list[dict]):
    """Plot CPU A* grid benchmark — time and nodes expanded vs grid size."""
    os.makedirs(RESULTS_DIR, exist_ok=True)

    valid = [r for r in rows if r["domain"] == "grid" and r["found"]]
    if not valid:
        return

    sizes  = [r["size"] for r in valid]
    times  = [r["time_mean_s"] for r in valid]
    stds   = [r["time_std_s"]  for r in valid]
    nodes  = [r["nodes_expanded"] for r in valid]
    cells  = [s * s for s in sizes]

    # --- Time vs grid size ---
    fig, ax = plt.subplots(figsize=(9, 5))
    ax.errorbar(sizes, times, yerr=stds, marker="o", linewidth=2,
                capsize=4, color="#e74c3c", label="CPU A* (Python)")
    ax.set_xlabel("Grid Size (N×N)", fontsize=13)
    ax.set_ylabel("Time (seconds)", fontsize=13)
    ax.set_title("CPU A* — Grid Pathfinding Solve Time\n"
                 "(GPU comparison will be added from Colab results)", fontsize=13)
    ax.set_xticks(sizes)
    ax.set_xticklabels([f"{s}×{s}" for s in sizes], rotation=15)
    ax.set_yscale("log")
    ax.grid(True, which="both", alpha=0.3)
    ax.legend(fontsize=11)
    plt.tight_layout()
    plt.savefig(os.path.join(RESULTS_DIR, "cpu_grid_time.png"), dpi=150)
    plt.close()
    print("  Saved: results/cpu_grid_time.png")

    # --- Nodes expanded vs grid cells ---
    fig, ax = plt.subplots(figsize=(8, 5))
    ax.plot(cells, nodes, marker="o", linewidth=2, color="#8e44ad")
    ax.set_xlabel("Grid Cells (N²)", fontsize=13)
    ax.set_ylabel("Nodes Expanded", fontsize=13)
    ax.set_title("CPU A* — Search Space Growth vs Grid Size", fontsize=13)
    ax.set_yscale("log")
    ax.set_xscale("log")
    ax.grid(True, which="both", alpha=0.3)
    # Annotate points
    for s, c, n in zip(sizes, cells, nodes):
        ax.annotate(f"{s}×{s}", (c, n), textcoords="offset points",
                    xytext=(5, 5), fontsize=9)
    plt.tight_layout()
    plt.savefig(os.path.join(RESULTS_DIR, "cpu_nodes_expanded.png"), dpi=150)
    plt.close()
    print("  Saved: results/cpu_nodes_expanded.png")


def plot_fd_results(rows: list[dict]):
    """Plot Fast Downward Blocksworld benchmark."""
    os.makedirs(RESULTS_DIR, exist_ok=True)

    valid = [r for r in rows if r["domain"] == "blocksworld" and r["found"]]
    if not valid:
        return

    sizes = [r["size"] for r in valid]
    times = [r["time_mean_s"] for r in valid]
    stds  = [r["time_std_s"] for r in valid]

    fig, ax = plt.subplots(figsize=(8, 5))
    ax.errorbar(sizes, times, yerr=stds, marker="s", linewidth=2,
                capsize=4, color="#2980b9", label="Fast Downward (A*+FF)")
    ax.set_xlabel("Number of Blocks", fontsize=13)
    ax.set_ylabel("Time (seconds)", fontsize=13)
    ax.set_title("Fast Downward — Blocksworld Solve Time", fontsize=13)
    ax.set_xticks(sizes)
    ax.set_yscale("log")
    ax.grid(True, which="both", alpha=0.3)
    ax.legend(fontsize=11)
    plt.tight_layout()
    plt.savefig(os.path.join(RESULTS_DIR, "fd_blocksworld_time.png"), dpi=150)
    plt.close()
    print("  Saved: results/fd_blocksworld_time.png")


def plot_speedup_placeholder(cpu_rows: list[dict]):
    """
    Placeholder speedup chart — GPU times will be filled in from Colab results.
    Shows CPU baseline and leaves GPU line empty for now.
    """
    os.makedirs(RESULTS_DIR, exist_ok=True)

    valid = [r for r in cpu_rows if r["found"]]
    if not valid:
        return

    sizes = [r["size"] for r in valid]
    times = [r["time_mean_s"] for r in valid]

    fig, ax = plt.subplots(figsize=(9, 5))
    ax.plot(sizes, times, marker="o", linewidth=2, color="#e74c3c",
            label="CPU A* (Python)", linestyle="--")
    ax.plot([], [], marker="^", linewidth=2, color="#27ae60",
            label="GPU A* (CUDA — Colab results pending)")

    ax.set_xlabel("Grid Size (N×N)", fontsize=13)
    ax.set_ylabel("Time (seconds)", fontsize=13)
    ax.set_title("CPU vs GPU A* — Grid Pathfinding Speedup\n"
                 "(GPU results to be added from Colab)", fontsize=13)
    ax.set_xticks(sizes)
    ax.set_xticklabels([f"{s}×{s}" for s in sizes], rotation=15)
    ax.set_yscale("log")
    ax.grid(True, which="both", alpha=0.3)
    ax.legend(fontsize=11)
    plt.tight_layout()
    plt.savefig(os.path.join(RESULTS_DIR, "speedup_cpu_vs_gpu.png"), dpi=150)
    plt.close()
    print("  Saved: results/speedup_cpu_vs_gpu.png  (GPU times pending from Colab)")


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    os.makedirs(RESULTS_DIR, exist_ok=True)

    # ---- Suite 1: Grid A* ------------------------------------------------
    print("\n" + "=" * 65)
    print("  Suite 1 — Grid Pathfinding (CPU A*)")
    print("=" * 65)
    print(f"  {'Grid':>8}  {'Path':>6}  {'Expanded':>12}  {'Time (mean)':>12}  {'Std':>8}")
    print("  " + "-" * 55)

    grid_results = []
    for size in GRID_SIZES:
        print(f"  {size}×{size} ...", end="  ", flush=True)
        r = benchmark_grid_astar(size)
        grid_results.append(r)
        if r["found"]:
            print(f"{r['path_length']:>6}  {r['nodes_expanded']:>12,}  "
                  f"{r['time_mean_s']:>11.4f}s  {r['time_std_s']:>7.4f}s")
        else:
            print("NO PATH")

    # ---- Suite 2: Fast Downward Blocksworld ------------------------------
    print("\n" + "=" * 65)
    print("  Suite 2 — Blocksworld PDDL (Fast Downward)")
    print("=" * 65)

    # Generate PDDL problems
    gen_script = os.path.join(PROBLEMS_DIR, "generate_problems.py")
    os.system(f"python3 {gen_script}")
    print()

    fd_results = []
    for n in BLOCK_SIZES:
        print(f"  Blocks={n} ...", end="  ", flush=True)
        r = benchmark_fast_downward(n)
        fd_results.append(r)
        if r["found"]:
            print(f"plan={r['plan_length']}  time={r['time_mean_s']:.4f}s ± {r['time_std_s']:.4f}s")
        else:
            print("NOT FOUND / ERROR")

    # ---- Save results ----------------------------------------------------
    all_results = grid_results + fd_results

    csv_path = os.path.join(RESULTS_DIR, "benchmark_results.csv")
    fieldnames = ["solver", "domain", "size", "found", "plan_length",
                  "nodes_expanded", "nodes_generated",
                  "time_mean_s", "time_std_s", "time_min_s"]
    with open(csv_path, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames, extrasaction="ignore")
        writer.writeheader()
        writer.writerows(all_results)
    print(f"\n  Saved CSV: {csv_path}")

    json_path = os.path.join(RESULTS_DIR, "benchmark_results.json")
    with open(json_path, "w") as f:
        json.dump(all_results, f, indent=2)
    print(f"  Saved JSON: {json_path}")

    # ---- Plots -----------------------------------------------------------
    print("\nGenerating plots...")
    plot_grid_results(grid_results)
    plot_fd_results(fd_results)
    plot_speedup_placeholder(grid_results)

    # ---- Summary ---------------------------------------------------------
    print("\n" + "=" * 65)
    print("  SUMMARY")
    print("=" * 65)
    print("\n  Grid Pathfinding (CPU A*):")
    for r in grid_results:
        if r["found"]:
            print(f"    {r['size']:4}×{r['size']:<4}  {r['time_mean_s']:.4f}s  "
                  f"expanded={r['nodes_expanded']:,}")
    print("\n  Blocksworld (Fast Downward):")
    for r in fd_results:
        if r["found"]:
            print(f"    {r['size']:2} blocks    {r['time_mean_s']:.4f}s  "
                  f"plan={r['plan_length']} moves")
    print()


if __name__ == "__main__":
    main()
