"""
generate_problems.py — Generate hard Blocksworld PDDL problems of varying sizes.

Uses random scrambled multi-stack configurations for initial and goal states
so the search space is large enough to meaningfully benchmark solvers.
"""

import random
import os


def random_stacks(blocks: list, seed: int) -> list[list]:
    """Randomly distribute blocks into multiple stacks."""
    rng = random.Random(seed)
    shuffled = list(blocks)
    rng.shuffle(shuffled)

    # Decide number of stacks: between 2 and max(2, n//2)
    n = len(blocks)
    n_stacks = rng.randint(2, max(2, n // 2))

    stacks = [[] for _ in range(n_stacks)]
    for i, block in enumerate(shuffled):
        stacks[i % n_stacks].append(block)

    return [s for s in stacks if s]


def generate_blocksworld_problem(n_blocks: int, seed: int = 42) -> str:
    """
    Generate a hard Blocksworld PDDL problem with n_blocks blocks.
    Initial and goal states are random multi-stack configurations.
    """
    blocks = [f"b{i}" for i in range(1, n_blocks + 1)]

    # Different seeds for init and goal so they differ
    init_stacks = random_stacks(blocks, seed=seed)
    goal_stacks = random_stacks(blocks, seed=seed + 1000)

    def stacks_to_pddl(stacks):
        lines = []
        for stack in stacks:
            lines.append(f"    (ontable {stack[0]})")
            for i in range(1, len(stack)):
                lines.append(f"    (on {stack[i]} {stack[i-1]})")
            lines.append(f"    (clear {stack[-1]})")
        lines.append("    (handempty)")
        return lines

    init_lines = stacks_to_pddl(init_stacks)
    goal_lines = stacks_to_pddl(goal_stacks)[:-1]  # drop handempty from goal

    pddl = f"""(define (problem blocksworld-{n_blocks})
  (:domain blocksworld)
  (:objects {" ".join(blocks)})
  (:init
{chr(10).join(init_lines)}
  )
  (:goal
    (and
{chr(10).join(goal_lines)}
    )
  )
)
"""
    return pddl


if __name__ == "__main__":
    out_dir = os.path.dirname(os.path.abspath(__file__))
    sizes = [4, 6, 8, 10, 12]

    for n in sizes:
        content = generate_blocksworld_problem(n)
        fname = os.path.join(out_dir, f"blocksworld_p{n:02d}.pddl")
        with open(fname, "w") as f:
            f.write(content)
        print(f"Generated: {fname}")

    print("Done.")
