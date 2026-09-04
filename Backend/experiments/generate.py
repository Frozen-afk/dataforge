import random
from typing import Any, Dict, List

from core.graph import (
    Graph,
    make_branch,
    make_cycle,
    make_diamond,
    make_disconnected,
    make_line,
)


def random_graph(rng: random.Random, n: int, density: float) -> Graph:
    """
    Generates a random directed graph.

    Edge u -> v is added with probability density.
    Self-loops are skipped.
    """
    edges = []

    for u in range(n):
        for v in range(n):
            if u == v:
                continue

            if rng.random() < density:
                edges.append((u, v))

    return Graph(n=n, edges=edges)


def random_case(
    rng: random.Random,
    n: int | None = None,
    density: float | None = None,
    max_R: int = 12,
) -> Dict[str, Any]:
    """
    Generates one random test case.
    """
    if n is None:
        n = rng.choice([8, 12, 16, 24])

    if density is None:
        density = rng.choice([0.08, 0.15, 0.25])

    graph = random_graph(rng, n=n, density=density)

    source = rng.randrange(n)
    target = rng.randrange(n)
    R = rng.randint(0, max_R)

    return {
        "graph": graph,
        "source": source,
        "target": target,
        "R": R,
    }


def generate_cases(num_cases: int = 100, seed: int = 0) -> List[Dict[str, Any]]:
    """
    Generates test cases for the exact invariant.

    Includes:
        - all line graphs of length 1..12
        - all R values 0..12 for those lines
        - fixed preset graphs
        - random graphs
    """
    rng = random.Random(seed)
    cases: List[Dict[str, Any]] = []

    # Deterministic line graphs.
    for length in range(1, 13):
        graph, source, target = make_line(length)

        for R in range(0, 13):
            cases.append(
                {
                    "graph": graph,
                    "source": source,
                    "target": target,
                    "R": R,
                }
            )

    # Deterministic preset graphs.
    preset_factories = [
        make_branch,
        make_diamond,
        make_cycle,
        make_disconnected,
    ]

    for factory in preset_factories:
        graph, source, target = factory()

        for R in range(0, 13):
            cases.append(
                {
                    "graph": graph,
                    "source": source,
                    "target": target,
                    "R": R,
                }
            )

    # Random graphs.
    while len(cases) < num_cases:
        cases.append(random_case(rng))

    return cases[:num_cases]
