"""
Case generation for exact-solver validation.

Architecture section 10.1 requires the suite to be stratified by node count
n in {8, 12, 16, 24}, by shortest-path distance d in {1, ..., 10, infinity},
and across several edge-density regimes, evaluated at R in {0, ..., 12}, with
at least 10,000 seeded cases.

The invariant under test is

    h_q^(R) > epsilon   <=>   d(s, q) <= R

Stratification matters because a purely random generator produces mostly
unreachable or trivially-close pairs at these sizes, leaving the interesting
band -- where R sits just below or just above the true distance -- barely
sampled. Building cases at a *requested* distance and then sweeping R across
that distance puts most of the test budget on the boundary the invariant is
actually about.
"""

from __future__ import annotations

import random
from typing import Any, Dict, List

from core.generator import DENSITIES, NODE_COUNTS, make_case, sample_target_in_degree
from core.graph import (
    Graph,
    make_branch,
    make_cycle,
    make_diamond,
    make_disconnected,
    make_line,
)

MAX_R = 12
MAX_DISTANCE = 10


def random_graph(rng: random.Random, n: int, density: float) -> Graph:
    """Erdos-Renyi directed graph. Retained for density-regime coverage."""
    edges = [
        (u, v)
        for u in range(n)
        for v in range(n)
        if u != v and rng.random() < density
    ]
    return Graph(n=n, edges=edges)


def random_case(
    rng: random.Random,
    n: int | None = None,
    density: float | None = None,
    max_R: int = MAX_R,
) -> Dict[str, Any]:
    """One unstratified random case, used to cover the uncontrolled regime."""
    if n is None:
        n = rng.choice(list(NODE_COUNTS))
    if density is None:
        density = rng.choice(list(DENSITIES))

    graph = random_graph(rng, n=n, density=density)

    return {
        "graph": graph,
        "source": rng.randrange(n),
        "target": rng.randrange(n),
        "R": rng.randint(0, max_R),
        "stratum": "random",
        "n": n,
        "density": density,
    }


def deterministic_cases() -> List[Dict[str, Any]]:
    """
    Hand-checkable cases, swept across every R.

    These are the cases a reader can verify by eye, so a regression in the
    recurrence shows up here before it shows up in the random strata.
    """
    cases: List[Dict[str, Any]] = []

    for length in range(1, MAX_R + 1):
        graph, source, target = make_line(length)
        for R in range(0, MAX_R + 1):
            cases.append({
                "graph": graph, "source": source, "target": target, "R": R,
                "stratum": f"line-{length}", "n": graph.n, "density": None,
            })

    for name, factory in (
        ("branch", make_branch),
        ("diamond", make_diamond),
        ("cycle", make_cycle),
        ("disconnected", make_disconnected),
    ):
        graph, source, target = factory()
        for R in range(0, MAX_R + 1):
            cases.append({
                "graph": graph, "source": source, "target": target, "R": R,
                "stratum": name, "n": graph.n, "density": None,
            })

    return cases


def stratified_case_sweep(
    rng: random.Random,
    n: int,
    density: float,
    distance: int | None,
) -> List[Dict[str, Any]]:
    """
    Build one graph at a requested distance and sweep R across the boundary.

    For a reachable pair at distance d the sweep covers d - 2 through d + 2
    (clipped to the legal range), which is where the invariant can actually
    fail. For an unreachable pair every R is equally informative, so a spread
    across the whole range is used.
    """
    case = make_case(
        rng,
        n=n,
        distance=distance,
        density=density,
        target_in_degree=sample_target_in_degree(rng),
    )
    if case is None:
        return []

    graph = case["graph"]
    source = case["source"]
    target = case["target"]
    actual = case["distance"]

    if actual is None:
        depths = sorted({0, 1, rng.randint(2, MAX_R - 1), MAX_R})
    else:
        depths = sorted({
            max(0, min(MAX_R, actual + offset))
            for offset in (-2, -1, 0, 1, 2)
        })

    stratum = f"d={actual if actual is not None else 'inf'}"

    return [
        {
            "graph": graph, "source": source, "target": target, "R": R,
            "stratum": stratum, "n": n, "density": density,
            "distance": actual,
        }
        for R in depths
    ]


def generate_cases(num_cases: int = 10000, seed: int = 0) -> List[Dict[str, Any]]:
    """
    Assemble the validation suite.

    Composition, in order: the deterministic hand-checkable cases, then the
    distance-stratified sweep cycling through every (n, density, distance)
    combination, then unstratified random cases to fill any remainder.
    """
    rng = random.Random(seed)
    cases: List[Dict[str, Any]] = deterministic_cases()

    distances: List[int | None] = list(range(1, MAX_DISTANCE + 1)) + [None]
    grid = [
        (n, density, distance)
        for n in NODE_COUNTS
        for density in DENSITIES
        for distance in distances
        if distance is None or distance <= n - 1
    ]

    index = 0
    guard = 0
    guard_limit = num_cases * 10 + len(grid) * 4

    while len(cases) < num_cases and guard < guard_limit:
        guard += 1
        n, density, distance = grid[index % len(grid)]
        index += 1
        cases.extend(stratified_case_sweep(rng, n, density, distance))

    # Top up with uncontrolled random cases if the grid ran short.
    while len(cases) < num_cases:
        cases.append(random_case(rng))

    return cases[:num_cases]
