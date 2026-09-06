"""
Distance-stratified directed-graph generator.

This module is the single source of problem instances for BOTH computational
layers of Latent Loop Lab:

    * the exact noisy-OR graph recurrence (core/recurrent.py), and
    * the learned shared-weight recurrent GNN (learned/*).

Using one generator for both is deliberate. The exact mechanism is the
reference solver on exactly the same distribution the learned model is trained
and evaluated on, so "accuracy versus inference depth R" compares like with
like.

Why the previous path-only generator was scientifically inadequate
------------------------------------------------------------------
The earlier dataset built positives as a forward chain 0 -> 1 -> ... -> L and
negatives as the same chain reversed. In the negative the target has in-degree
zero, so "is the target reachable?" collapses to "does the target have any
incoming edge?" -- a one-hop question. The task was solvable at R = 1 and the
measured accuracy was 1.000 at every R and every path length, which makes the
inference-depth experiment vacuous.

The generator here removes that shortcut. A negative instance is a *hard*
negative: the target keeps incoming edges, the source keeps outgoing edges, and
node-local statistics are matched against the positive instances. Deciding
reachability then genuinely requires propagating information along the graph,
so the target node cannot be classified until the recurrent depth R reaches the
source-to-target distance.

Guarantees provided by :func:`make_case`
----------------------------------------
For a reachable request with requested distance ``d``, the returned graph
satisfies ``bfs_distance(s, q) == d`` exactly. For an unreachable request,
``bfs_distance(s, q) is None``. Both are verified with the independent BFS
oracle before the case is returned, so a malformed case can never enter a
dataset or a test suite.
"""

from __future__ import annotations

import random
from typing import Dict, List, Optional, Tuple

from .bfs import bfs_distance
from .graph import Graph

# Node-count ladder used by the stratified suites (architecture section 10.1).
NODE_COUNTS: Tuple[int, ...] = (8, 12, 16, 24)

# Edge-density regimes. "Density" is the probability that any given ordered
# non-backbone pair is *offered* as an edge; offers that would break the
# requested distance are rejected, so realised density is lower.
DENSITIES: Tuple[float, ...] = (0.08, 0.15, 0.25)


def _distance(edges: List[Tuple[int, int]], n: int, source: int, target: int) -> Optional[int]:
    """Shortest-path distance via the independent BFS oracle."""
    outgoing: List[List[int]] = [[] for _ in range(n)]
    for u, v in edges:
        outgoing[u].append(v)
    return bfs_distance(outgoing, source, target)


def _choose_backbone(
    rng: random.Random,
    n: int,
    source: int,
    target: int,
    distance: int,
) -> Optional[List[int]]:
    """
    Pick a node sequence source -> ... -> target using ``distance`` edges.

    Returns None when the graph is too small to hold that many distinct
    interior nodes.
    """
    interior_needed = distance - 1
    available = [v for v in range(n) if v not in (source, target)]

    if interior_needed > len(available):
        return None

    interior = rng.sample(available, interior_needed)
    return [source] + interior + [target]


def make_case(
    rng: random.Random,
    n: int,
    distance: Optional[int],
    density: float,
    target_in_degree: Optional[int] = None,
    max_attempts: int = 40,
) -> Optional[Dict[str, object]]:
    """
    Build one directed graph with a controlled source-to-target distance.

    Args:
        rng: seeded random source, so every case is reproducible.
        n: number of nodes.
        distance: required shortest-path distance from source to target, or
            None to request an unreachable pair (a hard negative).
        density: probability that a candidate non-backbone edge is offered.
        target_in_degree: if given, the generator tries to give the target this
            many incoming edges. Matching this statistic between positives and
            negatives removes the strongest node-local shortcut.
        max_attempts: retries before giving up on this configuration.

    Returns:
        dict with keys ``graph``, ``source``, ``target``, ``distance``,
        ``density``, ``targetInDegree``, or None if no valid case was found.
    """
    if n < 2:
        return None

    if distance is not None and distance < 1:
        return None

    for _ in range(max_attempts):
        source = 0
        target = n - 1

        edges: List[Tuple[int, int]] = []
        backbone_pairs = set()

        if distance is not None:
            backbone = _choose_backbone(rng, n, source, target, distance)
            if backbone is None:
                return None

            for i in range(len(backbone) - 1):
                pair = (backbone[i], backbone[i + 1])
                edges.append(pair)
                backbone_pairs.add(pair)

            if _distance(edges, n, source, target) != distance:
                # A repeated node would shorten the walk; retry.
                continue
        else:
            # Hard negative: the source must still have outgoing edges, so seed
            # it with one edge to a node that is not the target.
            candidates = [v for v in range(1, n - 1)]
            if candidates:
                first = rng.choice(candidates)
                edges.append((source, first))

        def accepts(u: int, v: int) -> bool:
            """Would adding u -> v preserve the requested distance?"""
            if (u, v) in backbone_pairs:
                return False
            if any(e == (u, v) for e in edges):
                return False

            trial = edges + [(u, v)]
            d = _distance(trial, n, source, target)

            if distance is None:
                return d is None
            return d == distance

        # Offer every ordered pair once, in random order. Edges into the target
        # are withheld from this pass so that the target's in-degree is set
        # only by the controlled step below.
        controlling_in_degree = target_in_degree is not None
        pairs = [
            (u, v)
            for u in range(n)
            for v in range(n)
            if u != v and not (controlling_in_degree and v == target)
        ]
        rng.shuffle(pairs)

        for u, v in pairs:
            if rng.random() >= density:
                continue
            if accepts(u, v):
                edges.append((u, v))

        # Match the target's in-degree across positives and negatives. This is
        # the strongest node-local cue, so a case that cannot hit the budget is
        # discarded rather than emitted with a giveaway neighbourhood.
        if controlling_in_degree:
            current = sum(1 for _, v in edges if v == target)
            preds = [u for u in range(n) if u != target]
            rng.shuffle(preds)
            for u in preds:
                if current >= target_in_degree:
                    break
                if accepts(u, target):
                    edges.append((u, target))
                    current += 1

            if current != target_in_degree:
                continue

        actual = _distance(edges, n, source, target)

        if distance is None and actual is not None:
            continue
        if distance is not None and actual != distance:
            continue

        graph = Graph(n=n, edges=edges)

        # Graph.__post_init__ de-duplicates and drops self-loops; re-verify on
        # the cleaned edge list rather than trusting the pre-clean one.
        if _distance(graph.edges, n, source, target) != actual:
            continue

        return {
            "graph": graph,
            "source": source,
            "target": target,
            "distance": actual,
            "density": density,
            "targetInDegree": sum(1 for _, v in graph.edges if v == target),
        }

    return None


def sample_target_in_degree(rng: random.Random) -> int:
    """
    In-degree budget for the target node.

    Drawn from the same distribution for reachable and unreachable cases, which
    is what makes the negatives hard: a classifier cannot read the answer off
    the target's local neighbourhood.
    """
    return rng.choice([1, 1, 2, 2, 3])


def _min_edge_cut(
    edges: List[Tuple[int, int]],
    n: int,
    source: int,
    target: int,
) -> List[Tuple[int, int]]:
    """
    A minimum set of edges whose removal disconnects source from target.

    Unit-capacity Edmonds-Karp: the max flow equals the number of edge-disjoint
    source-target paths, and the saturated edges crossing the residual cut are
    a minimum cut. Graphs here have at most 24 nodes, so this is instant.
    """
    capacity: Dict[Tuple[int, int], int] = {}
    adjacency: List[set] = [set() for _ in range(n)]

    for u, v in edges:
        capacity[(u, v)] = capacity.get((u, v), 0) + 1
        capacity.setdefault((v, u), 0)
        adjacency[u].add(v)
        adjacency[v].add(u)

    while True:
        # BFS for an augmenting path in the residual graph.
        parent: Dict[int, int] = {source: source}
        queue = [source]

        while queue and target not in parent:
            u = queue.pop(0)
            for v in adjacency[u]:
                if v not in parent and capacity.get((u, v), 0) > 0:
                    parent[v] = u
                    queue.append(v)

        if target not in parent:
            break

        # Unit capacities, so every augmenting path carries exactly one unit.
        v = target
        while v != source:
            u = parent[v]
            capacity[(u, v)] -= 1
            capacity[(v, u)] = capacity.get((v, u), 0) + 1
            v = u

    # Nodes still reachable in the residual graph form the source side.
    reachable_side = {source}
    queue = [source]
    while queue:
        u = queue.pop(0)
        for v in adjacency[u]:
            if v not in reachable_side and capacity.get((u, v), 0) > 0:
                reachable_side.add(v)
                queue.append(v)

    return [
        (u, v)
        for u, v in edges
        if u in reachable_side and v not in reachable_side
    ]


def _layer_cut(
    rng: random.Random,
    edges: List[Tuple[int, int]],
    n: int,
    source: int,
    target: int,
    distance: Optional[object],
) -> List[Tuple[int, int]]:
    """
    Cut every edge leaving the source's first ``k`` BFS layers, for random k.

    Always taking the *minimum* cut biases the break toward the source, because
    severing the source's own out-edges is usually cheapest. That would make
    every negative look the same. Choosing a random layer instead spreads the
    break along the path, so the interface can show breaks that are near the
    source, near the target, and in between.
    """
    if not isinstance(distance, int) or distance < 1:
        return []

    outgoing: List[List[int]] = [[] for _ in range(n)]
    for u, v in edges:
        outgoing[u].append(v)

    # BFS layers from the source.
    layer = {source: 0}
    queue = [source]
    while queue:
        u = queue.pop(0)
        for v in outgoing[u]:
            if v not in layer:
                layer[v] = layer[u] + 1
                queue.append(v)

    k = rng.randint(1, distance)
    near_side = {node for node, depth in layer.items() if depth < k}

    if target in near_side or not near_side:
        return []

    return [(u, v) for u, v in edges if u in near_side and v not in near_side]


def make_cut_negative(
    rng: random.Random,
    positive: Dict[str, object],
    max_attempts: int = 25,
) -> Optional[Dict[str, object]]:
    """
    Turn a reachable case into an unreachable one with matched statistics.

    A minimum edge cut is removed to disconnect the pair, then exactly the same
    number of edges is added back elsewhere, chosen so the pair stays
    disconnected. The negative therefore has the *identical* node count and
    edge count as the positive it was derived from.

    This matters because a model can otherwise answer "is the target
    reachable?" from graph statistics -- ancestor-cone size or edge density --
    instead of actually propagating information along the path. With counts
    matched, the only signal left is global connectivity, which the target node
    cannot observe until the recurrent depth reaches the source-target
    distance. That is precisely the property the inference-depth experiment is
    meant to measure.
    """
    graph: Graph = positive["graph"]  # type: ignore[assignment]
    source = int(positive["source"])  # type: ignore[arg-type]
    target = int(positive["target"])  # type: ignore[arg-type]
    distance = positive["distance"]
    n = graph.n

    edges = list(graph.edges)
    cut = _layer_cut(rng, edges, n, source, target, distance)
    if not cut:
        cut = _min_edge_cut(edges, n, source, target)
    if not cut:
        return None

    remaining = [e for e in edges if e not in set(cut)]
    if _distance(remaining, n, source, target) is not None:
        return None

    # Add back exactly len(cut) edges that do not reconnect the pair.
    needed = len(cut)
    candidates = [
        (u, v)
        for u in range(n)
        for v in range(n)
        if u != v and (u, v) not in set(remaining)
    ]
    rng.shuffle(candidates)

    # A cut taken at the source's own layer strips every edge leaving the
    # source. Restoring the source's out-degree first keeps its neighbourhood
    # looking like a positive's, so the negative cannot be spotted by noticing
    # that the source goes nowhere.
    source_out_degree = sum(1 for u, _ in graph.edges if u == source)
    priority = [(u, v) for u, v in candidates if u == source]
    others = [(u, v) for u, v in candidates if u != source]

    for _ in range(max_attempts):
        trial = list(remaining)
        added = 0

        current_source_out = sum(1 for u, _ in trial if u == source)
        ordered = (
            priority + others
            if current_source_out < min(source_out_degree, needed)
            else others + priority
        )

        for u, v in ordered:
            if added >= needed:
                break
            probe = trial + [(u, v)]
            if _distance(probe, n, source, target) is None:
                trial.append((u, v))
                added += 1

        if added != needed:
            return None

        result = Graph(n=n, edges=trial)
        if _distance(result.edges, n, source, target) is not None:
            continue

        return {
            "graph": result,
            "source": source,
            "target": target,
            "distance": None,
            "density": positive["density"],
            "targetInDegree": sum(1 for _, v in result.edges if v == target),
            "derivedFrom": "cut-negative",
            "cutSize": len(cut),
            # Distance of the reachable case this negative was cut from. Lets
            # evaluation bucket each negative with its matched positive, so
            # per-distance accuracy is measured on balanced label pairs.
            "cutFromDistance": distance,
        }

    return None


def make_balanced_pair(
    rng: random.Random,
    n: int,
    distance: int,
    density: float,
) -> Optional[Tuple[Dict[str, object], Dict[str, object]]]:
    """
    Build one reachable case at ``distance`` and one matched unreachable case.

    The negative is derived from the positive by :func:`make_cut_negative`, so
    the two graphs share a node count and an edge count exactly. Falling back to
    an independently generated negative would reintroduce the density gap that
    lets a model shortcut the task, so a pair that cannot be cut is dropped
    rather than mismatched.
    """
    in_degree = sample_target_in_degree(rng)

    positive = make_case(rng, n=n, distance=distance, density=density,
                         target_in_degree=in_degree)
    if positive is None:
        return None

    negative = make_cut_negative(rng, positive)
    if negative is None:
        return None

    return positive, negative


def stratified_cases(
    num_cases: int,
    seed: int = 0,
    max_distance: int = 10,
    node_counts: Tuple[int, ...] = NODE_COUNTS,
    densities: Tuple[float, ...] = DENSITIES,
) -> List[Dict[str, object]]:
    """
    Distance-stratified case suite for exact-solver validation.

    Sweeps node counts, densities and distances d in {1, ..., max_distance}
    plus the unreachable stratum, cycling through the grid until ``num_cases``
    cases exist. Every case carries its own stratum labels so a failure can be
    traced back to a specific regime.
    """
    rng = random.Random(seed)
    cases: List[Dict[str, object]] = []

    distances: List[Optional[int]] = list(range(1, max_distance + 1)) + [None]

    grid = [
        (n, density, distance)
        for n in node_counts
        for density in densities
        for distance in distances
    ]

    index = 0
    guard = 0
    guard_limit = num_cases * 20 + len(grid) * 4

    while len(cases) < num_cases and guard < guard_limit:
        guard += 1
        n, density, distance = grid[index % len(grid)]
        index += 1

        # A distance can exceed what a graph of this size can hold.
        if distance is not None and distance > n - 1:
            continue

        case = make_case(
            rng,
            n=n,
            distance=distance,
            density=density,
            target_in_degree=sample_target_in_degree(rng),
        )
        if case is None:
            continue

        cases.append(case)

    return cases
