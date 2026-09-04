from typing import Any, Dict, List

from .graph import Graph
from .bfs import bfs_distance


def one_step(
    h: List[float],
    incoming: List[List[int]],
    alpha: float,
) -> List[float]:
    """
    One exact noisy-OR recurrent update.

    For every node v:

        h_next[v] = max(
            h[v],
            1 - product over incoming u of (1 - alpha * h[u])
        )
    """
    new_h = h.copy()

    for v, incoming_neighbors in enumerate(incoming):
        prod = 1.0

        for u in incoming_neighbors:
            prod *= (1.0 - alpha * h[u])

        candidate = 1.0 - prod

        if candidate > new_h[v]:
            new_h[v] = candidate

    return new_h


def run_exact(
    graph: Graph,
    source: int,
    target: int,
    R: int,
    alpha: float = 1.0,
    epsilon: float = 1e-12,
) -> Dict[str, Any]:
    """
    Runs the exact graph recurrence for R steps.

    Returns the full exact trajectory object expected by the frontend.
    """
    if R < 0 or R > 12:
        raise ValueError("Recurrent depth R must be between 0 and 12.")

    if not (0 <= source < graph.n):
        raise ValueError("Source node is outside graph range.")

    if not (0 <= target < graph.n):
        raise ValueError("Target node is outside graph range.")

    if not (0.0 < alpha <= 1.0):
        raise ValueError("Alpha must satisfy 0 < alpha <= 1.")

    incoming = graph.incoming()
    outgoing = graph.outgoing()

    # h(0) = one-hot source vector
    h = [0.0] * graph.n
    h[source] = 1.0

    trajectory: List[List[float]] = [h.copy()]

    for _ in range(R):
        h = one_step(h, incoming, alpha)
        trajectory.append(h.copy())

    estimate = bool(h[target] > epsilon)

    # Independent ground truth
    distance = bfs_distance(outgoing, source, target)
    reachable = distance is not None

    # Expected truth:
    # target active after R updates iff distance(source, target) <= R
    expected = bool(reachable and distance <= R)

    invariant_pass = estimate == expected

    return {
        "graph": graph.to_dict(),
        "source": source,
        "target": target,
        "alpha": alpha,
        "epsilon": epsilon,
        "R": R,
        "trajectory": trajectory,
        "estimate": estimate,
        "targetActivation": h[target],
        "bfsDistance": distance,
        "reachable": reachable,
        "expected": expected,
        "invariantPass": invariant_pass,
        "evidence": {
            "evidenceType": "Live computation",
            "source": "Exact noisy-OR graph recurrence",
            "experimentId": "exact-graph-recurrence",
        },
    }
