from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Dict, List, Tuple

Node = int
Edge = Tuple[Node, Node]


@dataclass
class Graph:
    """
    Directed graph used for exact latent recurrence.

    Constraints from the architecture document:
        n <= 24
        R <= 12 is enforced in recurrent.py
    """

    n: int
    edges: List[Edge]

    def __post_init__(self) -> None:
        if self.n <= 0:
            raise ValueError("Graph must have at least one node.")

        if self.n > 24:
            raise ValueError("Graph size is capped at n <= 24.")

        clean_edges: List[Edge] = []
        seen = set()

        for u, v in self.edges:
            u = int(u)
            v = int(v)

            if not (0 <= u < self.n and 0 <= v < self.n):
                raise ValueError(f"Edge {u} -> {v} is outside node range 0..{self.n - 1}.")

            # Skip self-loops for the first clean mechanism.
            if u == v:
                continue

            if (u, v) not in seen:
                seen.add((u, v))
                clean_edges.append((u, v))

        self.edges = clean_edges

    def outgoing(self) -> List[List[Node]]:
        """
        Returns outgoing adjacency list.
        outgoing[u] contains all v such that u -> v exists.
        """
        adj: List[List[Node]] = [[] for _ in range(self.n)]

        for u, v in self.edges:
            adj[u].append(v)

        return adj

    def incoming(self) -> List[List[Node]]:
        """
        Returns incoming adjacency list.
        incoming[v] contains all u such that u -> v exists.
        """
        inc: List[List[Node]] = [[] for _ in range(self.n)]

        for u, v in self.edges:
            inc[v].append(u)

        return inc

    def to_dict(self) -> Dict[str, Any]:
        return {
            "n": self.n,
            "edges": [[u, v] for u, v in self.edges],
        }

    @classmethod
    def from_dict(cls, payload: Dict[str, Any]) -> "Graph":
        return cls(
            n=int(payload["n"]),
            edges=[(int(u), int(v)) for u, v in payload.get("edges", [])],
        )


# ------------------------------------------------------------------
# Preset graphs
# ------------------------------------------------------------------


def make_line(length: int) -> Tuple[Graph, int, int]:
    """
    Makes a line graph:

        0 -> 1 -> 2 -> ... -> length

    shortest path distance = length
    """
    if length < 1:
        raise ValueError("Line length must be at least 1.")

    n = length + 1
    edges = [(i, i + 1) for i in range(length)]

    graph = Graph(n=n, edges=edges)
    source = 0
    target = length

    return graph, source, target


def make_branch() -> Tuple[Graph, int, int]:
    """
    Branching graph with two paths from 0 to 5.

    Shortest path distance = 3.
    """
    edges = [
        (0, 1),
        (1, 2),
        (2, 5),
        (0, 3),
        (3, 4),
        (4, 5),
        (1, 4),
    ]

    graph = Graph(n=6, edges=edges)
    source = 0
    target = 5

    return graph, source, target


def make_diamond() -> Tuple[Graph, int, int]:
    """
    Diamond graph:

        0 -> 1 -> 3
        0 -> 2 -> 3

    Shortest path distance = 2.
    """
    edges = [
        (0, 1),
        (0, 2),
        (1, 3),
        (2, 3),
    ]

    graph = Graph(n=4, edges=edges)
    source = 0
    target = 3

    return graph, source, target


def make_cycle() -> Tuple[Graph, int, int]:
    """
    Directed cycle of length 6.

    source = 0
    target = 4

    Shortest path distance = 4.
    """
    n = 6
    edges = [(i, (i + 1) % n) for i in range(n)]

    graph = Graph(n=n, edges=edges)
    source = 0
    target = 4

    return graph, source, target


def make_disconnected() -> Tuple[Graph, int, int]:
    """
    Disconnected graph.

    source = 0
    target = 5

    No path exists.
    """
    edges = [
        (0, 1),
        (1, 2),
        (4, 5),
    ]

    graph = Graph(n=7, edges=edges)
    source = 0
    target = 5

    return graph, source, target


PRESETS: Dict[str, Dict[str, Any]] = {
    "line": {
        "factory": make_line,
        "kwargs": {"length": 4},
        "description": "Four-hop line. Target becomes active at R = 4.",
    },
    "line_short": {
        "factory": make_line,
        "kwargs": {"length": 2},
        "description": "Two-hop line. Target becomes active at R = 2.",
    },
    "branch": {
        "factory": make_branch,
        "kwargs": {},
        "description": "Branching graph. Target becomes active at R = 3.",
    },
    "diamond": {
        "factory": make_diamond,
        "kwargs": {},
        "description": "Diamond graph. Target becomes active at R = 2.",
    },
    "cycle": {
        "factory": make_cycle,
        "kwargs": {},
        "description": "Directed cycle. Target becomes active at R = 4.",
    },
    "disconnected": {
        "factory": make_disconnected,
        "kwargs": {},
        "description": "Disconnected graph. Target never becomes active.",
    },
}


def get_preset(name: str) -> Tuple[Graph, int, int, str]:
    if name not in PRESETS:
        raise KeyError(f"Unknown preset: {name}")

    item = PRESETS[name]
    graph, source, target = item["factory"](**item.get("kwargs", {}))
    description = item["description"]

    return graph, source, target, description


def preset_names() -> List[str]:
    return list(PRESETS.keys())
