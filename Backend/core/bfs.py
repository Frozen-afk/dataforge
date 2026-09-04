from collections import deque
from typing import List, Optional


def bfs_distance(
    outgoing: List[List[int]],
    source: int,
    target: int,
) -> Optional[int]:
    """
    Independent BFS oracle.

    Returns:
        shortest path distance from source to target
        or None if target is unreachable.

    This must remain independent from the recurrent solver.
    """
    if source == target:
        return 0

    n = len(outgoing)

    if not (0 <= source < n):
        raise ValueError("BFS source is outside graph range.")

    if not (0 <= target < n):
        raise ValueError("BFS target is outside graph range.")

    distance = [-1] * n
    distance[source] = 0

    queue = deque([source])

    while queue:
        u = queue.popleft()

        for v in outgoing[u]:
            if distance[v] == -1:
                distance[v] = distance[u] + 1

                if v == target:
                    return distance[v]

                queue.append(v)

    return None
