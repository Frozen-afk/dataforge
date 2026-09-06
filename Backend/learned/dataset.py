"""
Dataset construction for the learned shared-weight recurrent GNN.

Every instance is a directed graph with a marked source and target, and the
label is whether the target is reachable from the source. Instances come from
:mod:`core.generator`, the same distance-stratified generator that feeds the
exact-solver validation suite, so the exact mechanism is an exact reference on
the identical distribution.

Task design
-----------
Positives are generated with an exact source-to-target distance ``d``.
Negatives are *hard*: the target keeps a matched in-degree and the source keeps
outgoing edges, so reachability cannot be read off any node's immediate
neighbourhood. A classifier reading the target node's state after ``R``
recurrent updates can only see its ``R``-hop ancestor cone, so an instance at
distance ``d`` is undecidable until ``R >= d``. This is what makes "accuracy
versus inference depth" a real measurement rather than a flat line.

How the negatives are matched (and what remains unmatched)
----------------------------------------------------------
Each negative is derived from a specific positive by cutting the edges leaving
a randomly chosen BFS layer and adding the same number of edges back elsewhere,
so the pair shares a node count and an edge count *exactly*. An earlier version
generated negatives independently, which left them roughly 30 edges against 42
for positives; a model learned to read that density gap instead of propagating,
and scored above the exact mechanism at depths where the answer was not yet
knowable. Matching the counts removed that shortcut.

Two smaller asymmetries remain and are reported rather than hidden: negatives
average a slightly lower target in-degree (about 1.63 against 1.78) and a lower
source out-degree (about 1.56 against 1.99). Neither is visible from the target
node until the recurrent depth reaches the source, so neither substitutes for
propagation, but the per-distance breakdown in ``learned/evaluate.py`` exists so
this can be inspected instead of assumed away.

Batching
--------
Graphs have different node counts, so a batch is assembled block-diagonally:
node indices are offset per graph, edge lists are concatenated, and each
graph's target index is recorded in global coordinates. One forward pass then
covers the whole batch, which is what makes five-seed sweeps over R = 1..10
tractable on CPU.
"""

from __future__ import annotations

import random
from dataclasses import dataclass
from typing import Dict, List, Optional, Sequence, Tuple

import torch

from core.generator import DENSITIES, make_balanced_pair, make_case, sample_target_in_degree
from core.graph import Graph

# Node counts used per distance. Longer distances need larger graphs, since a
# distance-d backbone consumes d + 1 distinct nodes.
SMALL_NODE_COUNTS: Tuple[int, ...] = (8, 12)
LARGE_NODE_COUNTS: Tuple[int, ...] = (12, 16, 24)

TRAIN_DISTANCES: Tuple[int, ...] = (1, 2, 3, 4)
TEST_DISTANCES: Tuple[int, ...] = (1, 2, 3, 4, 5, 6, 7, 8, 9, 10)


def node_counts_for(distance: Optional[int]) -> Tuple[int, ...]:
    """Node-count options that can actually hold a backbone of this length."""
    if distance is None:
        return SMALL_NODE_COUNTS + LARGE_NODE_COUNTS
    if distance <= 4:
        return SMALL_NODE_COUNTS
    return tuple(n for n in LARGE_NODE_COUNTS if n - 1 >= distance)


def case_to_sample(case: Dict[str, object]) -> Dict[str, object]:
    """
    Convert a generator case into tensors.

    Node features are ``[is_source, is_target]`` exactly as the architecture
    specifies, so the model is given the problem statement and nothing else --
    in particular it is never given the distance or the answer.
    """
    graph: Graph = case["graph"]  # type: ignore[assignment]
    source = int(case["source"])  # type: ignore[arg-type]
    target = int(case["target"])  # type: ignore[arg-type]
    distance = case["distance"]

    # A negative has no distance of its own, so it inherits the distance of the
    # reachable case it was cut from. Bucketing by this keeps every per-distance
    # figure balanced between one positive and one matched negative, which is
    # what makes 0.5 the honest chance baseline at each distance.
    bucket_distance = distance if distance is not None else case.get("cutFromDistance")

    x = torch.zeros(graph.n, 2)
    x[source, 0] = 1.0
    x[target, 1] = 1.0

    if graph.edges:
        edge_index = torch.tensor(
            [[u for u, _ in graph.edges], [v for _, v in graph.edges]],
            dtype=torch.long,
        )
    else:
        edge_index = torch.zeros(2, 0, dtype=torch.long)

    return {
        "x": x,
        "edge_index": edge_index,
        "source_idx": source,
        "target_idx": target,
        "label": 1.0 if distance is not None else 0.0,
        "distance": distance,
        "bucket_distance": bucket_distance,
        "num_nodes": graph.n,
        "num_edges": len(graph.edges),
        "graph": graph.to_dict(),
    }


@dataclass
class Batch:
    """A block-diagonal batch of graphs."""

    x: torch.Tensor            # [total_nodes, 2]
    edge_index: torch.Tensor   # [2, total_edges], global node indices
    target_idx: torch.Tensor   # [B] global index of each graph's target
    y: torch.Tensor            # [B] reachability labels
    distances: List[Optional[int]]
    bucket_distances: List[Optional[int]]
    num_graphs: int

    def to(self, device: torch.device) -> "Batch":
        return Batch(
            x=self.x.to(device),
            edge_index=self.edge_index.to(device),
            target_idx=self.target_idx.to(device),
            y=self.y.to(device),
            distances=self.distances,
            bucket_distances=self.bucket_distances,
            num_graphs=self.num_graphs,
        )


def collate(samples: Sequence[Dict[str, object]]) -> Batch:
    """Assemble samples into one block-diagonal batch."""
    xs: List[torch.Tensor] = []
    edge_srcs: List[torch.Tensor] = []
    edge_dsts: List[torch.Tensor] = []
    targets: List[int] = []
    labels: List[float] = []
    distances: List[Optional[int]] = []
    bucket_distances: List[Optional[int]] = []

    offset = 0
    for sample in samples:
        x: torch.Tensor = sample["x"]  # type: ignore[assignment]
        edge_index: torch.Tensor = sample["edge_index"]  # type: ignore[assignment]

        xs.append(x)
        if edge_index.numel() > 0:
            edge_srcs.append(edge_index[0] + offset)
            edge_dsts.append(edge_index[1] + offset)

        targets.append(int(sample["target_idx"]) + offset)  # type: ignore[arg-type]
        labels.append(float(sample["label"]))  # type: ignore[arg-type]
        distances.append(sample["distance"])  # type: ignore[arg-type]
        bucket_distances.append(sample.get("bucket_distance"))  # type: ignore[arg-type]

        offset += int(sample["num_nodes"])  # type: ignore[arg-type]

    if edge_srcs:
        edge_index = torch.stack([torch.cat(edge_srcs), torch.cat(edge_dsts)])
    else:
        edge_index = torch.zeros(2, 0, dtype=torch.long)

    return Batch(
        x=torch.cat(xs, dim=0),
        edge_index=edge_index,
        target_idx=torch.tensor(targets, dtype=torch.long),
        y=torch.tensor(labels, dtype=torch.float),
        distances=distances,
        bucket_distances=bucket_distances,
        num_graphs=len(samples),
    )


def build_dataset(
    distances: Sequence[int],
    pairs_per_distance: int,
    seed: int,
    densities: Sequence[float] = DENSITIES,
) -> List[Dict[str, object]]:
    """
    Build a label-balanced dataset.

    For each requested distance the generator produces matched (reachable,
    unreachable) pairs, so labels are balanced at every distance and a model
    cannot profit from a class prior that varies with distance.
    """
    rng = random.Random(seed)
    samples: List[Dict[str, object]] = []

    for distance in distances:
        options = node_counts_for(distance)
        if not options:
            continue

        made = 0
        attempts = 0
        while made < pairs_per_distance and attempts < pairs_per_distance * 20:
            attempts += 1
            n = rng.choice(options)
            density = rng.choice(list(densities))

            pair = make_balanced_pair(rng, n=n, distance=distance, density=density)
            if pair is None:
                continue

            positive, negative = pair
            samples.append(case_to_sample(positive))
            samples.append(case_to_sample(negative))
            made += 1

    rng.shuffle(samples)
    return samples


def build_training_set(seed: int, pairs_per_distance: int = 150) -> List[Dict[str, object]]:
    """Training set: distances 1..4 only. Longer distances are never seen."""
    return build_dataset(TRAIN_DISTANCES, pairs_per_distance, seed=seed)


def build_validation_set(seed: int, pairs_per_distance: int = 40) -> List[Dict[str, object]]:
    """Validation set drawn from the training distance range, disjoint seed."""
    return build_dataset(TRAIN_DISTANCES, pairs_per_distance, seed=seed + 5000)


def build_test_set(seed: int, pairs_per_distance: int = 40) -> List[Dict[str, object]]:
    """Test set: distances 1..10, so 5..10 are outside the training range."""
    return build_dataset(TEST_DISTANCES, pairs_per_distance, seed=seed + 9000)


def iterate_batches(
    samples: Sequence[Dict[str, object]],
    batch_size: int,
    shuffle: bool = False,
    rng: Optional[random.Random] = None,
):
    """Yield :class:`Batch` objects over a sample list."""
    order = list(range(len(samples)))
    if shuffle:
        (rng or random).shuffle(order)

    for start in range(0, len(order), batch_size):
        chunk = [samples[i] for i in order[start:start + batch_size]]
        if chunk:
            yield collate(chunk)


def make_demo_case(
    distance: Optional[int],
    n: int = 12,
    density: float = 0.15,
    seed: int = 0,
) -> Optional[Dict[str, object]]:
    """
    Single reproducible case for live inference from the web interface.

    Used by ``POST /learned/run`` so the page can show the learned model and
    the exact mechanism operating on one identical graph.
    """
    rng = random.Random(seed)
    case = make_case(
        rng,
        n=n,
        distance=distance,
        density=density,
        target_in_degree=sample_target_in_degree(rng),
    )
    if case is None:
        return None
    return case_to_sample(case)
