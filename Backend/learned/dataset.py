"""
Dataset generation for Latent Loop Lab learned GNN.

Generates directed path graphs with source/target indicators.
Training: paths of length <= 4
Testing: paths of length 1 to 10 (including unseen longer paths)

Each sample:
    - node_features: [num_nodes, 2] with [is_source, is_target]
    - edge_index: [2, num_edges] directed edges
    - target_idx: index of target node
    - label: 1 if reachable, 0 if not
    - path_length: length of the path
"""

import random
import torch
from torch.utils.data import Dataset
from typing import List, Tuple, Dict


def make_path_graph(
    length: int,
    reachable: bool = True,
    rng: random.Random = None,
) -> Dict:
    """
    Create a directed path graph.

    If reachable=True: source at start, target at end, edges go forward.
    If reachable=False: edges go backward (target not reachable from source).

    Args:
        length: number of edges in the path (nodes = length + 1)
        reachable: whether target is reachable from source
        rng: random number generator

    Returns:
        dict with node_features, edge_index, target_idx, label, path_length
    """
    num_nodes = length + 1

    # Source is always node 0, target is always last node
    source_idx = 0
    target_idx = length

    # Node features: [is_source, is_target]
    node_features = torch.zeros(num_nodes, 2)
    node_features[source_idx, 0] = 1.0  # source indicator
    node_features[target_idx, 1] = 1.0  # target indicator

    # Edges
    if reachable:
        # Forward path: 0 -> 1 -> 2 -> ... -> length
        src = list(range(length))
        dst = list(range(1, length + 1))
    else:
        # Backward path: length -> ... -> 1 -> 0
        # Target (last node) is NOT reachable from source (first node)
        src = list(range(1, length + 1))
        dst = list(range(length))

    edge_index = torch.tensor([src, dst], dtype=torch.long)

    label = 1.0 if reachable else 0.0

    return {
        "node_features": node_features,
        "edge_index": edge_index,
        "source_idx": source_idx,
        "target_idx": target_idx,
        "label": label,
        "path_length": length,
    }


def make_disconnected_graph(
    num_nodes: int,
    rng: random.Random = None,
) -> Dict:
    """
    Create a graph where source and target are in disconnected components.
    Target is not reachable.
    """
    if rng is None:
        rng = random.Random()

    source_idx = 0
    target_idx = num_nodes - 1

    # Node features
    node_features = torch.zeros(num_nodes, 2)
    node_features[source_idx, 0] = 1.0
    node_features[target_idx, 1] = 1.0

    # Edges only in first half (source side)
    # No edges connecting to target
    half = num_nodes // 2
    src = list(range(half - 1))
    dst = list(range(1, half))

    edge_index = torch.tensor([src, dst], dtype=torch.long) if src else torch.zeros(2, 0, dtype=torch.long)

    return {
        "node_features": node_features,
        "edge_index": edge_index,
        "source_idx": source_idx,
        "target_idx": target_idx,
        "label": 0.0,
        "path_length": num_nodes - 1,
    }


class PathReachabilityDataset(Dataset):
    """
    Dataset of path reachability problems.

    Generates balanced positive (reachable) and negative (unreachable) examples.
    """

    def __init__(
        self,
        path_lengths: List[int],
        samples_per_length: int = 100,
        seed: int = 42,
        include_disconnected: bool = True,
    ):
        """
        Args:
            path_lengths: list of path lengths to generate
            samples_per_length: number of samples per path length
            seed: random seed
            include_disconnected: whether to include disconnected graphs
        """
        self.samples = []
        rng = random.Random(seed)

        for length in path_lengths:
            for _ in range(samples_per_length // 2):
                # Positive: reachable
                self.samples.append(make_path_graph(length, reachable=True, rng=rng))

                # Negative: not reachable (reversed direction)
                self.samples.append(make_path_graph(length, reachable=False, rng=rng))

            # Add some disconnected graphs
            if include_disconnected and length >= 3:
                for _ in range(samples_per_length // 4):
                    self.samples.append(
                        make_disconnected_graph(length + 1, rng=rng)
                    )

        # Shuffle
        rng.shuffle(self.samples)

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        return self.samples[idx]


def collate_fn(batch: List[Dict]) -> List[Dict]:
    """
    Custom collate: keep graphs as separate items since they have
    different numbers of nodes.
    """
    return batch


def get_training_dataset(seed: int = 42) -> PathReachabilityDataset:
    """
    Training dataset: paths of length 1 to 4.
    """
    return PathReachabilityDataset(
        path_lengths=[1, 2, 3, 4],
        samples_per_length=200,
        seed=seed,
    )


def get_test_dataset(seed: int = 123) -> PathReachabilityDataset:
    """
    Test dataset: paths of length 1 to 10.
    Includes unseen longer paths (5 to 10).
    """
    return PathReachabilityDataset(
        path_lengths=[1, 2, 3, 4, 5, 6, 7, 8, 9, 10],
        samples_per_length=50,
        seed=seed,
    )
