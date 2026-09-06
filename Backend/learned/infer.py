"""
Deterministic inference from a frozen checkpoint.

Architecture section 5.3 requires a deterministic inference script so that any
number shown in the interface can be reproduced from the command line. Given
the same checkpoint, the same graph and the same R, this prints the same
probability every time: the model is in eval mode, no gradient is taken, and
the only randomness -- which graph is generated -- is seeded explicitly.

Usage:
    # A generated case at a chosen distance
    python -m learned.infer --checkpoint results/model_seed1.pt --distance 6 --R 8

    # The same case, unreachable
    python -m learned.infer --checkpoint results/model_seed1.pt --unreachable --R 8

    # An explicit graph
    python -m learned.infer --checkpoint results/model_seed1.pt --R 4 \\
        --graph '{"n":5,"edges":[[0,1],[1,2],[2,3],[3,4]]}' --source 0 --target 4

    # Sweep depth on one fixed case: same weights, same input, different R
    python -m learned.infer --checkpoint results/model_seed1.pt --distance 6 --sweep
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Dict, Optional

import torch

from core.graph import Graph
from core.recurrent import run_exact
from learned.dataset import case_to_sample, collate, make_demo_case
from learned.evaluate import file_hash, load_checkpoint


def sample_from_graph(graph: Graph, source: int, target: int) -> Dict[str, object]:
    """Wrap an explicit graph in the sample format, using BFS for the label."""
    from core.bfs import bfs_distance

    distance = bfs_distance(graph.outgoing(), source, target)
    return case_to_sample({
        "graph": graph,
        "source": source,
        "target": target,
        "distance": distance,
        "density": None,
        "targetInDegree": sum(1 for _, v in graph.edges if v == target),
    })


@torch.no_grad()
def infer(model, sample: Dict[str, object], R: int) -> Dict[str, object]:
    """One deterministic forward pass at depth R."""
    batch = collate([sample])
    logit, trajectory, logits_per_step = model.forward_with_trajectory(
        batch.x, batch.edge_index, batch.target_idx, R
    )

    probability = torch.sigmoid(logit)[0].item()
    prediction = 1.0 if probability > 0.5 else 0.0
    label = float(sample["label"])  # type: ignore[arg-type]

    return {
        "R": R,
        "distance": sample["distance"],
        "label": label,
        "prediction": prediction,
        "probability": probability,
        "confidence": probability if prediction == 1.0 else 1.0 - probability,
        "correct": prediction == label,
        "probabilityPerStep": [
            torch.sigmoid(step)[0].item() for step in logits_per_step
        ],
        "targetStateNormPerStep": [
            step[batch.target_idx[0]].norm().item() for step in trajectory
        ],
    }


def main() -> None:
    parser = argparse.ArgumentParser(description="Deterministic learned inference")
    parser.add_argument("--checkpoint", type=str, default="results/model_seed1.pt")
    parser.add_argument("--R", type=int, default=4)
    parser.add_argument("--sweep", action="store_true",
                        help="Report every depth from 1 to --max-R on one case")
    parser.add_argument("--max-R", type=int, default=10)

    parser.add_argument("--distance", type=int, default=None,
                        help="Generate a reachable case at this distance")
    parser.add_argument("--unreachable", action="store_true",
                        help="Generate an unreachable case instead")
    parser.add_argument("--nodes", type=int, default=12)
    parser.add_argument("--density", type=float, default=0.15)
    parser.add_argument("--case-seed", type=int, default=0)

    parser.add_argument("--graph", type=str, default=None,
                        help='Explicit graph as JSON, e.g. {"n":5,"edges":[[0,1]]}')
    parser.add_argument("--source", type=int, default=0)
    parser.add_argument("--target", type=int, default=None)
    parser.add_argument("--json", action="store_true", help="Emit JSON only")
    args = parser.parse_args()

    checkpoint_path = Path(args.checkpoint)
    if not checkpoint_path.exists():
        raise SystemExit(
            f"Checkpoint not found: {checkpoint_path}\n"
            f"Train one first:  python -m learned.train --seed 1"
        )

    device = torch.device("cpu")
    model, checkpoint = load_checkpoint(checkpoint_path, device)
    digest = file_hash(checkpoint_path)

    if args.graph:
        payload = json.loads(args.graph)
        graph = Graph.from_dict(payload)
        target = args.target if args.target is not None else graph.n - 1
        sample = sample_from_graph(graph, args.source, target)
    else:
        distance: Optional[int] = None if args.unreachable else (args.distance or 4)
        sample = make_demo_case(
            distance=distance,
            n=args.nodes,
            density=args.density,
            seed=args.case_seed,
        )
        if sample is None:
            raise SystemExit(
                f"Could not generate a case with distance={distance} on "
                f"{args.nodes} nodes. Try a larger --nodes."
            )

    graph = Graph.from_dict(sample["graph"])  # type: ignore[arg-type]
    source = int(sample["source_idx"])  # type: ignore[arg-type]
    target = int(sample["target_idx"])  # type: ignore[arg-type]

    depths = range(1, args.max_R + 1) if args.sweep else [args.R]
    ceiling = model.max_inference_depth

    rows = []
    for R in depths:
        if R > ceiling:
            rows.append({
                "R": R,
                "unavailable": True,
                "reason": f"Model has only {ceiling} update blocks (no shared weights).",
            })
            continue

        row = infer(model, sample, R)
        exact = run_exact(graph, source, target, R=min(R, 12))
        row["exactEstimate"] = exact["estimate"]
        row["exactTargetActivation"] = exact["targetActivation"]
        row["bfsDistance"] = exact["bfsDistance"]
        rows.append(row)

    output = {
        "checkpoint": str(checkpoint_path),
        "checkpointHash": digest,
        "sharedWeights": checkpoint.get("shared_weights", True),
        "trainingMaxPathLength": checkpoint.get("trainingMaxPathLength"),
        "trainingMaxR": checkpoint.get("max_R"),
        "graph": sample["graph"],
        "source": source,
        "target": target,
        "bfsDistance": sample["distance"],
        "label": sample["label"],
        "results": rows,
        "evidence": {
            "evidenceType": "Live computation",
            "source": "Deterministic inference from a frozen toy checkpoint",
            "experimentId": f"infer-{digest}",
        },
    }

    if args.json:
        print(json.dumps(output, indent=2))
        return

    print(f"Checkpoint      {checkpoint_path.name}  (sha256 {digest})")
    print(f"Shared weights  {output['sharedWeights']}  "
          f"| trained on distances <= {output['trainingMaxPathLength']}")
    print(f"Graph           n={graph.n}, {len(graph.edges)} edges, "
          f"source={source}, target={target}")
    print(f"BFS distance    {sample['distance'] if sample['distance'] is not None else 'unreachable'}")
    print(f"True label      {'reachable' if sample['label'] == 1.0 else 'unreachable'}")
    print()
    print(f"{'R':>3}  {'learned p':>10}  {'predicts':>12}  {'exact':>12}  {'correct':>8}")
    print("-" * 54)

    for row in rows:
        if row.get("unavailable"):
            print(f"{row['R']:>3}  {'--':>10}  {'unavailable':>12}  "
                  f"{'--':>12}  {'--':>8}")
            continue
        predicts = "reachable" if row["prediction"] == 1.0 else "unreachable"
        exact_says = "reachable" if row["exactEstimate"] else "unreachable"
        print(f"{row['R']:>3}  {row['probability']:>10.4f}  {predicts:>12}  "
              f"{exact_says:>12}  {'yes' if row['correct'] else 'NO':>8}")


if __name__ == "__main__":
    main()
