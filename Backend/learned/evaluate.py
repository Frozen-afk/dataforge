"""
The inference-depth experiment.

Main question (architecture section 6.1):

    With model parameters fixed, what changes when the model is allowed to
    apply its latent update more times at inference?

The comparison held constant is

    same input + same weights + same architecture + different R

so nothing but the amount of computation varies. Every checkpoint is frozen
before this script runs; no gradient is ever taken here.

What this produces
------------------
* accuracy versus inference depth R = 1..10;
* the same broken down per source-to-target distance, separating distances seen
  in training (1..4) from unseen ones (5..10);
* the exact mechanism's accuracy on the identical test set, computed live as
  the reference ceiling rather than assumed to be 1.0;
* mean latent-state norm per depth, which is the evidence for the claim that
  excessive recurrence can saturate or destabilise learned dynamics;
* categorised successful and failed examples for the interface.

Usage:
    python -m learned.evaluate --seed 1 --checkpoint results/model_seed1.pt
"""

from __future__ import annotations

import argparse
import hashlib
import json
import random
from collections import defaultdict
from pathlib import Path
from typing import Dict, List, Optional

import numpy as np
import torch

from core.graph import Graph
from core.recurrent import run_exact
from learned.dataset import (
    TEST_DISTANCES,
    TRAIN_DISTANCES,
    build_test_set,
    collate,
    iterate_batches,
)
from learned.model import build_model

TRAINING_MAX_DISTANCE = max(TRAIN_DISTANCES)


def _sort_key(key: str):
    """Order per-distance buckets numerically, with non-numeric keys last."""
    return (int(key), "") if key.isdigit() else (10**9, key)


def set_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)


def file_hash(path: Path) -> str:
    digest = hashlib.sha256()
    with open(path, "rb") as handle:
        for chunk in iter(lambda: handle.read(8192), b""):
            digest.update(chunk)
    return digest.hexdigest()[:16]


def load_checkpoint(path: Path, device: torch.device):
    """Rebuild a frozen model exactly as it was trained."""
    # Tensors and plain scalars only, so the safe loader is sufficient. It
    # refuses to unpickle arbitrary objects, which matters because this is the
    # function anyone reproducing the results points at a .pt file.
    checkpoint = torch.load(path, map_location=device, weights_only=True)

    model = build_model(
        shared=checkpoint.get("shared_weights", True),
        hidden_dim=checkpoint.get("hidden_dim", 32),
        aggregation=checkpoint.get("aggregation", "max"),
        num_steps=checkpoint.get("max_R", 4),
    ).to(device)

    model.load_state_dict(checkpoint["model_state_dict"])
    model.eval()

    return model, checkpoint


def failure_category(
    correct: bool,
    label: float,
    distance: Optional[int],
    R: int,
) -> str:
    """
    Name the reason an instance was decided the way it was.

    The interface uses these labels directly, so that a wrong answer caused by
    too little computation is never confused with one caused by a limit of what
    the model learned.
    """
    if correct:
        if label == 1.0 and distance is not None and distance > TRAINING_MAX_DISTANCE:
            return "correct-extrapolation"
        return "correct"

    if label == 0.0:
        return "false-positive"

    if distance is not None and R < distance:
        return "insufficient-depth"

    if distance is not None and distance > TRAINING_MAX_DISTANCE:
        return "extrapolation-failure"

    return "in-distribution-error"


@torch.no_grad()
def exact_reference(samples: List[Dict[str, object]], R: int) -> Dict[str, object]:
    """
    Accuracy of the exact noisy-OR mechanism on the same test set at depth R.

    This is deliberately measured, not asserted. The exact mechanism activates
    the target exactly when d(s, q) <= R, so its accuracy is a sharp step: it is
    perfect on unreachable pairs at every depth and perfect on reachable pairs
    only once R reaches their distance. That step is the ceiling the learned
    model is compared against.
    """
    correct = 0
    per_distance: Dict[str, Dict[str, float]] = defaultdict(
        lambda: {"correct": 0, "total": 0}
    )

    for sample in samples:
        graph = Graph.from_dict(sample["graph"])  # type: ignore[arg-type]
        result = run_exact(
            graph=graph,
            source=int(sample["source_idx"]),  # type: ignore[arg-type]
            target=int(sample["target_idx"]),  # type: ignore[arg-type]
            R=min(R, 12),
        )

        predicted = 1.0 if result["estimate"] else 0.0
        is_correct = predicted == float(sample["label"])  # type: ignore[arg-type]

        bucket = sample.get("bucket_distance")
        key = str(bucket) if bucket is not None else "unbucketed"
        per_distance[key]["correct"] += int(is_correct)
        per_distance[key]["total"] += 1
        correct += int(is_correct)

    return {
        "accuracy": correct / max(len(samples), 1),
        "perDistance": {
            key: {
                "accuracy": value["correct"] / max(value["total"], 1),
                "samples": value["total"],
            }
            for key, value in sorted(
                per_distance.items(), key=lambda kv: (not kv[0].isdigit(), _sort_key(kv[0]))
            )
        },
    }


@torch.no_grad()
def evaluate_at_depth(
    model,
    samples: List[Dict[str, object]],
    R: int,
    device: torch.device,
    batch_size: int = 64,
) -> Dict[str, object]:
    """Run the frozen model over the whole test set at one inference depth."""
    model.eval()

    correct = 0
    total = 0
    confidence_sum = 0.0
    norm_sum = 0.0
    norm_nodes = 0

    per_distance: Dict[str, Dict[str, float]] = defaultdict(
        lambda: {"correct": 0, "total": 0, "confidence": 0.0}
    )
    categories: Dict[str, int] = defaultdict(int)

    for batch in iterate_batches(samples, batch_size):
        batch = batch.to(device)

        logits, trajectory, _ = model.forward_with_trajectory(
            batch.x, batch.edge_index, batch.target_idx, R
        )
        probs = torch.sigmoid(logits)
        preds = (probs > 0.5).float()

        # Summed, not averaged per batch. Graphs vary in node count and the
        # last batch is usually short, so a mean of batch means would weight
        # a node in a small batch more heavily than one in a full batch.
        node_norms = trajectory[-1].norm(dim=-1)
        norm_sum += node_norms.sum().item()
        norm_nodes += node_norms.numel()

        for i in range(batch.num_graphs):
            label = batch.y[i].item()
            prob = probs[i].item()
            pred = preds[i].item()
            is_correct = pred == label
            confidence = prob if pred == 1.0 else 1.0 - prob
            distance = batch.distances[i]
            bucket = batch.bucket_distances[i]

            key = str(bucket) if bucket is not None else "unbucketed"
            per_distance[key]["correct"] += int(is_correct)
            per_distance[key]["total"] += 1
            per_distance[key]["confidence"] += confidence

            categories[failure_category(is_correct, label, distance, R)] += 1

            correct += int(is_correct)
            confidence_sum += confidence
            total += 1

    # Each bucket holds one positive at that distance and its matched negative,
    # so these splits are label-balanced and 0.5 is the chance baseline.
    seen = {"correct": 0, "total": 0}
    unseen = {"correct": 0, "total": 0}
    for key, value in per_distance.items():
        if not key.isdigit():
            continue
        target_bucket = seen if int(key) <= TRAINING_MAX_DISTANCE else unseen
        target_bucket["correct"] += value["correct"]
        target_bucket["total"] += value["total"]

    return {
        "R": R,
        "accuracy": correct / max(total, 1),
        "confidence": confidence_sum / max(total, 1),
        "totalSamples": total,
        "meanStateNorm": norm_sum / max(norm_nodes, 1),
        "accuracySeenDistances": seen["correct"] / max(seen["total"], 1),
        "accuracyUnseenDistances": unseen["correct"] / max(unseen["total"], 1),
        "perDistance": {
            key: {
                "accuracy": value["correct"] / max(value["total"], 1),
                "confidence": value["confidence"] / max(value["total"], 1),
                "samples": int(value["total"]),
            }
            for key, value in sorted(
                per_distance.items(), key=lambda kv: (not kv[0].isdigit(), _sort_key(kv[0]))
            )
        },
        "failureCategories": dict(categories),
    }


@torch.no_grad()
def collect_examples(
    model,
    samples: List[Dict[str, object]],
    device: torch.device,
    per_category: int = 3,
) -> List[Dict[str, object]]:
    """
    Gather concrete cases the interface can show, one card per category.

    Both successes and failures are collected on purpose: a page that only
    shows wins would misrepresent what recurrence does.
    """
    wanted = [
        "correct",
        "correct-extrapolation",
        "insufficient-depth",
        "extrapolation-failure",
        "in-distribution-error",
        "false-positive",
    ]
    found: Dict[str, List[Dict[str, object]]] = {key: [] for key in wanted}

    model.eval()

    for sample in samples:
        if all(len(found[key]) >= per_category for key in wanted):
            break

        distance = sample["distance"]
        # Probe each case below, at and above its distance. The unshared
        # ablation has no weights past its trained depth, so probes are clipped
        # to whatever the model can actually run.
        ceiling = model.max_inference_depth
        probe_depths = sorted(
            {min(depth, ceiling) for depth in (2, TRAINING_MAX_DISTANCE, 7, 10)}
        )

        for R in probe_depths:
            if all(len(found[key]) >= per_category for key in wanted):
                break

            batch = collate([sample]).to(device)
            logits, trajectory, logits_per_step = model.forward_with_trajectory(
                batch.x, batch.edge_index, batch.target_idx, R
            )

            prob = torch.sigmoid(logits)[0].item()
            pred = 1.0 if prob > 0.5 else 0.0
            label = float(sample["label"])  # type: ignore[arg-type]
            is_correct = pred == label
            category = failure_category(is_correct, label, distance, R)

            if len(found[category]) >= per_category:
                continue

            found[category].append({
                "exampleId": f"d{distance}_R{R}_n{sample['num_nodes']}",
                "distance": distance,
                "R": R,
                "trainingMaxDistance": TRAINING_MAX_DISTANCE,
                "numNodes": sample["num_nodes"],
                "numEdges": sample["num_edges"],
                "graph": sample["graph"],
                "source": sample["source_idx"],
                "target": sample["target_idx"],
                "label": label,
                "prediction": pred,
                "probability": prob,
                "confidence": prob if pred == 1.0 else 1.0 - prob,
                "correct": is_correct,
                "category": category,
                "probabilityPerStep": [
                    torch.sigmoid(step)[0].item() for step in logits_per_step
                ],
                "targetStateNormPerStep": [
                    step[batch.target_idx[0]].norm().item() for step in trajectory
                ],
            })

    examples: List[Dict[str, object]] = []
    for key in wanted:
        examples.extend(found[key])
    return examples


def main() -> None:
    parser = argparse.ArgumentParser(description="Inference-depth experiment")
    parser.add_argument("--seed", type=int, default=1)
    parser.add_argument("--checkpoint", type=str, required=True)
    parser.add_argument("--max-R", type=int, default=10)
    parser.add_argument("--pairs-per-distance", type=int, default=40)
    parser.add_argument("--output-dir", type=str, default="results")
    parser.add_argument("--tag", type=str, default="")
    args = parser.parse_args()

    set_seed(args.seed)
    device = torch.device("cpu")

    checkpoint_path = Path(args.checkpoint)
    model, checkpoint = load_checkpoint(checkpoint_path, device)
    digest = file_hash(checkpoint_path)

    suffix = f"_{args.tag}" if args.tag else ""
    shared = checkpoint.get("shared_weights", True)
    depth_ceiling = model.max_inference_depth

    print(f"Checkpoint {checkpoint_path.name} (sha256 {digest})")
    print(f"Shared weights: {shared} | trained at R <= {checkpoint.get('max_R')}")
    print(f"Test distances {list(TEST_DISTANCES)} (training saw <= {TRAINING_MAX_DISTANCE})")

    test_samples = build_test_set(args.seed, args.pairs_per_distance)
    print(f"Test graphs: {len(test_samples)}\n")

    depth_results: List[Dict[str, object]] = []
    exact_results: List[Dict[str, object]] = []

    for R in range(1, args.max_R + 1):
        if R > depth_ceiling:
            # The unshared ablation genuinely has no weights for this depth.
            depth_results.append({
                "R": R,
                "unavailable": True,
                "reason": (
                    f"Model has only {depth_ceiling} update blocks. Without shared "
                    f"weights, inference depth cannot exceed training depth."
                ),
            })
            print(f"  R={R:2d} | unavailable (no shared weights)")
            continue

        result = evaluate_at_depth(model, test_samples, R, device)
        result["seed"] = args.seed
        depth_results.append(result)

        exact = exact_reference(test_samples, R)
        exact["R"] = R
        exact_results.append(exact)

        print(f"  R={R:2d} | learned {result['accuracy']:.3f} "
              f"(seen {result['accuracySeenDistances']:.3f}, "
              f"unseen {result['accuracyUnseenDistances']:.3f}) | "
              f"exact {exact['accuracy']:.3f} | "
              f"|z| {result['meanStateNorm']:.2f}")

    print("\nCollecting examples...")
    examples = collect_examples(model, test_samples, device)
    print(f"Examples: {len(examples)}")

    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    experiment = {
        "modelId": f"recurrent-gnn-seed{args.seed}{suffix}",
        "checkpointHash": digest,
        "seed": args.seed,
        "sharedWeights": shared,
        "maxInferenceDepth": depth_ceiling if shared else checkpoint.get("max_R"),
        "trainingMaxPathLength": TRAINING_MAX_DISTANCE,
        "trainingDistances": list(TRAIN_DISTANCES),
        "trainingMaxR": checkpoint.get("max_R"),
        "testDistances": list(TEST_DISTANCES),
        "config": checkpoint.get("config", {}),
        "trainingConfig": {
            "epochs": checkpoint.get("epochs"),
            "lr": checkpoint.get("lr"),
            "batchSize": checkpoint.get("batch_size"),
            "hiddenDim": checkpoint.get("hidden_dim"),
            "aggregation": checkpoint.get("aggregation"),
            "pairsPerDistance": checkpoint.get("pairs_per_distance"),
            "depthSchedule": "R ~ Uniform{1..max_R} per batch",
        },
        "inferenceDepthResults": depth_results,
        "exactReference": exact_results,
        "trainingLog": checkpoint.get("training_log", []),
        "evidence": {
            "evidenceType": "Precomputed result",
            "source": (
                "Local inference-depth sweep over a frozen toy checkpoint. "
                "Not a BDH or BDH-CQ measurement."
            ),
            "experimentId": f"depth-sweep-seed{args.seed}{suffix}",
        },
    }

    experiment_path = output_dir / f"learned_experiment_seed{args.seed}{suffix}.json"
    experiment_path.write_text(json.dumps(experiment, indent=2))
    print(f"Experiment: {experiment_path}")

    examples_path = output_dir / f"learned_examples_seed{args.seed}{suffix}.json"
    examples_path.write_text(json.dumps({
        "seed": args.seed,
        "checkpointHash": digest,
        "trainingMaxPathLength": TRAINING_MAX_DISTANCE,
        "examples": examples,
        "evidence": {
            "evidenceType": "Precomputed result",
            "source": "Frozen toy checkpoint evaluated on held-out generated graphs",
            "experimentId": f"examples-seed{args.seed}{suffix}",
        },
    }, indent=2))
    print(f"Examples: {examples_path}")


if __name__ == "__main__":
    main()
