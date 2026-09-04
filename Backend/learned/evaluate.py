"""
Inference-depth experiment for Latent Loop Lab.

Core question:
    With model parameters fixed, what changes when the model is allowed
    to apply its latent update more times at inference?

Experiment:
    - Freeze trained checkpoint
    - Sweep R = 1 to 10
    - Test on path lengths 1 to 10
    - Compare seen (<=4) vs unseen (>4) path lengths
    - Record accuracy and confidence

Usage:
    python -m learned.evaluate --seed 1 --checkpoint results/model_seed1.pt
"""

import argparse
import json
import random
from pathlib import Path

import numpy as np
import torch

from learned.model import SharedRecurrentGNN
from learned.dataset import PathReachabilityDataset, collate_fn


def set_seed(seed: int):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)


def evaluate_at_depth(
    model: SharedRecurrentGNN,
    dataset: PathReachabilityDataset,
    R: int,
    device: torch.device,
) -> dict:
    """
    Evaluate model at a fixed inference depth R.

    Returns accuracy, confidence, and per-path-length breakdown.
    """
    model.eval()

    results_by_length = {}
    total_correct = 0
    total_samples = 0
    total_confidence = 0.0

    with torch.no_grad():
        for sample in dataset:
            node_features = sample["node_features"].to(device)
            edge_index = sample["edge_index"].to(device)
            target_idx = sample["target_idx"]
            label = sample["label"]
            path_length = sample["path_length"]

            logit = model(node_features, edge_index, target_idx, R)
            prob = torch.sigmoid(logit).item()
            pred = 1.0 if prob > 0.5 else 0.0

            correct = abs(pred - label) < 0.5
            confidence = prob if pred == 1.0 else 1.0 - prob

            if path_length not in results_by_length:
                results_by_length[path_length] = {
                    "correct": 0,
                    "total": 0,
                    "confidence_sum": 0.0,
                }

            results_by_length[path_length]["correct"] += int(correct)
            results_by_length[path_length]["total"] += 1
            results_by_length[path_length]["confidence_sum"] += confidence

            total_correct += int(correct)
            total_samples += 1
            total_confidence += confidence

    # Compute per-length accuracy
    per_length = {}
    for length, data in sorted(results_by_length.items()):
        per_length[length] = {
            "accuracy": data["correct"] / max(data["total"], 1),
            "confidence": data["confidence_sum"] / max(data["total"], 1),
            "samples": data["total"],
        }

    return {
        "R": R,
        "accuracy": total_correct / max(total_samples, 1),
        "confidence": total_confidence / max(total_samples, 1),
        "total_samples": total_samples,
        "per_path_length": per_length,
    }


def run_inference_depth_experiment(
    model: SharedRecurrentGNN,
    seed: int,
    checkpoint_hash: str,
    training_max_path_length: int,
    device: torch.device,
    max_R: int = 10,
    test_lengths: list = None,
) -> dict:
    """
    Run the full inference-depth experiment.

    Sweeps R from 1 to max_R and evaluates on all test path lengths.
    """
    if test_lengths is None:
        test_lengths = list(range(1, 11))

    # Create test dataset
    test_dataset = PathReachabilityDataset(
        path_lengths=test_lengths,
        samples_per_length=50,
        seed=seed + 1000,  # Different seed from training
    )

    results = []

    for R in range(1, max_R + 1):
        result = evaluate_at_depth(model, test_dataset, R, device)
        result["seed"] = seed
        result["checkpoint_hash"] = checkpoint_hash
        result["training_max_path_length"] = training_max_path_length
        results.append(result)

        print(f"  R={R:2d} | Accuracy: {result['accuracy']:.3f} | "
              f"Confidence: {result['confidence']:.3f}")

    return {
        "modelId": f"shared-gnn-seed{seed}",
        "checkpointHash": checkpoint_hash,
        "trainingMaxPathLength": training_max_path_length,
        "seed": seed,
        "inference_depth_results": results,
    }


def get_failure_examples(
    model: SharedRecurrentGNN,
    seed: int,
    device: torch.device,
    num_examples: int = 5,
) -> list:
    """
    Collect successful and failed examples for the frontend.
    """
    test_dataset = PathReachabilityDataset(
        path_lengths=[3, 5, 7],
        samples_per_length=20,
        seed=seed + 2000,
    )

    examples = []
    model.eval()

    with torch.no_grad():
        for sample in test_dataset:
            if len(examples) >= num_examples * 2:
                break

            node_features = sample["node_features"].to(device)
            edge_index = sample["edge_index"].to(device)
            target_idx = sample["target_idx"]
            label = sample["label"]
            path_length = sample["path_length"]

            # Use R = path_length
            R = path_length

            logit, trajectory = model.forward_with_trajectory(
                node_features, edge_index, target_idx, R
            )
            prob = torch.sigmoid(logit).item()
            pred = 1.0 if prob > 0.5 else 0.0
            correct = abs(pred - label) < 0.5

            examples.append({
                "exampleId": f"seed{seed}_len{path_length}_R{R}",
                "pathLength": path_length,
                "R": R,
                "label": label,
                "prediction": pred,
                "confidence": prob,
                "correct": correct,
                "numNodes": node_features.size(0),
                "edgeIndex": edge_index.cpu().tolist(),
                "trajectoryNorms": [
                    z.norm(dim=-1).mean().item() for z in trajectory
                ],
            })

    return examples


def main():
    parser = argparse.ArgumentParser(description="Inference-depth experiment")
    parser.add_argument("--seed", type=int, default=1)
    parser.add_argument("--checkpoint", type=str, required=True)
    parser.add_argument("--max-R", type=int, default=10)
    parser.add_argument("--output-dir", type=str, default="results")
    args = parser.parse_args()

    set_seed(args.seed)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    print(f"Loading checkpoint: {args.checkpoint}")
    checkpoint = torch.load(args.checkpoint, map_location=device)

    hidden_dim = checkpoint["hidden_dim"]
    training_max_R = checkpoint["max_R"]

    # Rebuild model
    model = SharedRecurrentGNN(
        input_dim=2,
        hidden_dim=hidden_dim,
        message_dim=hidden_dim,
        num_classes=1,
    ).to(device)
    model.load_state_dict(checkpoint["model_state_dict"])
    model.eval()

    # Compute checkpoint hash
    import hashlib
    sha256 = hashlib.sha256()
    with open(args.checkpoint, "rb") as f:
        for chunk in iter(lambda: f.read(8192), b""):
            sha256.update(chunk)
    checkpoint_hash = sha256.hexdigest()[:16]

    print(f"Checkpoint hash: {checkpoint_hash}")
    print(f"Hidden dim: {hidden_dim}")
    print(f"Training max R: {training_max_R}")
    print(f"Device: {device}")
    print()

    # Run inference-depth experiment
    print("Running inference-depth sweep (R = 1 to 10)...")
    print()

    experiment = run_inference_depth_experiment(
        model=model,
        seed=args.seed,
        checkpoint_hash=checkpoint_hash,
        training_max_path_length=4,
        device=device,
        max_R=args.max_R,
    )

    # Get failure examples
    print()
    print("Collecting examples...")
    examples = get_failure_examples(model, args.seed, device)

    # Save results
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    experiment_path = output_dir / f"learned_experiment_seed{args.seed}.json"
    with open(experiment_path, "w") as f:
        json.dump(experiment, f, indent=2)
    print(f"Experiment saved: {experiment_path}")

    examples_path = output_dir / f"learned_examples_seed{args.seed}.json"
    with open(examples_path, "w") as f:
        json.dump(examples, f, indent=2)
    print(f"Examples saved: {examples_path}")


if __name__ == "__main__":
    main()
