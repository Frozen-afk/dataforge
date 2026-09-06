"""
Training for the learned shared-weight recurrent GNN.

Training regime
---------------
* Distances 1..4 only. Distances 5..10 are never seen during training, which
  is what makes the later depth-extrapolation test meaningful.
* The inference depth ``R`` is resampled uniformly from ``{1, ..., max_R}`` for
  every batch. The same weights must therefore work at every depth, and the
  model cannot secretly specialise to one fixed number of iterations. Under a
  depth smaller than a sample's distance the instance is genuinely undecidable,
  so the loss-minimising output there is 0.5 -- the model is being taught
  calibration, not taught to bluff.
* Labels are balanced at every distance by construction (see ``dataset.py``),
  so accuracy is directly interpretable against a 0.5 chance baseline.

Outputs (architecture section 5.3)
----------------------------------
``results/model_seed<N>.pt``          frozen checkpoint with full config
``results/training_log_seed<N>.json`` loss and validation accuracy per epoch

Usage:
    python -m learned.train --seed 1 --epochs 40
    python -m learned.train --seed 1 --unshared        # ablation
"""

from __future__ import annotations

import argparse
import hashlib
import json
import random
import time
from pathlib import Path
from typing import Dict, List

import numpy as np
import torch
import torch.nn as nn

from learned.dataset import (
    TRAIN_DISTANCES,
    build_training_set,
    build_validation_set,
    iterate_batches,
)
from learned.model import build_model


def set_seed(seed: int) -> None:
    """Seed every source of randomness the run touches."""
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.use_deterministic_algorithms(False)


def checkpoint_hash(path: Path) -> str:
    """SHA-256 prefix of the checkpoint file, shown in the interface."""
    digest = hashlib.sha256()
    with open(path, "rb") as handle:
        for chunk in iter(lambda: handle.read(8192), b""):
            digest.update(chunk)
    return digest.hexdigest()[:16]


@torch.no_grad()
def evaluate_split(
    model: nn.Module,
    samples: List[Dict[str, object]],
    R: int,
    device: torch.device,
    batch_size: int = 64,
) -> Dict[str, float]:
    """Accuracy and mean loss on a split at one fixed inference depth."""
    model.eval()
    criterion = nn.BCEWithLogitsLoss(reduction="sum")

    correct = 0
    total = 0
    loss_sum = 0.0

    for batch in iterate_batches(samples, batch_size):
        batch = batch.to(device)
        logits = model(batch.x, batch.edge_index, batch.target_idx, R)

        loss_sum += criterion(logits, batch.y).item()
        preds = (torch.sigmoid(logits) > 0.5).float()
        correct += (preds == batch.y).sum().item()
        total += batch.num_graphs

    return {
        "accuracy": correct / max(total, 1),
        "loss": loss_sum / max(total, 1),
        "samples": total,
    }


def main() -> None:
    parser = argparse.ArgumentParser(description="Train the recurrent GNN")
    parser.add_argument("--seed", type=int, default=1)
    parser.add_argument("--epochs", type=int, default=40)
    parser.add_argument("--hidden-dim", type=int, default=32)
    parser.add_argument("--lr", type=float, default=3e-3)
    parser.add_argument("--batch-size", type=int, default=32)
    parser.add_argument("--max-R", type=int, default=4,
                        help="Maximum recurrent depth used during training")
    parser.add_argument("--aggregation", type=str, default="max",
                        choices=["max", "sum", "mean"])
    parser.add_argument("--pairs-per-distance", type=int, default=150)
    parser.add_argument("--unshared", action="store_true",
                        help="Train the unshared-weight ablation instead")
    parser.add_argument("--output-dir", type=str, default="results")
    parser.add_argument("--tag", type=str, default="",
                        help="Filename suffix, e.g. 'unshared'")
    args = parser.parse_args()

    set_seed(args.seed)
    device = torch.device("cpu")

    tag = args.tag or ("unshared" if args.unshared else "")
    suffix = f"_{tag}" if tag else ""

    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    print(f"Seed {args.seed} | shared weights: {not args.unshared} | "
          f"aggregation: {args.aggregation}")
    print(f"Training distances {list(TRAIN_DISTANCES)}, R sampled from 1..{args.max_R}")

    train_samples = build_training_set(args.seed, args.pairs_per_distance)
    val_samples = build_validation_set(args.seed)
    print(f"Train graphs: {len(train_samples)} | validation graphs: {len(val_samples)}")

    model = build_model(
        shared=not args.unshared,
        hidden_dim=args.hidden_dim,
        aggregation=args.aggregation,
        num_steps=args.max_R,
    ).to(device)

    config = model.config()
    print(f"Parameters: {config['numParameters']:,}")

    optimizer = torch.optim.Adam(model.parameters(), lr=args.lr)
    criterion = nn.BCEWithLogitsLoss()
    rng = random.Random(args.seed)

    training_log: List[Dict[str, float]] = []
    global_step = 0
    started = time.time()

    for epoch in range(1, args.epochs + 1):
        model.train()
        epoch_loss = 0.0
        batches = 0

        for batch in iterate_batches(
            train_samples, args.batch_size, shuffle=True, rng=rng
        ):
            batch = batch.to(device)

            # The unshared ablation has exactly max_R blocks and no freedom to
            # vary depth; that constraint is the ablation's finding.
            R = args.max_R if args.unshared else rng.randint(1, args.max_R)

            optimizer.zero_grad()
            logits = model(batch.x, batch.edge_index, batch.target_idx, R)
            loss = criterion(logits, batch.y)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimizer.step()

            epoch_loss += loss.item()
            batches += 1
            global_step += 1

        train_loss = epoch_loss / max(batches, 1)

        # Validate at the maximum training depth, where every training-range
        # instance is in principle decidable.
        val = evaluate_split(model, val_samples, args.max_R, device)

        training_log.append({
            "epoch": epoch,
            "step": global_step,
            "train_loss": train_loss,
            "val_accuracy": val["accuracy"],
            "val_loss": val["loss"],
        })

        if epoch == 1 or epoch % 5 == 0 or epoch == args.epochs:
            print(f"Epoch {epoch:3d}/{args.epochs} | loss {train_loss:.4f} | "
                  f"val acc {val['accuracy']:.3f}")

    elapsed = time.time() - started

    checkpoint_path = output_dir / f"model_seed{args.seed}{suffix}.pt"
    torch.save(
        {
            "model_state_dict": model.state_dict(),
            "config": config,
            "seed": args.seed,
            "epochs": args.epochs,
            "max_R": args.max_R,
            "hidden_dim": args.hidden_dim,
            "aggregation": args.aggregation,
            "shared_weights": not args.unshared,
            "training_distances": list(TRAIN_DISTANCES),
            "trainingMaxPathLength": max(TRAIN_DISTANCES),
            "lr": args.lr,
            "batch_size": args.batch_size,
            "pairs_per_distance": args.pairs_per_distance,
            "training_log": training_log,
        },
        checkpoint_path,
    )

    digest = checkpoint_hash(checkpoint_path)
    print(f"\nCheckpoint: {checkpoint_path}  (sha256 {digest}, {elapsed:.0f}s)")

    log_path = output_dir / f"training_log_seed{args.seed}{suffix}.json"
    log_path.write_text(json.dumps({
        "seed": args.seed,
        "checkpointHash": digest,
        "config": config,
        "sharedWeights": not args.unshared,
        "maxTrainingR": args.max_R,
        "trainingDistances": list(TRAIN_DISTANCES),
        "trainingMaxPathLength": max(TRAIN_DISTANCES),
        "epochs": args.epochs,
        "trainingSeconds": round(elapsed, 1),
        "trainingLog": training_log,
        "evidence": {
            "evidenceType": "Live computation",
            "source": "Local training run of the toy shared-weight recurrent GNN",
            "experimentId": f"train-seed{args.seed}{suffix}",
        },
    }, indent=2))
    print(f"Training log: {log_path}")


if __name__ == "__main__":
    main()
