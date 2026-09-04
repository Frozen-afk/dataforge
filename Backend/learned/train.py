"""
Training script for the shared-weight recurrent GNN.

Trains on paths of length <= 4 with R <= 4.
Saves a frozen checkpoint with a hash for reproducibility.

Usage:
    python -m learned.train --seed 1 --epochs 50 --hidden-dim 32
"""

import argparse
import hashlib
import json
import os
import random
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader

from learned.model import SharedRecurrentGNN
from learned.dataset import get_training_dataset, collate_fn


def set_seed(seed: int):
    """Set all random seeds for reproducibility."""
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def train_one_epoch(
    model: SharedRecurrentGNN,
    dataloader: DataLoader,
    optimizer: optim.Optimizer,
    criterion: nn.Module,
    device: torch.device,
    max_R: int = 4,
) -> float:
    """Train for one epoch. Returns average loss."""
    model.train()
    total_loss = 0.0
    num_samples = 0

    for batch in dataloader:
        for sample in batch:
            node_features = sample["node_features"].to(device)
            edge_index = sample["edge_index"].to(device)
            target_idx = sample["target_idx"]
            label = torch.tensor([sample["label"]], device=device)

            # Use R = min(path_length, max_R) during training
            path_length = sample["path_length"]
            R = min(path_length, max_R)

            optimizer.zero_grad()

            logit = model(node_features, edge_index, target_idx, R)
            loss = criterion(logit, label)

            loss.backward()
            optimizer.step()

            total_loss += loss.item()
            num_samples += 1

    return total_loss / max(num_samples, 1)


def evaluate_model(
    model: SharedRecurrentGNN,
    dataloader: DataLoader,
    device: torch.device,
    max_R: int = 4,
) -> float:
    """Evaluate accuracy. Returns accuracy as fraction."""
    model.eval()
    correct = 0
    total = 0

    with torch.no_grad():
        for batch in dataloader:
            for sample in batch:
                node_features = sample["node_features"].to(device)
                edge_index = sample["edge_index"].to(device)
                target_idx = sample["target_idx"]
                label = sample["label"]

                path_length = sample["path_length"]
                R = min(path_length, max_R)

                logit = model(node_features, edge_index, target_idx, R)
                pred = (torch.sigmoid(logit) > 0.5).float().item()

                if abs(pred - label) < 0.5:
                    correct += 1
                total += 1

    return correct / max(total, 1)


def compute_checkpoint_hash(filepath: str) -> str:
    """Compute SHA-256 hash of a checkpoint file."""
    sha256 = hashlib.sha256()
    with open(filepath, "rb") as f:
        for chunk in iter(lambda: f.read(8192), b""):
            sha256.update(chunk)
    return sha256.hexdigest()[:16]


def main():
    parser = argparse.ArgumentParser(description="Train shared recurrent GNN")
    parser.add_argument("--seed", type=int, default=1)
    parser.add_argument("--epochs", type=int, default=80)
    parser.add_argument("--hidden-dim", type=int, default=32)
    parser.add_argument("--lr", type=float, default=0.003)
    parser.add_argument("--max-R", type=int, default=4,
                        help="Maximum recurrent depth during training")
    parser.add_argument("--output-dir", type=str, default="results")
    args = parser.parse_args()

    set_seed(args.seed)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Device: {device}")
    print(f"Seed: {args.seed}")
    print(f"Hidden dim: {args.hidden_dim}")
    print(f"Training R: 1 to {args.max_R}")
    print(f"Training paths: length 1 to 4")
    print()

    # Create output directory
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    # Dataset
    train_dataset = get_training_dataset(seed=args.seed)
    train_loader = DataLoader(
        train_dataset,
        batch_size=16,
        shuffle=True,
        collate_fn=collate_fn,
    )

    # Model
    model = SharedRecurrentGNN(
        input_dim=2,
        hidden_dim=args.hidden_dim,
        message_dim=args.hidden_dim,
        num_classes=1,
    ).to(device)

    print(f"Model parameters: {sum(p.numel() for p in model.parameters()):,}")
    print()

    # Training setup
    optimizer = optim.Adam(model.parameters(), lr=args.lr)
    criterion = nn.BCEWithLogitsLoss()

    # Training loop
    training_log = []

    for epoch in range(1, args.epochs + 1):
        train_loss = train_one_epoch(
            model, train_loader, optimizer, criterion, device, args.max_R
        )

        training_log.append({
            "epoch": epoch,
            "train_loss": train_loss,
        })

        if epoch % 10 == 0 or epoch == 1:
            print(f"Epoch {epoch:3d}/{args.epochs} | Loss: {train_loss:.4f}")

    # Save checkpoint
    checkpoint_path = output_dir / f"model_seed{args.seed}.pt"
    torch.save({
        "model_state_dict": model.state_dict(),
        "hidden_dim": args.hidden_dim,
        "seed": args.seed,
        "epochs": args.epochs,
        "max_R": args.max_R,
        "training_log": training_log,
    }, checkpoint_path)

    checkpoint_hash = compute_checkpoint_hash(str(checkpoint_path))

    print()
    print(f"Checkpoint saved: {checkpoint_path}")
    print(f"Checkpoint hash: {checkpoint_hash}")

    # Save training log
    log_path = output_dir / f"training_log_seed{args.seed}.json"
    with open(log_path, "w") as f:
        json.dump({
            "seed": args.seed,
            "hidden_dim": args.hidden_dim,
            "epochs": args.epochs,
            "max_R": args.max_R,
            "checkpoint_hash": checkpoint_hash,
            "training_log": training_log,
        }, f, indent=2)

    print(f"Training log saved: {log_path}")


if __name__ == "__main__":
    main()
