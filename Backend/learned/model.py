"""
Shared-weight recurrent Graph Neural Network for Latent Loop Lab.

Architecture:
    - Encoder: maps node features [is_source, is_target] to latent states
    - Message passing: aggregate incoming neighbour messages
    - Recurrent update: same MLP/GRU cell reused at every step
    - Decoder: predicts reachability from target node's final state

Critical property: θ_0 = θ_1 = ... = θ_{R-1}
The model does NOT get new parameters at each reasoning step.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F


class SharedRecurrentGNN(nn.Module):
    """
    A small shared-weight recurrent GNN for graph reachability.

    At each recurrent step:
        1. Compute messages from incoming neighbours
        2. Aggregate messages per node
        3. Update node states using shared weights

    The same update block is reused for all R steps.
    """

    def __init__(
        self,
        input_dim: int = 2,
        hidden_dim: int = 32,
        message_dim: int = 32,
        num_classes: int = 1,
    ):
        super().__init__()

        self.hidden_dim = hidden_dim

        # Encoder: node features -> latent state
        self.encoder = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim),
        )

        # Message function: takes (sender_state, receiver_state) -> message
        self.message_fn = nn.Sequential(
            nn.Linear(hidden_dim * 2, message_dim),
            nn.ReLU(),
            nn.Linear(message_dim, message_dim),
        )

        # Update function: GRU-like gated update with shared weights
        # This is the recurrent cell reused at every step
        self.update_cell = nn.GRUCell(message_dim, hidden_dim)

        # Decoder: target node state -> reachability probability
        self.decoder = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim // 2),
            nn.ReLU(),
            nn.Linear(hidden_dim // 2, num_classes),
        )

    def encode(self, x: torch.Tensor) -> torch.Tensor:
        """
        Encode node features into initial latent states.

        Args:
            x: [num_nodes, input_dim] node features

        Returns:
            z0: [num_nodes, hidden_dim] initial latent states
        """
        return self.encoder(x)

    def message_passing(
        self,
        z: torch.Tensor,
        edge_index: torch.Tensor,
    ) -> torch.Tensor:
        """
        One step of message passing with shared weights.

        Args:
            z: [num_nodes, hidden_dim] current node states
            edge_index: [2, num_edges] directed edges (source, target)

        Returns:
            aggregated: [num_nodes, message_dim] aggregated messages
        """
        num_nodes = z.size(0)
        src, dst = edge_index  # src -> dst

        if src.numel() == 0:
            # No edges: zero messages
            return torch.zeros(num_nodes, self.message_fn[0].in_features // 2,
                             device=z.device)

        # Compute messages for each edge
        sender_states = z[src]   # [num_edges, hidden_dim]
        receiver_states = z[dst]  # [num_edges, hidden_dim]

        edge_input = torch.cat([sender_states, receiver_states], dim=-1)
        messages = self.message_fn(edge_input)  # [num_edges, message_dim]

        # Aggregate messages at destination nodes (mean aggregation)
        message_dim = messages.size(1)
        aggregated = torch.zeros(num_nodes, message_dim, device=z.device)
        counts = torch.zeros(num_nodes, 1, device=z.device)

        aggregated.scatter_add_(0, dst.unsqueeze(1).expand_as(messages), messages)
        counts.scatter_add_(0, dst.unsqueeze(1), torch.ones_like(dst.unsqueeze(1).float()))

        # Avoid division by zero
        counts = counts.clamp(min=1.0)
        aggregated = aggregated / counts

        return aggregated

    def recurrent_step(
        self,
        z: torch.Tensor,
        edge_index: torch.Tensor,
    ) -> torch.Tensor:
        """
        One recurrent update step. Same weights every time.

        Args:
            z: [num_nodes, hidden_dim] current states
            edge_index: [2, num_edges] directed edges

        Returns:
            z_next: [num_nodes, hidden_dim] updated states
        """
        # Get aggregated messages
        messages = self.message_passing(z, edge_index)

        # GRU update: shared weights at every step
        z_next = self.update_cell(messages, z)

        return z_next

    def forward(
        self,
        x: torch.Tensor,
        edge_index: torch.Tensor,
        target_idx: int,
        R: int,
    ) -> torch.Tensor:
        """
        Full forward pass: encode -> R recurrent steps -> decode target.

        Args:
            x: [num_nodes, input_dim] node features
            edge_index: [2, num_edges] directed edges
            target_idx: index of target node
            R: number of recurrent steps (inference depth)

        Returns:
            logit: [1] reachability logit for target node
        """
        # Encode initial states
        z = self.encode(x)

        # Apply shared recurrent update R times
        for _ in range(R):
            z = self.recurrent_step(z, edge_index)

        # Decode from target node
        target_state = z[target_idx].unsqueeze(0)  # [1, hidden_dim]
        logit = self.decoder(target_state)  # [1, 1]

        return logit.squeeze(-1)

    def forward_with_trajectory(
        self,
        x: torch.Tensor,
        edge_index: torch.Tensor,
        target_idx: int,
        R: int,
    ):
        """
        Forward pass that also returns the full trajectory for visualization.

        Returns:
            logit: [1] reachability logit
            trajectory: list of [num_nodes, hidden_dim] states at each step
        """
        z = self.encode(x)
        trajectory = [z.detach().clone()]

        for _ in range(R):
            z = self.recurrent_step(z, edge_index)
            trajectory.append(z.detach().clone())

        target_state = z[target_idx].unsqueeze(0)
        logit = self.decoder(target_state)

        return logit.squeeze(-1), trajectory
