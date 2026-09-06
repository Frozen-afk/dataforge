"""
Shared-weight recurrent graph neural network -- the learned layer of Latent
Loop Lab.

This is an independent toy reimplementation written for teaching. It is not
BDH, it is not BDH-CQ, and it is not derived from any Pathway checkpoint.

Architecture (matching section 5.1 of the revised architecture document):

    z_v^(0)   = E_theta(x_v)                            encoder
    m_v^(r)   = AGG_{u in N^-(v)} M_theta(z_u^r, z_v^r) message
    z_v^(r+1) = F_theta(z_v^r, m_v^r)                   recurrent update
    y_hat_R   = G_theta(z_q^(R))                        decoder

The property that makes the inference-depth experiment meaningful is

    theta_0 = theta_1 = ... = theta_{R-1}

so increasing R buys computation, never parameters. The same forward pass
therefore accepts any R at inference, including depths never used in training.

:class:`UnsharedRecurrentGNN` is the ablation. It allocates a separate update
block per step, so its parameter count grows with depth and it simply has no
weights to run beyond its trained depth. That contrast is the point: recurrence
is what turns depth into a free inference-time dial.

Batching
--------
All modules operate on block-diagonal batches (see ``learned/dataset.py``):
``x`` is ``[total_nodes, 2]``, ``edge_index`` is ``[2, total_edges]`` in global
node coordinates, and ``target_idx`` is ``[B]``. Aggregation is a scatter over
destination nodes, so graphs in a batch never exchange messages.
"""

from __future__ import annotations

from typing import List, Tuple

import torch
import torch.nn as nn

AGGREGATIONS = ("max", "sum", "mean")


def aggregate_messages(
    messages: torch.Tensor,
    dst: torch.Tensor,
    num_nodes: int,
    aggregation: str,
) -> torch.Tensor:
    """
    Scatter edge messages onto their destination nodes.

    ``max`` is the default because reachability is a logical OR over incoming
    neighbours, and max is its differentiable analogue -- the same role the
    noisy-OR plays in the exact mechanism. ``sum`` and ``mean`` are provided so
    the choice can be ablated rather than asserted.
    """
    dim = messages.size(1)

    if aggregation == "max":
        out = torch.full((num_nodes, dim), float("-inf"), device=messages.device)
        out = out.scatter_reduce(
            0, dst.unsqueeze(1).expand(-1, dim), messages, reduce="amax",
            include_self=True,
        )
        # Nodes with no incoming edge receive no message.
        return torch.where(torch.isinf(out), torch.zeros_like(out), out)

    out = torch.zeros(num_nodes, dim, device=messages.device)
    out = out.index_add(0, dst, messages)

    if aggregation == "mean":
        counts = torch.zeros(num_nodes, 1, device=messages.device)
        counts = counts.index_add(
            0, dst, torch.ones(dst.size(0), 1, device=messages.device)
        )
        out = out / counts.clamp(min=1.0)

    return out


class RecurrentBlock(nn.Module):
    """One message-and-update block: the unit that recurrence repeats."""

    def __init__(self, hidden_dim: int, message_dim: int, aggregation: str = "max"):
        super().__init__()

        if aggregation not in AGGREGATIONS:
            raise ValueError(f"aggregation must be one of {AGGREGATIONS}")

        self.aggregation = aggregation

        self.message_fn = nn.Sequential(
            nn.Linear(hidden_dim * 2, message_dim),
            nn.ReLU(),
            nn.Linear(message_dim, message_dim),
        )
        self.update_cell = nn.GRUCell(message_dim, hidden_dim)

    def forward(self, z: torch.Tensor, edge_index: torch.Tensor) -> torch.Tensor:
        num_nodes = z.size(0)
        message_dim = self.update_cell.input_size

        if edge_index.numel() == 0:
            messages = torch.zeros(num_nodes, message_dim, device=z.device)
        else:
            src, dst = edge_index[0], edge_index[1]
            edge_input = torch.cat([z[src], z[dst]], dim=-1)
            messages = aggregate_messages(
                self.message_fn(edge_input), dst, num_nodes, self.aggregation
            )

        # Gated update keeps repeated application from diverging, which is what
        # allows the same block to be run far past its training depth.
        return self.update_cell(messages, z)


class SharedRecurrentGNN(nn.Module):
    """
    The main model. One update block, reused at every recurrent step.

    Args:
        input_dim: node feature width (2 for ``[is_source, is_target]``).
        hidden_dim: latent state width.
        message_dim: message width.
        aggregation: neighbour aggregation, one of ``max``/``sum``/``mean``.
    """

    def __init__(
        self,
        input_dim: int = 2,
        hidden_dim: int = 32,
        message_dim: int = 32,
        aggregation: str = "max",
    ):
        super().__init__()

        self.input_dim = input_dim
        self.hidden_dim = hidden_dim
        self.message_dim = message_dim
        self.aggregation = aggregation
        self.shared_weights = True

        self.encoder = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim),
        )
        self.block = RecurrentBlock(hidden_dim, message_dim, aggregation)
        self.decoder = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim // 2),
            nn.ReLU(),
            nn.Linear(hidden_dim // 2, 1),
        )

    # Sentinel for "no ceiling". Shared weights can be applied any number of
    # times, so this is only large enough to exceed any depth the lab offers.
    UNBOUNDED_DEPTH = 10_000

    @property
    def max_inference_depth(self) -> int:
        """Shared weights impose no depth ceiling."""
        return self.UNBOUNDED_DEPTH

    @property
    def depth_is_bounded(self) -> bool:
        """True when the model runs out of weights past a fixed depth."""
        return False

    def step(self, z: torch.Tensor, edge_index: torch.Tensor, r: int) -> torch.Tensor:
        """Apply the update block. ``r`` is ignored: the same weights every time."""
        return self.block(z, edge_index)

    def forward(
        self,
        x: torch.Tensor,
        edge_index: torch.Tensor,
        target_idx: torch.Tensor,
        R: int,
    ) -> torch.Tensor:
        """Returns one reachability logit per graph in the batch."""
        z = self.encoder(x)
        for r in range(R):
            z = self.step(z, edge_index, r)
        return self.decoder(z[target_idx]).squeeze(-1)

    def forward_with_trajectory(
        self,
        x: torch.Tensor,
        edge_index: torch.Tensor,
        target_idx: torch.Tensor,
        R: int,
    ) -> Tuple[torch.Tensor, List[torch.Tensor], List[torch.Tensor]]:
        """
        Forward pass that also returns what the interface needs to animate.

        Returns:
            logit: [B] final logit per graph.
            trajectory: latent states z^(0) .. z^(R).
            logits_per_step: the decoder read out after every step, so the page
                can show how the prediction firms up as depth increases.
        """
        z = self.encoder(x)
        trajectory = [z.detach().clone()]
        logits_per_step = [self.decoder(z[target_idx]).squeeze(-1).detach().clone()]

        for r in range(R):
            z = self.step(z, edge_index, r)
            trajectory.append(z.detach().clone())
            logits_per_step.append(
                self.decoder(z[target_idx]).squeeze(-1).detach().clone()
            )

        return self.decoder(z[target_idx]).squeeze(-1), trajectory, logits_per_step

    def config(self) -> dict:
        return {
            "modelClass": type(self).__name__,
            "sharedWeights": self.shared_weights,
            "inputDim": self.input_dim,
            "hiddenDim": self.hidden_dim,
            "messageDim": self.message_dim,
            "aggregation": self.aggregation,
            "numParameters": sum(p.numel() for p in self.parameters()),
        }


class UnsharedRecurrentGNN(SharedRecurrentGNN):
    """
    Ablation: a distinct update block per step, so no weights are shared.

    Parameter count scales with ``num_steps``, and the model cannot be run at
    any depth beyond ``num_steps`` because those layers do not exist. Comparing
    it against the shared model is the evidence that recurrence -- not merely
    "more layers" -- is what makes inference depth adjustable.
    """

    def __init__(
        self,
        input_dim: int = 2,
        hidden_dim: int = 32,
        message_dim: int = 32,
        aggregation: str = "max",
        num_steps: int = 4,
    ):
        super().__init__(input_dim, hidden_dim, message_dim, aggregation)

        self.shared_weights = False
        self.num_steps = num_steps

        # Replace the single shared block with one block per step.
        del self.block
        self.blocks = nn.ModuleList(
            [RecurrentBlock(hidden_dim, message_dim, aggregation) for _ in range(num_steps)]
        )

    @property
    def max_inference_depth(self) -> int:
        return self.num_steps

    @property
    def depth_is_bounded(self) -> bool:
        return True

    def step(self, z: torch.Tensor, edge_index: torch.Tensor, r: int) -> torch.Tensor:
        if r >= self.num_steps:
            raise ValueError(
                f"Unshared model has {self.num_steps} update blocks and cannot run "
                f"at depth {r + 1}. This is the ablation's point: without shared "
                f"weights, inference depth is fixed at training time."
            )
        return self.blocks[r](z, edge_index)

    def config(self) -> dict:
        base = super().config()
        base["numSteps"] = self.num_steps
        return base


def build_model(
    shared: bool = True,
    hidden_dim: int = 32,
    aggregation: str = "max",
    num_steps: int = 4,
) -> SharedRecurrentGNN:
    """Factory used by the training, evaluation and inference scripts."""
    if shared:
        return SharedRecurrentGNN(
            input_dim=2,
            hidden_dim=hidden_dim,
            message_dim=hidden_dim,
            aggregation=aggregation,
        )
    return UnsharedRecurrentGNN(
        input_dim=2,
        hidden_dim=hidden_dim,
        message_dim=hidden_dim,
        aggregation=aggregation,
        num_steps=num_steps,
    )
