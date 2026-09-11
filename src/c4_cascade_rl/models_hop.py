"""Small GAT/GNN hop policy: pi, Q, V + legal mask. CPU-friendly."""

from __future__ import annotations

from typing import Optional, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F


class HopGNN(nn.Module):
    """Lightweight self-attn hop encoder (~CPU-friendly; scale d_model for 2–10M)."""

    def __init__(
        self,
        n_nodes: int,
        n_rels: int = 16,
        n_actions: int = 64,
        d_model: int = 128,
        n_layers: int = 2,
        n_heads: int = 4,
    ):
        super().__init__()
        self.n_nodes = n_nodes
        self.n_actions = n_actions
        self.node_emb = nn.Embedding(max(n_nodes, 1), d_model)
        self.rel_emb = nn.Embedding(max(n_rels, 1), d_model)
        self.action_emb = nn.Embedding(max(n_actions, 1), d_model)
        heads = max(1, min(n_heads, d_model))
        while d_model % heads != 0:
            heads -= 1
        self.enc_layers = nn.ModuleList(
            [
                nn.TransformerEncoderLayer(
                    d_model=d_model,
                    nhead=heads,
                    dim_feedforward=d_model * 4,
                    batch_first=True,
                    activation="gelu",
                    dropout=0.0,
                )
                for _ in range(n_layers)
            ]
        )
        self.pi_head = nn.Linear(d_model, n_actions)
        self.q_head = nn.Linear(d_model * 2, 1)
        self.v_head = nn.Linear(d_model, 1)
        self.d_model = d_model

    def encode_state(self, node_ids: torch.Tensor) -> torch.Tensor:
        """node_ids: (B, L) long — mean pool after self-attn."""
        x = self.node_emb(node_ids.clamp(min=0))
        for layer in self.enc_layers:
            x = layer(x)
        return x.mean(dim=1)

    def forward_pi(
        self,
        node_ids: torch.Tensor,
        legal_mask: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        h = self.encode_state(node_ids)
        logits = self.pi_head(h)
        if legal_mask is not None:
            logits = logits.masked_fill(legal_mask <= 0, -1e9)
        return logits

    def forward_v(self, node_ids: torch.Tensor) -> torch.Tensor:
        h = self.encode_state(node_ids)
        return self.v_head(h).squeeze(-1)

    def forward_q(self, node_ids: torch.Tensor, action_ids: torch.Tensor) -> torch.Tensor:
        h = self.encode_state(node_ids)
        a = self.action_emb(action_ids.clamp(min=0))
        return self.q_head(torch.cat([h, a], dim=-1)).squeeze(-1)

    def sample(
        self,
        node_ids: torch.Tensor,
        legal_mask: Optional[torch.Tensor] = None,
        greedy: bool = False,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        logits = self.forward_pi(node_ids, legal_mask)
        if greedy:
            act = logits.argmax(dim=-1)
        else:
            dist = torch.distributions.Categorical(logits=logits)
            act = dist.sample()
        logp = F.log_softmax(logits, dim=-1).gather(1, act.unsqueeze(-1)).squeeze(-1)
        return act, logp

    def entropy(
        self,
        node_ids: torch.Tensor,
        legal_mask: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        logits = self.forward_pi(node_ids, legal_mask)
        dist = torch.distributions.Categorical(logits=logits)
        return dist.entropy()


def count_parameters(model: nn.Module) -> int:
    return sum(p.numel() for p in model.parameters())


def build_default_model(
    n_nodes: int = 100,
    n_actions: int = 32,
    d_model: int = 64,
) -> HopGNN:
    """CPU-friendly tiny model for tests."""
    return HopGNN(
        n_nodes=n_nodes,
        n_actions=n_actions,
        d_model=d_model,
        n_layers=2,
        n_heads=4,
    )
