"""IQL: expectile V, Q backup, pi with BC KL to L0; Gate E entropy check."""

from __future__ import annotations

from copy import deepcopy
from dataclasses import dataclass
from typing import Dict, Optional

import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader, TensorDataset

from c4_cascade_rl.gates import gate_e
from c4_cascade_rl.models_hop import HopGNN


@dataclass
class IQLConfig:
    lr: float = 1e-3
    epochs: int = 5
    batch_size: int = 32
    expectile: float = 0.7
    gamma: float = 0.99
    beta: float = 3.0  # advantage temp for pi
    bc_kl_coef: float = 0.1
    device: str = "cpu"


def expectile_loss(diff: torch.Tensor, expectile: float) -> torch.Tensor:
    w = torch.where(diff > 0, expectile, 1.0 - expectile)
    return (w * diff.pow(2)).mean()


def train_iql(
    model: HopGNN,
    l0_model: HopGNN,
    node_ids: torch.Tensor,
    actions: torch.Tensor,
    rewards: torch.Tensor,
    next_node_ids: torch.Tensor,
    dones: torch.Tensor,
    legal_mask: torch.Tensor,
    cfg: Optional[IQLConfig] = None,
    runs_dir: str = "runs",
    week: str = "W4",
    run_gate_e: bool = True,
) -> Dict[str, float]:
    cfg = cfg or IQLConfig()
    device = torch.device(cfg.device)
    model = model.to(device)
    l0_model = l0_model.to(device)
    l0_model.eval()
    for p in l0_model.parameters():
        p.requires_grad_(False)

    node_ids = node_ids.to(device)
    actions = actions.to(device)
    rewards = rewards.to(device).float()
    next_node_ids = next_node_ids.to(device)
    dones = dones.to(device).float()
    legal_mask = legal_mask.to(device)

    ds = TensorDataset(node_ids, actions, rewards, next_node_ids, dones, legal_mask)
    loader = DataLoader(ds, batch_size=cfg.batch_size, shuffle=True)
    opt = torch.optim.Adam(model.parameters(), lr=cfg.lr)

    metrics = {"v_loss": 0.0, "q_loss": 0.0, "pi_loss": 0.0}
    model.train()
    for _ in range(cfg.epochs):
        for nb, ab, rb, nnb, db, mb in loader:
            with torch.no_grad():
                v_next = model.forward_v(nnb)
                q_tgt = rb + cfg.gamma * (1.0 - db) * v_next

            q = model.forward_q(nb, ab)
            q_loss = F.mse_loss(q, q_tgt)

            v = model.forward_v(nb)
            v_loss = expectile_loss(q.detach() - v, cfg.expectile)

            adv = (q.detach() - v.detach())
            logits = model.forward_pi(nb, mb)
            logp = F.log_softmax(logits, dim=-1).gather(1, ab.unsqueeze(-1)).squeeze(-1)
            # Advantage-weighted BC + KL to L0
            w = torch.exp(cfg.beta * adv).clamp(max=100.0)
            awbc = -(w * logp).mean()
            with torch.no_grad():
                l0_logits = l0_model.forward_pi(nb, mb)
                l0_logp = F.log_softmax(l0_logits, dim=-1)
            pi_logp_all = F.log_softmax(logits, dim=-1)
            kl = (l0_logp.exp() * (l0_logp - pi_logp_all)).sum(dim=-1).mean()
            pi_loss = awbc + cfg.bc_kl_coef * kl

            loss = q_loss + v_loss + pi_loss
            opt.zero_grad()
            loss.backward()
            opt.step()
            metrics = {
                "v_loss": float(v_loss.item()),
                "q_loss": float(q_loss.item()),
                "pi_loss": float(pi_loss.item()),
                "kl_to_l0": float(kl.item()),
            }

    model.eval()
    with torch.no_grad():
        ent = model.entropy(node_ids[: min(256, len(node_ids))], legal_mask[: min(256, len(legal_mask))])
        policy_entropy = float(ent.mean().item())
        # KL estimate
        logits = model.forward_pi(node_ids[:256], legal_mask[:256])
        l0_logits = l0_model.forward_pi(node_ids[:256], legal_mask[:256])
        p = F.softmax(l0_logits, dim=-1)
        kl = float((p * (F.log_softmax(l0_logits, dim=-1) - F.log_softmax(logits, dim=-1))).sum(-1).mean().item())

    metrics["policy_entropy"] = policy_entropy
    metrics["kl_to_l0"] = kl
    if run_gate_e:
        gate_e(policy_entropy=policy_entropy, kl_to_l0=kl, runs_dir=runs_dir, week=week)
    return metrics
