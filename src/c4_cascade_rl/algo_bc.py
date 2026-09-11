"""L0 BC, L1 filtered BC, L2 AWAC."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, Optional, Sequence

import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader, TensorDataset

from c4_cascade_rl.models_hop import HopGNN


@dataclass
class BCConfig:
    lr: float = 1e-3
    epochs: int = 5
    batch_size: int = 32
    awac_eta: float = 1.0
    device: str = "cpu"


def _batch_loss_bc(
    model: HopGNN,
    node_ids: torch.Tensor,
    actions: torch.Tensor,
    legal_mask: torch.Tensor,
) -> torch.Tensor:
    logits = model.forward_pi(node_ids, legal_mask)
    return F.cross_entropy(logits, actions)


def train_bc(
    model: HopGNN,
    node_ids: torch.Tensor,
    actions: torch.Tensor,
    legal_mask: torch.Tensor,
    cfg: Optional[BCConfig] = None,
    weights: Optional[torch.Tensor] = None,
) -> Dict[str, float]:
    """Behavioral cloning (L0) or weighted BC (L1/L2)."""
    cfg = cfg or BCConfig()
    device = torch.device(cfg.device)
    model = model.to(device)
    node_ids = node_ids.to(device)
    actions = actions.to(device)
    legal_mask = legal_mask.to(device)
    if weights is None:
        weights = torch.ones(len(actions), device=device)
    else:
        weights = weights.to(device)

    ds = TensorDataset(node_ids, actions, legal_mask, weights)
    loader = DataLoader(ds, batch_size=cfg.batch_size, shuffle=True)
    opt = torch.optim.Adam(model.parameters(), lr=cfg.lr)

    last_loss = 0.0
    last_acc = 0.0
    model.train()
    for _ in range(cfg.epochs):
        total_loss = 0.0
        total_correct = 0
        total_n = 0
        for nb, ab, mb, wb in loader:
            logits = model.forward_pi(nb, mb)
            per = F.cross_entropy(logits, ab, reduction="none")
            loss = (per * wb).mean()
            opt.zero_grad()
            loss.backward()
            opt.step()
            total_loss += float(loss.item()) * len(ab)
            total_correct += int((logits.argmax(-1) == ab).sum().item())
            total_n += len(ab)
        last_loss = total_loss / max(total_n, 1)
        last_acc = total_correct / max(total_n, 1)
    return {"loss": last_loss, "accuracy": last_acc}


def train_l0(model: HopGNN, *tensors, cfg: Optional[BCConfig] = None) -> Dict[str, float]:
    return train_bc(model, *tensors, cfg=cfg, weights=None)


def train_l1_filtered(
    model: HopGNN,
    node_ids: torch.Tensor,
    actions: torch.Tensor,
    legal_mask: torch.Tensor,
    dir_ok: torch.Tensor,
    r_abl: torch.Tensor,
    cfg: Optional[BCConfig] = None,
) -> Dict[str, float]:
    """L1: keep rows where DIR ok & R_abl > 0."""
    keep = (dir_ok.bool()) & (r_abl > 0)
    if keep.sum() == 0:
        return {"loss": float("nan"), "accuracy": 0.0, "n": 0}
    return {
        **train_bc(
            model,
            node_ids[keep],
            actions[keep],
            legal_mask[keep],
            cfg=cfg,
        ),
        "n": int(keep.sum().item()),
    }


def train_l2_awac(
    model: HopGNN,
    node_ids: torch.Tensor,
    actions: torch.Tensor,
    legal_mask: torch.Tensor,
    advantages: torch.Tensor,
    eta: float = 1.0,
    cfg: Optional[BCConfig] = None,
) -> Dict[str, float]:
    """L2 AWAC: weight = exp(adv / eta), eta in {0.5, 1, 2}."""
    cfg = cfg or BCConfig()
    cfg.awac_eta = eta
    adv = advantages.float()
    w = torch.exp((adv - adv.mean()) / max(eta, 1e-6)).clamp(max=100.0)
    return {**train_bc(model, node_ids, actions, legal_mask, cfg=cfg, weights=w), "eta": eta}


def awac_eta_grid() -> Sequence[float]:
    return (0.5, 1.0, 2.0)
