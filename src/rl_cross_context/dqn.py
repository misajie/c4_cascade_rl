"""Double DQN with legal action mask (torch if available; dry-run CPU synthetic)."""

from __future__ import annotations

from collections import deque
from dataclasses import dataclass
from typing import Deque, Tuple

import numpy as np

from .baselines import BaselineState, run_episode
from .predictor import DeltaPredictor, build_xy_from_revealed
from .replay import AcquisitionQueue

try:
    import torch
    import torch.nn as nn
    import torch.nn.functional as F

    TORCH_AVAILABLE = True
except ImportError:  # pragma: no cover
    torch = None  # type: ignore
    nn = None  # type: ignore
    F = None  # type: ignore
    TORCH_AVAILABLE = False


if TORCH_AVAILABLE:

    class QNet(nn.Module):
        def __init__(self, state_dim: int, n_actions: int, hidden: int = 64):
            super().__init__()
            self.net = nn.Sequential(
                nn.Linear(state_dim, hidden),
                nn.ReLU(),
                nn.Linear(hidden, hidden),
                nn.ReLU(),
                nn.Linear(hidden, n_actions),
            )

        def forward(self, x: "torch.Tensor") -> "torch.Tensor":
            return self.net(x)


Transition = Tuple[np.ndarray, int, float, np.ndarray, bool, np.ndarray]


@dataclass
class DoubleDQNAgent:
    state_dim: int
    n_actions: int
    hidden: int = 64
    lr: float = 1e-3
    gamma: float = 0.99
    buffer_size: int = 2000
    batch_size: int = 32
    target_sync: int = 50
    device: str = "cpu"

    def __post_init__(self) -> None:
        if not TORCH_AVAILABLE:
            self.online = None
            self.target = None
            self.opt = None
            self.buffer: Deque[Transition] = deque(maxlen=self.buffer_size)
            self._steps = 0
            return
        self.online = QNet(self.state_dim, self.n_actions, self.hidden).to(self.device)
        self.target = QNet(self.state_dim, self.n_actions, self.hidden).to(self.device)
        self.target.load_state_dict(self.online.state_dict())
        self.opt = torch.optim.Adam(self.online.parameters(), lr=self.lr)
        self.buffer = deque(maxlen=self.buffer_size)
        self._steps = 0

    def encode_state(self, queue: AcquisitionQueue, state: BaselineState) -> np.ndarray:
        """Policy state: revealed mask + mean revealed source delta. H never included."""
        revealed_mask = (~queue.legal_mask).astype(np.float64)
        if state.revealed_ids:
            mean_delta = state.source_delta[state.revealed_ids].mean(axis=0)
        else:
            mean_delta = np.zeros(state.source_delta.shape[1], dtype=np.float64)
        return np.concatenate([revealed_mask, mean_delta]).astype(np.float64)

    def select(self, state_vec: np.ndarray, legal_mask: np.ndarray, eps: float, rng: np.random.Generator) -> int:
        legal = np.flatnonzero(legal_mask)
        if len(legal) == 0:
            raise RuntimeError("No legal actions")
        if (not TORCH_AVAILABLE) or rng.random() < eps:
            return int(rng.choice(legal))
        assert self.online is not None
        with torch.no_grad():
            q = self.online(torch.tensor(state_vec, dtype=torch.float32, device=self.device))
            q_np = q.cpu().numpy()
        q_np = np.where(legal_mask, q_np, -1e9)
        return int(np.argmax(q_np))

    def push(self, s, a, r, s2, done, legal2) -> None:
        self.buffer.append((s, a, r, s2, done, legal2))

    def train_step(self) -> float | None:
        if not TORCH_AVAILABLE or self.online is None or self.opt is None or self.target is None:
            return None
        if len(self.buffer) < self.batch_size:
            return None
        idx = np.random.randint(0, len(self.buffer), size=self.batch_size)
        batch = [self.buffer[i] for i in idx]
        s = torch.tensor(np.stack([b[0] for b in batch]), dtype=torch.float32, device=self.device)
        a = torch.tensor([b[1] for b in batch], dtype=torch.int64, device=self.device)
        r = torch.tensor([b[2] for b in batch], dtype=torch.float32, device=self.device)
        s2 = torch.tensor(np.stack([b[3] for b in batch]), dtype=torch.float32, device=self.device)
        done = torch.tensor([b[4] for b in batch], dtype=torch.float32, device=self.device)
        legal2 = torch.tensor(np.stack([b[5] for b in batch]), dtype=torch.bool, device=self.device)

        q = self.online(s).gather(1, a.view(-1, 1)).squeeze(1)
        with torch.no_grad():
            q2_online = self.online(s2)
            q2_online = q2_online.masked_fill(~legal2, -1e9)
            a2 = q2_online.argmax(dim=1)
            q2_target = self.target(s2).gather(1, a2.view(-1, 1)).squeeze(1)
            y = r + self.gamma * (1.0 - done) * q2_target
        loss = F.mse_loss(q, y)
        self.opt.zero_grad()
        loss.backward()
        self.opt.step()
        self._steps += 1
        if self._steps % self.target_sync == 0:
            self.target.load_state_dict(self.online.state_dict())
        return float(loss.item())


def train_dqn_synthetic(
    state: BaselineState,
    queue_template: AcquisitionQueue,
    budgets: list[int],
    episodes: int = 20,
    seed: int = 0,
    hidden: int = 64,
    lr: float = 1e-3,
    dry_run: bool = False,
) -> dict:
    """Train Double DQN on synthetic / provided arrays. dry_run skips torch opt if unavailable."""
    rng = np.random.default_rng(seed)
    n_actions = len(queue_template.items)
    state_dim = n_actions + state.source_delta.shape[1]
    agent = DoubleDQNAgent(
        state_dim=state_dim,
        n_actions=n_actions,
        hidden=hidden,
        lr=lr,
    )
    budget = max(budgets)
    eps = 1.0
    episode_returns = []
    losses = []

    for ep in range(episodes):
        queue = queue_template.clone()
        # reset revealed
        ep_state = BaselineState(
            source_delta=state.source_delta,
            target_delta=state.target_delta,
            revealed_ids=[],
            audit_ids=list(state.audit_ids),
            cond_to_idx=dict(state.cond_to_idx),
            rng=rng,
        )
        s = agent.encode_state(queue, ep_state)
        ep_ret = 0.0
        L_prev = None
        for t in range(min(budget, queue.n_remaining)):
            legal = queue.legal_mask
            a = agent.select(s, legal, eps=eps if TORCH_AVAILABLE and not dry_run else 1.0, rng=rng)
            cond = queue.reveal_index(a)
            cid = ep_state.cond_to_idx[cond]
            ep_state.revealed_ids.append(cid)
            pred = DeltaPredictor(alpha=1.0)
            X, Y = build_xy_from_revealed(ep_state.source_delta, ep_state.target_delta, ep_state.revealed_ids)
            pred.fit(X, Y)
            L = pred.loss(ep_state.source_delta[ep_state.audit_ids], ep_state.target_delta[ep_state.audit_ids])
            reward = 0.0 if L_prev is None else float(L_prev - L)
            L_prev = L
            ep_ret += reward
            s2 = agent.encode_state(queue, ep_state)
            done = queue.n_remaining == 0 or t == budget - 1
            agent.push(s, a, reward, s2, done, queue.legal_mask.copy())
            if TORCH_AVAILABLE and not dry_run:
                loss = agent.train_step()
                if loss is not None:
                    losses.append(loss)
            s = s2
        episode_returns.append(ep_ret)
        eps = max(0.05, eps * 0.95)

    # evaluate greedy-ish with random fallback via run_episode open_loop comparison hook
    return {
        "torch_available": TORCH_AVAILABLE,
        "dry_run": dry_run or not TORCH_AVAILABLE,
        "episode_returns": episode_returns,
        "mean_return": float(np.mean(episode_returns)) if episode_returns else 0.0,
        "n_losses": len(losses),
        "last_loss": float(losses[-1]) if losses else None,
        "state_dim": state_dim,
        "n_actions": n_actions,
    }
