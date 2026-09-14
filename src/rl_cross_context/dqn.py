"""Double DQN with legal action mask and candidate-wise scorer (optional torch)."""

from __future__ import annotations

from collections import deque
from dataclasses import dataclass
from typing import Deque, Tuple

import numpy as np

from .baselines import BaselineState
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

    class CandidateQNet(nn.Module):
        """Shared MLP over [global_state || candidate_feat] → scalar Q."""

        def __init__(self, global_dim: int, cand_dim: int, hidden: int = 64):
            super().__init__()
            self.net = nn.Sequential(
                nn.Linear(global_dim + cand_dim, hidden),
                nn.ReLU(),
                nn.Linear(hidden, hidden),
                nn.ReLU(),
                nn.Linear(hidden, 1),
            )

        def forward(self, global_s: "torch.Tensor", cand: "torch.Tensor") -> "torch.Tensor":
            # global_s: [B, G], cand: [B, A, C] → Q: [B, A]
            b, a, c = cand.shape
            g = global_s.unsqueeze(1).expand(b, a, global_s.shape[-1])
            x = torch.cat([g, cand], dim=-1)
            return self.net(x).squeeze(-1)


# Transition: global_s, cand_feats, action, reward, global_s2, cand2, done, legal2
Transition = Tuple[np.ndarray, np.ndarray, int, float, np.ndarray, np.ndarray, bool, np.ndarray]


@dataclass
class DoubleDQNAgent:
    n_actions: int
    gene_dim: int
    hidden: int = 64
    lr: float = 1e-3
    gamma: float = 0.99
    buffer_size: int = 2000
    batch_size: int = 32
    target_sync: int = 50
    device: str = "cpu"
    horizon: int = 3  # ≥3 uses bootstrapped target; 1 is myopic (gamma applied once)

    def __post_init__(self) -> None:
        self.global_dim = self.gene_dim + 2  # mean_delta + frac_revealed + budget_left_norm
        self.cand_dim = self.gene_dim
        self.buffer: Deque[Transition] = deque(maxlen=self.buffer_size)
        self._steps = 0
        if not TORCH_AVAILABLE:
            self.online = None
            self.target = None
            self.opt = None
            return
        self.online = CandidateQNet(self.global_dim, self.cand_dim, self.hidden).to(self.device)
        self.target = CandidateQNet(self.global_dim, self.cand_dim, self.hidden).to(self.device)
        self.target.load_state_dict(self.online.state_dict())
        self.opt = torch.optim.Adam(self.online.parameters(), lr=self.lr)

    def candidate_features(self, queue: AcquisitionQueue, state: BaselineState) -> np.ndarray:
        """[n_actions, gene_dim] source deltas in queue.items order — no gene-ID embedding."""
        feats = np.zeros((len(queue.items), state.source_delta.shape[1]), dtype=np.float64)
        for i, cond in enumerate(queue.items):
            cid = state.cond_to_idx[cond]
            feats[i] = state.source_delta[cid]
        return feats

    def encode_global(
        self,
        queue: AcquisitionQueue,
        state: BaselineState,
        budget_left: int,
        budget_max: int,
    ) -> np.ndarray:
        """Policy global state — never includes audit H labels."""
        if state.revealed_ids:
            mean_delta = state.source_delta[state.revealed_ids].mean(axis=0)
        else:
            mean_delta = np.zeros(state.source_delta.shape[1], dtype=np.float64)
        frac = len(state.revealed_ids) / max(1, len(queue.items))
        bnorm = budget_left / max(1, budget_max)
        return np.concatenate([mean_delta, [frac, bnorm]]).astype(np.float64)

    # backward-compat alias used by older tests
    def encode_state(self, queue: AcquisitionQueue, state: BaselineState) -> np.ndarray:
        return self.encode_global(queue, state, budget_left=0, budget_max=1)

    def select(
        self,
        global_s: np.ndarray,
        cand: np.ndarray,
        legal_mask: np.ndarray,
        eps: float,
        rng: np.random.Generator,
    ) -> int:
        legal = np.flatnonzero(legal_mask)
        if len(legal) == 0:
            raise RuntimeError("No legal actions")
        if (not TORCH_AVAILABLE) or rng.random() < eps:
            return int(rng.choice(legal))
        assert self.online is not None
        with torch.no_grad():
            g = torch.tensor(global_s[None], dtype=torch.float32, device=self.device)
            c = torch.tensor(cand[None], dtype=torch.float32, device=self.device)
            q = self.online(g, c).cpu().numpy()[0]
        q = np.where(legal_mask, q, -1e9)
        return int(np.argmax(q))

    def push(self, g, cand, a, r, g2, cand2, done, legal2) -> None:
        self.buffer.append((g, cand, a, r, g2, cand2, done, legal2))

    def train_step(self) -> float | None:
        if not TORCH_AVAILABLE or self.online is None or self.opt is None or self.target is None:
            return None
        if len(self.buffer) < self.batch_size:
            return None
        idx = np.random.randint(0, len(self.buffer), size=self.batch_size)
        batch = [self.buffer[i] for i in idx]
        g = torch.tensor(np.stack([b[0] for b in batch]), dtype=torch.float32, device=self.device)
        cand = torch.tensor(np.stack([b[1] for b in batch]), dtype=torch.float32, device=self.device)
        a = torch.tensor([b[2] for b in batch], dtype=torch.int64, device=self.device)
        r = torch.tensor([b[3] for b in batch], dtype=torch.float32, device=self.device)
        g2 = torch.tensor(np.stack([b[4] for b in batch]), dtype=torch.float32, device=self.device)
        cand2 = torch.tensor(np.stack([b[5] for b in batch]), dtype=torch.float32, device=self.device)
        done = torch.tensor([b[6] for b in batch], dtype=torch.float32, device=self.device)
        legal2 = torch.tensor(np.stack([b[7] for b in batch]), dtype=torch.bool, device=self.device)

        q_all = self.online(g, cand)
        q = q_all.gather(1, a.view(-1, 1)).squeeze(1)
        gamma = 0.0 if self.horizon <= 1 else self.gamma
        with torch.no_grad():
            q2_online = self.online(g2, cand2).masked_fill(~legal2, -1e9)
            a2 = q2_online.argmax(dim=1)
            q2_target = self.target(g2, cand2).gather(1, a2.view(-1, 1)).squeeze(1)
            y = r + gamma * (1.0 - done) * q2_target
        loss = F.mse_loss(q, y)
        self.opt.zero_grad()
        loss.backward()
        self.opt.step()
        self._steps += 1
        if self._steps % self.target_sync == 0:
            self.target.load_state_dict(self.online.state_dict())
        return float(loss.item())



def evaluate_policy_risk_curve(
    agent: "DoubleDQNAgent",
    state: BaselineState,
    queue_template: AcquisitionQueue,
    budgets: list[int],
    alpha: float = 1.0,
    seed: int = 0,
) -> dict:
    """Greedy (eps=0) rollout; score fixed audit H after each acquire. Never trains."""
    from .eval_curves import risk_at_budgets

    rng = np.random.default_rng(seed + 999)
    queue = queue_template.clone()
    ep_state = BaselineState(
        source_delta=state.source_delta,
        target_delta=state.target_delta,
        revealed_ids=[],
        audit_ids=list(state.audit_ids),
        cond_to_idx=dict(state.cond_to_idx),
        rng=rng,
    )
    cand = agent.candidate_features(queue, ep_state)
    budget = max(budgets)
    budget_left = budget
    g = agent.encode_global(queue, ep_state, budget_left, budget)
    losses: list[float] = []
    history = []
    L_prev = None
    ep_ret = 0.0
    for t in range(min(budget, queue.n_remaining)):
        a = agent.select(g, cand, queue.legal_mask, eps=0.0, rng=rng)
        cond = queue.reveal_index(a)
        cid = ep_state.cond_to_idx[cond]
        ep_state.revealed_ids.append(cid)
        pred = DeltaPredictor(alpha=alpha)
        X, Y = build_xy_from_revealed(
            ep_state.source_delta, ep_state.target_delta, ep_state.revealed_ids
        )
        pred.fit(X, Y)
        L = pred.loss(
            ep_state.source_delta[ep_state.audit_ids],
            ep_state.target_delta[ep_state.audit_ids],
        )
        reward = 0.0 if L_prev is None else float(L_prev - L)
        L_prev = L
        ep_ret += reward
        losses.append(float(L))
        history.append({"t": t, "condition": cond, "loss": float(L), "reward": reward})
        budget_left = budget - (t + 1)
        g = agent.encode_global(queue, ep_state, budget_left, budget)
    rb = risk_at_budgets(losses, budgets)
    return {
        "risk_at_budget": {str(k): float(v) for k, v in rb.items()},
        "final_loss": float(losses[-1]) if losses else float("nan"),
        "cumulative_reward": float(ep_ret),
        "history": history,
    }


def train_dqn_synthetic(
    state: BaselineState,
    queue_template: AcquisitionQueue,
    budgets: list[int],
    episodes: int = 200,
    seed: int = 0,
    hidden: int = 64,
    lr: float = 1e-3,
    dry_run: bool = False,
    horizon: int = 3,
    alpha: float = 1.0,
    eval_every: int = 25,
    eps_start: float = 1.0,
    eps_end: float = 0.05,
    eps_decay: float = 0.995,
    buffer_size: int = 10000,
) -> dict:
    """Train candidate-wise Double DQN. dry_run / no-torch → epsilon-greedy random.

    Also dumps greedy risk–budget eval curves periodically and at the end.
    """
    rng = np.random.default_rng(seed)
    n_actions = len(queue_template.items)
    gene_dim = state.source_delta.shape[1]
    agent = DoubleDQNAgent(
        n_actions=n_actions,
        gene_dim=gene_dim,
        hidden=hidden,
        lr=lr,
        horizon=horizon,
        buffer_size=buffer_size,
    )
    budget = max(budgets)
    eps = float(eps_start)
    episode_returns = []
    losses = []
    eval_snapshots = []

    for ep in range(episodes):
        queue = queue_template.clone()
        ep_state = BaselineState(
            source_delta=state.source_delta,
            target_delta=state.target_delta,
            revealed_ids=[],
            audit_ids=list(state.audit_ids),
            cond_to_idx=dict(state.cond_to_idx),
            rng=rng,
        )
        cand = agent.candidate_features(queue, ep_state)
        budget_left = budget
        g = agent.encode_global(queue, ep_state, budget_left, budget)
        ep_ret = 0.0
        L_prev = None
        for t in range(min(budget, queue.n_remaining)):
            legal = queue.legal_mask
            a = agent.select(
                g,
                cand,
                legal,
                eps=eps if TORCH_AVAILABLE and not dry_run else 1.0,
                rng=rng,
            )
            cond = queue.reveal_index(a)
            cid = ep_state.cond_to_idx[cond]
            ep_state.revealed_ids.append(cid)
            pred = DeltaPredictor(alpha=alpha)
            X, Y = build_xy_from_revealed(
                ep_state.source_delta, ep_state.target_delta, ep_state.revealed_ids
            )
            pred.fit(X, Y)
            L = pred.loss(
                ep_state.source_delta[ep_state.audit_ids],
                ep_state.target_delta[ep_state.audit_ids],
            )
            reward = 0.0 if L_prev is None else float(L_prev - L)
            L_prev = L
            ep_ret += reward
            budget_left = budget - (t + 1)
            g2 = agent.encode_global(queue, ep_state, budget_left, budget)
            done = queue.n_remaining == 0 or t == budget - 1
            agent.push(g, cand, a, reward, g2, cand, done, queue.legal_mask.copy())
            if TORCH_AVAILABLE and not dry_run:
                loss = agent.train_step()
                if loss is not None:
                    losses.append(loss)
            g = g2
        episode_returns.append(ep_ret)
        eps = max(float(eps_end), eps * float(eps_decay))
        # periodic greedy eval on audit risk–budget (no gradient)
        if eval_every > 0 and ((ep + 1) % eval_every == 0 or ep == 0 or ep == episodes - 1):
            snap = evaluate_policy_risk_curve(
                agent, state, queue_template, budgets, alpha=alpha, seed=seed + ep
            )
            snap["episode"] = ep + 1
            snap["eps"] = float(eps)
            # drop bulky step history from periodic snaps except final
            if ep != episodes - 1:
                snap.pop("history", None)
            eval_snapshots.append(snap)

    final_eval = evaluate_policy_risk_curve(
        agent, state, queue_template, budgets, alpha=alpha, seed=seed + 10_000
    )
    final_eval["episode"] = episodes
    final_eval["eps"] = 0.0

    return {
        "torch_available": TORCH_AVAILABLE,
        "dry_run": dry_run or not TORCH_AVAILABLE,
        "horizon": horizon,
        "episodes": episodes,
        "eval_every": eval_every,
        "episode_returns": episode_returns,
        "mean_return": float(np.mean(episode_returns)) if episode_returns else 0.0,
        "mean_return_last20": float(np.mean(episode_returns[-20:])) if episode_returns else 0.0,
        "n_losses": len(losses),
        "last_loss": float(losses[-1]) if losses else None,
        "state_dim": agent.global_dim,
        "n_actions": n_actions,
        "cand_dim": agent.cand_dim,
        "eval_snapshots": eval_snapshots,
        "eval_risk_at_budget": final_eval.get("risk_at_budget"),
        "eval_final_loss": final_eval.get("final_loss"),
        "eval_cumulative_reward": final_eval.get("cumulative_reward"),
        "eval_history": final_eval.get("history"),
    }
