"""Acquisition baselines: random, uncertainty, greedy VOI stub, contextual bandit stub, open-loop."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Callable, Protocol

import numpy as np

from .predictor import DeltaPredictor, build_xy_from_revealed
from .replay import AcquisitionQueue


class Policy(Protocol):
    def select(self, queue: AcquisitionQueue, state: "BaselineState") -> int:
        """Return index into queue.items (must be legal)."""
        ...


@dataclass
class BaselineState:
    source_delta: np.ndarray  # [n_cond, n_genes]
    target_delta: np.ndarray
    revealed_ids: list[int]
    audit_ids: list[int]
    cond_to_idx: dict[str, int]
    predictor: DeltaPredictor | None = None
    rng: np.random.Generator | None = None

    def legal_indices(self, queue: AcquisitionQueue) -> np.ndarray:
        return np.flatnonzero(queue.legal_mask)


def _ensure_rng(state: BaselineState) -> np.random.Generator:
    if state.rng is None:
        state.rng = np.random.default_rng(0)
    return state.rng


@dataclass
class RandomPolicy:
    def select(self, queue: AcquisitionQueue, state: BaselineState) -> int:
        legal = state.legal_indices(queue)
        rng = _ensure_rng(state)
        return int(rng.choice(legal))


@dataclass
class UncertaintyPolicy:
    """Pick legal condition with largest ||source_delta|| (proxy uncertainty)."""

    def select(self, queue: AcquisitionQueue, state: BaselineState) -> int:
        legal = state.legal_indices(queue)
        scores = np.linalg.norm(state.source_delta[legal], axis=1)
        return int(legal[int(np.argmax(scores))])


@dataclass
class GreedyVOIStub:
    """Greedy value-of-information stub: maximize predicted residual on audit via source proxy."""

    def select(self, queue: AcquisitionQueue, state: BaselineState) -> int:
        legal = state.legal_indices(queue)
        # score = alignment of candidate source delta with mean audit target residual direction
        audit = state.target_delta[state.audit_ids].mean(axis=0)
        if state.predictor is not None and len(state.revealed_ids) >= 1:
            try:
                X_a = state.source_delta[state.audit_ids]
                pred = state.predictor.predict(X_a)
                resid = state.target_delta[state.audit_ids] - pred
                audit = resid.mean(axis=0)
            except Exception:
                pass
        scores = state.source_delta[legal] @ audit
        return int(legal[int(np.argmax(scores))])


@dataclass
class ContextualBanditStub:
    """LinUCB-like stub over source_delta features (diagonal covariance)."""

    alpha: float = 0.5
    A_inv_diag: np.ndarray | None = None
    b: np.ndarray | None = None

    def select(self, queue: AcquisitionQueue, state: BaselineState) -> int:
        legal = state.legal_indices(queue)
        d = state.source_delta.shape[1]
        if self.A_inv_diag is None:
            self.A_inv_diag = np.ones(d)
            self.b = np.zeros(d)
        assert self.b is not None
        theta = self.A_inv_diag * self.b
        feats = state.source_delta[legal]
        mean = feats @ theta
        bonus = self.alpha * np.sqrt(np.sum(feats**2 * self.A_inv_diag, axis=1))
        scores = mean + bonus
        return int(legal[int(np.argmax(scores))])

    def update(self, feat: np.ndarray, reward: float) -> None:
        if self.A_inv_diag is None:
            self.A_inv_diag = np.ones(feat.shape[0])
            self.b = np.zeros(feat.shape[0])
        assert self.b is not None
        # diagonal Sherman-like: A_ii += x_i^2 ; b += r x
        self.A_inv_diag = 1.0 / (1.0 / self.A_inv_diag + feat**2 + 1e-8)
        self.b = self.b + reward * feat


@dataclass
class OpenLoopPolicy:
    """Fixed order: always reveal next in queue.items among legal (deterministic)."""

    def select(self, queue: AcquisitionQueue, state: BaselineState) -> int:
        legal = state.legal_indices(queue)
        return int(legal[0])


BASELINE_REGISTRY: dict[str, Callable[[], Policy]] = {
    "random": RandomPolicy,
    "uncertainty": UncertaintyPolicy,
    "greedy_voi": GreedyVOIStub,
    "contextual_bandit": ContextualBanditStub,
    "open_loop": OpenLoopPolicy,
}


def run_episode(
    policy: Policy,
    queue: AcquisitionQueue,
    state: BaselineState,
    budget: int,
    alpha: float = 1.0,
) -> dict:
    """Acquire up to `budget` conditions; reward = L_t - L_{t+1} on fixed audit H."""
    history = []
    L_prev = None
    bandit = policy if isinstance(policy, ContextualBanditStub) else None

    for t in range(budget):
        if queue.n_remaining == 0:
            break
        idx = policy.select(queue, state)
        cond = queue.reveal_index(idx)
        cid = state.cond_to_idx[cond]
        state.revealed_ids.append(cid)

        # refit predictor on revealed (need >=1)
        pred = DeltaPredictor(alpha=alpha)
        X, Y = build_xy_from_revealed(
            state.source_delta, state.target_delta, state.revealed_ids
        )
        pred.fit(X, Y)
        state.predictor = pred
        X_h = state.source_delta[state.audit_ids]
        Y_h = state.target_delta[state.audit_ids]
        L = pred.loss(X_h, Y_h)
        reward = 0.0 if L_prev is None else float(L_prev - L)
        if bandit is not None:
            bandit.update(state.source_delta[cid], reward)
        history.append(
            {
                "t": t,
                "condition": cond,
                "idx": idx,
                "loss": L,
                "reward": reward,
                "cost": 1.0,
            }
        )
        L_prev = L

    return {
        "history": history,
        "final_loss": history[-1]["loss"] if history else float("nan"),
        "revealed": list(queue.revealed),
        "cumulative_reward": float(sum(h["reward"] for h in history)),
    }
