"""Reward: R_task, path-order mute R_abl, hall flag; reward on STOP only."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Optional, Tuple

from c4_cascade_rl.schema import Trajectory


@dataclass
class RewardConfig:
    alpha: float = 0.5
    k_hops: int = 2
    lambda_abl: float = 1.0
    lambda_len: float = 0.05
    lambda_hall: float = 1.0


@dataclass
class RewardBreakdown:
    r_task: float
    r_muted: float
    r_abl: float
    hall: bool
    length: int
    r_total: float
    dir_ok: bool
    de_ok: bool


def _indicator(pred: int, gold: int) -> float:
    return 1.0 if pred == gold else 0.0


def task_reward(
    traj: Trajectory,
    gold_dir: int,
    gold_de: int,
    alpha: float = 0.5,
) -> Tuple[float, bool, bool]:
    """R_task = 1[DIR] + alpha * 1[DE]. Requires valid traj with stop."""
    if not traj.valid or traj.stop is None:
        return 0.0, False, False
    dir_ok = traj.stop.direction == gold_dir
    de_ok = traj.stop.de == gold_de
    r = _indicator(traj.stop.direction, gold_dir) + alpha * _indicator(traj.stop.de, gold_de)
    return float(r), dir_ok, de_ok


def muted_task_reward(
    traj: Trajectory,
    gold_dir: int,
    gold_de: int,
    k: int,
    alpha: float = 0.5,
) -> float:
    """R_task after deleting first k hops in path order (stop unchanged)."""
    muted = traj.muted(k)
    r, _, _ = task_reward(muted, gold_dir, gold_de, alpha=alpha)
    return r


def ablation_reward(r_task: float, r_muted: float) -> float:
    """R_abl = ReLU(R_task - R_muted)."""
    return max(0.0, float(r_task) - float(r_muted))


def detect_hallucination(traj: Trajectory, legal_edge_set: Optional[set] = None) -> bool:
    """Hall flag: hop not in legal edge set, or invalid traj."""
    if not traj.valid:
        return True
    if legal_edge_set is None:
        return False
    for h in traj.hops:
        key = (h.src, h.rel, h.dst, h.sign)
        key_nosign = (h.src, h.rel, h.dst)
        if key not in legal_edge_set and key_nosign not in legal_edge_set:
            return True
    return False


def compute_reward(
    traj: Trajectory,
    gold_dir: int,
    gold_de: int,
    cfg: Optional[RewardConfig] = None,
    legal_edge_set: Optional[set] = None,
    hall: Optional[bool] = None,
) -> RewardBreakdown:
    """Full reward on STOP only. Invalid traj → zeros + hall."""
    cfg = cfg or RewardConfig()
    if not traj.valid or traj.stop is None:
        return RewardBreakdown(
            r_task=0.0,
            r_muted=0.0,
            r_abl=0.0,
            hall=True,
            length=0,
            r_total=-float(cfg.lambda_hall),
            dir_ok=False,
            de_ok=False,
        )
    r_task, dir_ok, de_ok = task_reward(traj, gold_dir, gold_de, alpha=cfg.alpha)
    r_muted = muted_task_reward(traj, gold_dir, gold_de, k=cfg.k_hops, alpha=cfg.alpha)
    # Path-order mute does not change STOP labels, so r_muted == r_task unless
    # an external mute evaluator overrides. For schema-only mute, R_abl comes from
    # comparing against a separately scored muted prediction if provided via hall/
    # external. Here we also support flip detection via external muted scores.
    r_abl = ablation_reward(r_task, r_muted)
    if hall is None:
        hall = detect_hallucination(traj, legal_edge_set)
    length = traj.length
    r_total = (
        r_task
        + cfg.lambda_abl * r_abl
        - cfg.lambda_len * length
        - cfg.lambda_hall * (1.0 if hall else 0.0)
    )
    return RewardBreakdown(
        r_task=r_task,
        r_muted=r_muted,
        r_abl=r_abl,
        hall=bool(hall),
        length=length,
        r_total=r_total,
        dir_ok=dir_ok,
        de_ok=de_ok,
    )


def compute_reward_with_muted_preds(
    traj: Trajectory,
    gold_dir: int,
    gold_de: int,
    muted_pred_dir: int,
    muted_pred_de: int,
    cfg: Optional[RewardConfig] = None,
    legal_edge_set: Optional[set] = None,
    hall: Optional[bool] = None,
) -> RewardBreakdown:
    """R_abl using separately obtained muted STOP predictions (Gate B / collect).

    Path-order mute removes first k hops then re-infers STOP; that muted STOP
    is passed here so R_abl can be nonzero.
    """
    cfg = cfg or RewardConfig()
    if not traj.valid or traj.stop is None:
        return compute_reward(traj, gold_dir, gold_de, cfg=cfg, legal_edge_set=legal_edge_set, hall=True)
    r_task, dir_ok, de_ok = task_reward(traj, gold_dir, gold_de, alpha=cfg.alpha)
    r_muted = (
        _indicator(muted_pred_dir, gold_dir)
        + cfg.alpha * _indicator(muted_pred_de, gold_de)
    )
    r_abl = ablation_reward(r_task, r_muted)
    if hall is None:
        hall = detect_hallucination(traj, legal_edge_set)
    length = traj.length
    r_total = (
        r_task
        + cfg.lambda_abl * r_abl
        - cfg.lambda_len * length
        - cfg.lambda_hall * (1.0 if hall else 0.0)
    )
    return RewardBreakdown(
        r_task=r_task,
        r_muted=float(r_muted),
        r_abl=r_abl,
        hall=bool(hall),
        length=length,
        r_total=r_total,
        dir_ok=dir_ok,
        de_ok=de_ok,
    )


def path_order_mute_flips(
    r_task: float,
    r_muted: float,
    eps: float = 1e-9,
) -> bool:
    """True if muting first-k hops changes task reward (Gate B flip)."""
    return abs(r_task - r_muted) > eps
