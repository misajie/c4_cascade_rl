"""Greedy hops → freeze → verbalize adapter; drop extra edges."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Callable, Dict, List, Optional, Sequence

from c4_cascade_rl.graph_env import Edge, GraphEnv
from c4_cascade_rl.schema import Hop, Stop, Trajectory


@dataclass
class InferResult:
    trajectory: Trajectory
    frozen_edges: List[Edge]
    verbalization: str
    dropped_edges: List[Edge]


def greedy_hops(
    env: GraphEnv,
    policy_fn: Callable[[GraphEnv], Optional[Edge]],
    max_hops: Optional[int] = None,
) -> List[Hop]:
    """Run greedy hop selection until stop or horizon."""
    env.reset(env.focus or None)
    hops: List[Hop] = []
    horizon = max_hops or env.horizon
    for _ in range(horizon):
        edge = policy_fn(env)
        if edge is None:
            break
        legal = env.legal_actions()
        if edge not in set(legal):
            break
        env.apply_hop(edge)
        hops.append(env.current[-1])
        if env.done:
            break
    return hops


def freeze_subgraph(hops: Sequence[Hop]) -> List[Edge]:
    return [(h.src, h.rel, h.dst, h.sign) for h in hops]


def drop_extra_edges(all_edges: Sequence[Edge], keep: Sequence[Edge]) -> List[Edge]:
    keep_set = set(keep)
    return [e for e in all_edges if e not in keep_set]


def verbalize_stub(frozen: Sequence[Edge], adapter: Optional[Callable[[str], str]] = None) -> str:
    lines = [f"{s}-{r}->{d}({sign:+d})" for s, r, d, sign in frozen]
    text = "Cascade: " + "; ".join(lines) if lines else "Cascade: (empty)"
    if adapter is not None:
        return adapter(text)
    return text


def infer_path_n(
    env: GraphEnv,
    policy_fn: Callable[[GraphEnv], Optional[Edge]],
    stop_de: int = 1,
    stop_dir: int = 1,
    verbalize_fn: Optional[Callable[[str], str]] = None,
) -> InferResult:
    """Path N: greedy hops → freeze → 8B verbalize adapter; drop extras."""
    hops = greedy_hops(env, policy_fn)
    frozen = freeze_subgraph(hops)
    dropped = drop_extra_edges(env.graph.edges, frozen)
    traj = Trajectory(hops=list(hops), stop=Stop(de=stop_de, direction=stop_dir), valid=True)
    verbal = verbalize_stub(frozen, adapter=verbalize_fn)
    return InferResult(
        trajectory=traj,
        frozen_edges=frozen,
        verbalization=verbal,
        dropped_edges=dropped,
    )


def random_legal_policy(env: GraphEnv) -> Optional[Edge]:
    legal = env.legal_actions()
    if not legal:
        return None
    return legal[0]  # greedy = first legal
