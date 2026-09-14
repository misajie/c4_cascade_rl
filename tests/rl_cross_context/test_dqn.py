import numpy as np

from rl_cross_context.baselines import BaselineState
from rl_cross_context.data_manifest import build_synthetic_manifest
from rl_cross_context.dqn import DoubleDQNAgent, train_dqn_synthetic, TORCH_AVAILABLE
from rl_cross_context.predictor import condition_means, deltas_from_control
from rl_cross_context.replay import AcquisitionQueue
from rl_cross_context.splits import make_condition_splits


def _state_queue():
    man, arrays = build_synthetic_manifest(n_conditions=10, n_genes=6, seed=0)
    conds = arrays["conditions"]
    sp = make_condition_splits(conds, seed=0, acquisition_frac=0.6)
    n = len(conds)
    src_d = deltas_from_control(condition_means(arrays["source_X"], arrays["source_cond_ids"], n))
    tgt_d = deltas_from_control(condition_means(arrays["target_X"], arrays["target_cond_ids"], n))
    cond_to_idx = {c: i for i, c in enumerate(conds)}
    state = BaselineState(
        source_delta=src_d,
        target_delta=tgt_d,
        revealed_ids=[],
        audit_ids=[cond_to_idx[c] for c in sp.audit_conditions],
        cond_to_idx=cond_to_idx,
    )
    queue = AcquisitionQueue.from_conditions(sp.acquisition_conditions, seed=0)
    return state, queue


def test_dqn_dry_run_synthetic():
    state, queue = _state_queue()
    res = train_dqn_synthetic(state, queue, budgets=[2, 3], episodes=3, seed=0, dry_run=True, horizon=3)
    assert res["n_actions"] == len(queue.items)
    assert len(res["episode_returns"]) == 3
    assert res["horizon"] == 3
    agent = DoubleDQNAgent(n_actions=res["n_actions"], gene_dim=state.source_delta.shape[1])
    g = agent.encode_global(queue, state, budget_left=3, budget_max=3)
    cand = agent.candidate_features(queue, state)
    assert g.shape[0] == res["state_dim"]
    assert cand.shape == (len(queue.items), state.source_delta.shape[1])
    assert state.revealed_ids == []


def test_dqn_horizon1_myopic_flag():
    state, queue = _state_queue()
    res = train_dqn_synthetic(state, queue, budgets=[2], episodes=2, seed=0, dry_run=True, horizon=1)
    assert res["horizon"] == 1
