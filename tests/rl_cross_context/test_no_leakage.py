"""Policies must not read audit H labels when selecting."""

from __future__ import annotations

import numpy as np
import pytest

from rl_cross_context.baselines import BASELINE_REGISTRY, BaselineState, GreedyVOIStub
from rl_cross_context.data_manifest import build_synthetic_manifest
from rl_cross_context.episode_data import assert_no_audit_in_acquisition, load_episode_bundle
from rl_cross_context.predictor import DeltaPredictor, condition_means, deltas_from_control
from rl_cross_context.replay import AcquisitionQueue
from rl_cross_context.splits import make_condition_splits


def _bundle_state(tmp_path):
    man, arrays = build_synthetic_manifest(n_conditions=16, n_genes=8, seed=0)
    conds = arrays["conditions"]
    sp = make_condition_splits(conds, seed=0)
    n = len(conds)
    src = deltas_from_control(condition_means(arrays["source_X"], arrays["source_cond_ids"], n))
    tgt = deltas_from_control(condition_means(arrays["target_X"], arrays["target_cond_ids"], n))
    cond_to_idx = {c: i for i, c in enumerate(conds)}
    state = BaselineState(
        source_delta=src,
        target_delta=tgt,
        revealed_ids=[],
        audit_ids=[cond_to_idx[c] for c in sp.audit_conditions],
        cond_to_idx=cond_to_idx,
        rng=np.random.default_rng(0),
    )
    queue = AcquisitionQueue.from_conditions(sp.acquisition_conditions, seed=0)
    return state, queue, sp


def test_audit_disjoint_acquisition(tmp_path):
    bundle = load_episode_bundle(tmp_path, seed=0, n_conditions=20, n_genes=6, prefer_synthetic=True)
    assert_no_audit_in_acquisition(bundle)


def test_greedy_voi_ignores_poisoned_audit():
    state, queue, sp = _bundle_state(None)
    # poison audit targets; VOI must not change selection vs zeroed audit
    legal0 = GreedyVOIStub().select(queue, state)
    state.target_delta = state.target_delta.copy()
    state.target_delta[state.audit_ids] = 1e6
    legal1 = GreedyVOIStub().select(queue, state)
    assert legal0 == legal1


def test_baselines_never_reveal_audit():
    state, queue, sp = _bundle_state(None)
    from rl_cross_context.baselines import run_episode

    for name, factory in BASELINE_REGISTRY.items():
        q = queue.clone()
        st = BaselineState(
            source_delta=state.source_delta,
            target_delta=state.target_delta,
            revealed_ids=[],
            audit_ids=list(state.audit_ids),
            cond_to_idx=dict(state.cond_to_idx),
            rng=np.random.default_rng(1),
        )
        res = run_episode(factory(), q, st, budget=2)
        for h in res["history"]:
            assert h["condition"] not in sp.audit_conditions, name
