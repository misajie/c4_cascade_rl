import numpy as np

from rl_cross_context.baselines import BASELINE_REGISTRY, BaselineState, run_episode
from rl_cross_context.data_manifest import build_synthetic_manifest
from rl_cross_context.predictor import condition_means, deltas_from_control
from rl_cross_context.replay import paired_queues
from rl_cross_context.splits import make_condition_splits


def _state():
    man, arrays = build_synthetic_manifest(n_conditions=12, n_genes=8, seed=0)
    conds = arrays["conditions"]
    sp = make_condition_splits(conds, seed=0)
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
        rng=np.random.default_rng(0),
    )
    return state, sp


def test_all_baselines_paired():
    state, sp = _state()
    names = list(BASELINE_REGISTRY)
    queues = paired_queues(sp.acquisition_conditions, seed=0, n_methods=len(names))
    # paired: same initial order
    assert all(q.items == queues[0].items for q in queues)
    for name, q in zip(names, queues):
        st = BaselineState(
            source_delta=state.source_delta,
            target_delta=state.target_delta,
            revealed_ids=[],
            audit_ids=list(state.audit_ids),
            cond_to_idx=dict(state.cond_to_idx),
            rng=np.random.default_rng(0),
        )
        res = run_episode(BASELINE_REGISTRY[name](), q, st, budget=3)
        assert len(res["history"]) == 3
        assert all(h["cost"] == 1.0 for h in res["history"])
        # H not in revealed acquisition names necessarily checked via audit ids disjoint from acq
        for h in res["history"]:
            assert h["condition"] not in sp.audit_conditions
