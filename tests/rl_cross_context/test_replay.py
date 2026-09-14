import numpy as np

from rl_cross_context.replay import AcquisitionQueue, paired_queues


def test_reveal_without_replacement():
    q = AcquisitionQueue.from_conditions(["a", "b", "c"], seed=0)
    seen = []
    while q.n_remaining:
        c = q.reveal_next_random(np.random.default_rng(1))
        seen.append(c)
    assert sorted(seen) == ["a", "b", "c"]
    assert q.n_remaining == 0


def test_legal_mask_and_paired():
    qs = paired_queues(["a", "b", "c", "d"], seed=42, n_methods=3)
    assert qs[0].items == qs[1].items == qs[2].items
    qs[0].reveal(qs[0].items[0])
    assert qs[0].legal_mask.sum() == 3
    assert qs[1].legal_mask.sum() == 4  # independent clone
