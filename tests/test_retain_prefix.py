"""Retain prefix helpers; E1/E2b not averaged."""
from c4_cascade_rl.eval_ablation import (
    evaluate_retain_sweep,
    path_order_retain_mute,
    retain_prefix_hops,
)


def test_retain_prefix_values():
    assert retain_prefix_hops(4, 1.0) == 4
    assert retain_prefix_hops(4, 0.75) == 3
    assert retain_prefix_hops(4, 0.5) == 2
    assert retain_prefix_hops(4, 0.25) == 1
    assert retain_prefix_hops(4, 0.0) == 0


def test_path_order_delete_count():
    assert path_order_retain_mute(4, 1.0) == 0
    assert path_order_retain_mute(4, 0.0) == 4


def test_never_average_e1_e2b():
    records = []
    for split in ("E1", "E2b"):
        for i in range(10):
            records.append({
                "split": split,
                "gold_dir": 1,
                "gold_de": 1,
                "pred_dir": 1,
                "pred_de": 1,
                "muted_pred_dir": -1,
                "muted_pred_de": 0,
                "length": 4,
            })
    summary = evaluate_retain_sweep(records)
    assert "E1" in summary["splits"] and "E2b" in summary["splits"]
    assert summary["e1_e2b_averaged"] is None
    assert summary["e1_e2b_policy"] == "never_average"
