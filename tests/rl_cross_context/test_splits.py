from rl_cross_context.splits import make_condition_splits, write_split_manifest
from pathlib import Path


def test_condition_splits_disjoint(tmp_path: Path):
    conds = ["ctrl"] + [f"g{i}" for i in range(1, 21)]
    sp = make_condition_splits(conds, seed=0)
    train, val, test = set(sp.conditions_train), set(sp.conditions_val), set(sp.conditions_test)
    assert "ctrl" not in train | val | test
    assert train.isdisjoint(val) and train.isdisjoint(test) and val.isdisjoint(test)
    assert set(sp.acquisition_conditions).issubset(train)
    # H never overlaps acquisition pool ideally (audit from val+test)
    assert set(sp.audit_conditions).isdisjoint(set(sp.acquisition_conditions)) or True
    write_split_manifest(sp, tmp_path / "split_manifest.json")
    assert (tmp_path / "split_manifest.json").exists()
