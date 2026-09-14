from pathlib import Path
import json

from rl_cross_context.splits import (
    build_split_from_manifest,
    gene_stem,
    make_condition_splits,
    write_split_manifest,
)


def test_gene_stem_parser():
    assert gene_stem("AARS+ctrl") == "AARS"
    assert gene_stem("ctrl") is None
    assert gene_stem("FOO+ctrl") == "FOO"


def test_condition_splits_disjoint(tmp_path: Path):
    conds = ["ctrl"] + [f"g{i}" for i in range(1, 21)]
    sp = make_condition_splits(conds, seed=0)
    train, val, test = set(sp.conditions_train), set(sp.conditions_val), set(sp.conditions_test)
    assert "ctrl" not in train | val | test
    assert train.isdisjoint(val) and train.isdisjoint(test) and val.isdisjoint(test)
    assert set(sp.acquisition_conditions).issubset(train)
    write_split_manifest(sp, tmp_path / "split_manifest.json")
    assert (tmp_path / "split_manifest.json").exists()


def test_gene_stem_no_leakage(tmp_path: Path):
    conds = ["ctrl"] + [f"GENE{i}+ctrl" for i in range(30)]
    sp = make_condition_splits(conds, seed=1)
    def stems(xs):
        return {gene_stem(c) for c in xs}
    assert stems(sp.conditions_train).isdisjoint(stems(sp.conditions_val) | stems(sp.conditions_test))
    assert stems(sp.conditions_val).isdisjoint(stems(sp.conditions_test))


def test_build_from_manifest_shape(tmp_path: Path):
    man = {
        "schema_version": "1.0",
        "source": {
            "context": "k562",
            "n_conditions": 5,
            "conditions": ["ctrl", "A+ctrl", "B+ctrl", "C+ctrl", "D+ctrl"],
            "control_condition": "ctrl",
        },
        "target": {
            "context": "rpe1",
            "n_conditions": 5,
            "conditions": ["ctrl", "A+ctrl", "B+ctrl", "C+ctrl", "E+ctrl"],
            "control_condition": "ctrl",
        },
        "condition_name_overlap": ["A+ctrl", "B+ctrl", "C+ctrl"],
        "n_condition_overlap": 3,
    }
    p = tmp_path / "dataset_manifest.json"
    p.write_text(json.dumps(man))
    sp = build_split_from_manifest(p, seed=0)
    assert sp.n_overlap == 3
    assert set(sp.conditions_train) | set(sp.conditions_val) | set(sp.conditions_test) <= {"A+ctrl", "B+ctrl", "C+ctrl"}
