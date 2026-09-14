from pathlib import Path

from rl_cross_context.data_manifest import (
    build_synthetic_manifest,
    write_manifest,
    load_manifest,
    inventory_from_obs_table,
)


def test_synthetic_manifest_schema(tmp_path: Path):
    man, arrays = build_synthetic_manifest(n_conditions=10, n_genes=8, seed=1)
    assert man.schema_version == "1.0"
    assert man.source.n_genes == 8
    assert "ctrl" in man.source.conditions
    assert arrays["source_X"].shape[1] == 8
    path = write_manifest(man, tmp_path / "dataset_manifest.json")
    loaded = load_manifest(path)
    assert loaded["source"]["context"] == "k562"
    assert loaded["target"]["context"] == "rpe1"


def test_inventory_helper():
    m = inventory_from_obs_table(
        "k562",
        conditions=["ctrl", "g1"],
        condition_counts={"ctrl": 5, "g1": 5},
        obs_keys=["condition"],
        n_genes=4,
    )
    assert m.n_cells == 10
    assert m.n_conditions == 2
