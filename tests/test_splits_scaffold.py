"""Scaffold clustering on fake fingerprints."""
import numpy as np
import pytest

from c4_cascade_rl.splits import (
    e2b_tanimoto_holdout,
    scaffold_cluster_holdout,
    tanimoto_similarity,
    build_degree_split,
)


def test_tanimoto_identical():
    a = np.ones(32)
    assert tanimoto_similarity(a, a) == pytest.approx(1.0)


def test_scaffold_cluster_fake_fps():
    rng = np.random.default_rng(42)
    n = 60
    # Two distant clusters
    fps = np.zeros((n, 128), dtype=np.uint8)
    fps[:30, :64] = (rng.random((30, 64)) > 0.5).astype(np.uint8)
    fps[30:, 64:] = (rng.random((30, 64)) > 0.5).astype(np.uint8)
    drugs = [f"D{i}" for i in range(n)]
    res = scaffold_cluster_holdout(
        drugs, fps, distance_threshold=0.6, target_frac=0.2, min_holdout=5, recut_threshold=0.5
    )
    assert res["n_holdout"] >= 5
    assert res["n_train"] + res["n_holdout"] == n
    assert set(res["holdout_drugs"]).isdisjoint(res["train_drugs"])


def test_e2b_never_mixed_flag():
    rng = np.random.default_rng(0)
    fps = (rng.random((40, 64)) > 0.6).astype(np.uint8)
    drugs = [f"D{i}" for i in range(40)]
    res = e2b_tanimoto_holdout(drugs, fps, threshold=0.4)
    assert res["split"] == "E2b"
    assert "tanimoto_threshold" in res


def test_degree_bottom_quartile(tmp_path):
    deg = {f"N{i}": float(i) for i in range(20)}
    out = build_degree_split(deg, tmp_path / "degree.json")
    assert out["n_e2"] > 0
    assert (tmp_path / "degree.json").exists()


def test_rdkit_optional():
    pytest.importorskip("rdkit")
    from c4_cascade_rl.splits import morgan_fingerprints_from_smiles
    fps = morgan_fingerprints_from_smiles(["CCO", "c1ccccc1"], radius=2, n_bits=2048)
    assert fps.shape == (2, 2048)
