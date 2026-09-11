"""Gate B table and other gates."""
import json
from pathlib import Path

from c4_cascade_rl.gates import gate_a1, gate_a2, gate_b, gate_c, gate_d, gate_e, A1_TARGETS


def test_gate_b_pass_fail(tmp_path):
    # table: n_flips vs min_flip
    cases = [
        (2999, 3000, False),
        (3000, 3000, True),
        (5000, 3000, True),
        (0, 3000, False),
    ]
    for n_flips, min_flip, expect in cases:
        r = gate_b(n_flips=n_flips, min_flip=min_flip, runs_dir=tmp_path, week="W2", k_hops=2)
        assert r["pass"] is expect, (n_flips, min_flip)
        assert (tmp_path / "W2" / "gate_B.json").exists()
    # k=1 recommendation when fail
    r = gate_b(n_flips=10, min_flip=3000, runs_dir=tmp_path, week="W2", k_hops=2)
    assert r["action"] == "retry_k1"


def test_gate_a1_within_1pct(tmp_path):
    counts = dict(A1_TARGETS)
    assert gate_a1(counts, runs_dir=tmp_path)["pass"]
    bad = dict(counts)
    bad["A549"] = 0
    assert gate_a1(bad, runs_dir=tmp_path)["pass"] is False


def test_gate_a2(tmp_path):
    assert gate_a2(30, True, runs_dir=tmp_path)["pass"]
    assert gate_a2(10, True, runs_dir=tmp_path)["pass"] is False


def test_gate_c_d_e(tmp_path):
    assert gate_c(0.9, 0.1, runs_dir=tmp_path)["pass"]
    d = gate_d(0.1, 0.5, runs_dir=tmp_path)
    assert d["pass"] and d["shaping_unlocked"]
    assert gate_e(0.5, 1.0, runs_dir=tmp_path)["pass"]
