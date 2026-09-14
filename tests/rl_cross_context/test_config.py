from pathlib import Path

from rl_cross_context.config import load_config, CrossContextConfig


def test_default_config():
    cfg = load_config()
    assert cfg.seed == 0
    assert cfg.cost_per_pert == 1.0
    assert len(cfg.budgets) >= 1


def test_load_yaml(tmp_path: Path):
    p = tmp_path / "c.yaml"
    p.write_text("seed: 7\nn_genes: 16\nbudgets: [2, 4]\n")
    cfg = load_config(p)
    assert cfg.seed == 7
    assert cfg.n_genes == 16
    assert cfg.budgets == [2, 4]


def test_overrides():
    cfg = load_config(seed=3, n_conditions=12)
    assert cfg.seed == 3
    assert cfg.n_conditions == 12
