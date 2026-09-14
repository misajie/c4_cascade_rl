"""Budgets, seeds, and paths from YAML."""

from __future__ import annotations

from dataclasses import dataclass, field, asdict
from pathlib import Path
from typing import Any

import yaml


@dataclass
class CrossContextConfig:
    seed: int = 0
    budgets: list[int] = field(default_factory=lambda: [5, 10, 20, 40, 80])
    train_frac: float = 0.8
    val_frac: float = 0.1
    test_frac: float = 0.1
    acquisition_frac: float = 0.5  # of train conditions available to acquire
    audit_frac: float = 0.25  # fixed audit set H fraction of held-out conditions
    cost_per_pert: float = 1.0
    ridge_alpha: float = 1.0
    dqn_hidden: int = 64
    dqn_lr: float = 1e-3
    dqn_gamma: float = 0.99
    dqn_batch_size: int = 32
    dqn_buffer_size: int = 2000
    dqn_target_sync: int = 50
    dqn_episodes: int = 20
    dqn_eps_start: float = 1.0
    dqn_eps_end: float = 0.05
    dqn_eps_decay: float = 0.95
    n_genes: int = 32  # synthetic default
    n_conditions: int = 40  # synthetic default
    source_context: str = "k562"
    target_context: str = "rpe1"
    paths: dict[str, str] = field(default_factory=dict)
    output_dir: str = "runs/rl_cross_context"

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


def load_config(path: str | Path | None = None, **overrides: Any) -> CrossContextConfig:
    data: dict[str, Any] = {}
    if path is not None:
        with open(path, "r", encoding="utf-8") as f:
            loaded = yaml.safe_load(f) or {}
        if not isinstance(loaded, dict):
            raise ValueError(f"Config root must be a mapping: {path}")
        data.update(loaded)
    # flatten nested common keys
    if "budgets" in data and isinstance(data["budgets"], dict):
        data["budgets"] = data["budgets"].get("values", data.get("budgets"))
    cfg = CrossContextConfig(**{k: v for k, v in data.items() if k in CrossContextConfig.__dataclass_fields__})
    for k, v in overrides.items():
        if hasattr(cfg, k):
            setattr(cfg, k, v)
    return cfg


def default_paths() -> dict[str, str]:
    return {
        "k562": "~/VGAE/data/replogle_k562_essential/perturb_processed.h5ad",
        "rpe1": "~/VGAE/data/replogle_rpe1_essential/perturb_processed.h5ad",
        "norman": "~/VGAE/data/norman19/perturb_processed.h5ad",
        "output_dir": "~/vcrl/c4_cascade_rl/runs/rl_cross_context/",
    }
