"""Condition-level 80/10/10 + acquisition/reference cell split; write split_manifest.json."""

from __future__ import annotations

import json
from dataclasses import dataclass, asdict
from pathlib import Path
from typing import Any, Sequence

import numpy as np


@dataclass
class SplitManifest:
    seed: int
    conditions_train: list[str]
    conditions_val: list[str]
    conditions_test: list[str]
    acquisition_conditions: list[str]
    reference_conditions: list[str]
    audit_conditions: list[str]  # fixed H; never in policy state
    train_frac: float
    val_frac: float
    test_frac: float

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


def _split_indices(n: int, train_frac: float, val_frac: float, seed: int) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    rng = np.random.default_rng(seed)
    idx = rng.permutation(n)
    n_train = int(round(n * train_frac))
    n_val = int(round(n * val_frac))
    # ensure all non-empty when possible
    n_train = max(1, min(n_train, n - 2)) if n >= 3 else max(1, n - 2)
    n_val = max(1, min(n_val, n - n_train - 1)) if n >= 3 else 0
    train = idx[:n_train]
    val = idx[n_train : n_train + n_val]
    test = idx[n_train + n_val :]
    if len(test) == 0 and n >= 3:
        test = val[-1:]
        val = val[:-1]
    return train, val, test


def make_condition_splits(
    conditions: Sequence[str],
    control: str = "ctrl",
    train_frac: float = 0.8,
    val_frac: float = 0.1,
    test_frac: float = 0.1,
    acquisition_frac: float = 0.5,
    audit_frac: float = 0.25,
    seed: int = 0,
) -> SplitManifest:
    """Split non-control conditions; reserve fixed audit set H from val∪test."""
    assert abs(train_frac + val_frac + test_frac - 1.0) < 1e-6
    perts = [c for c in conditions if c != control]
    train_i, val_i, test_i = _split_indices(len(perts), train_frac, val_frac, seed)
    train = [perts[i] for i in train_i]
    val = [perts[i] for i in val_i]
    test = [perts[i] for i in test_i]

    rng = np.random.default_rng(seed + 7)
    # acquisition pool = subset of train; reference = remainder of train
    n_acq = max(1, int(round(len(train) * acquisition_frac)))
    n_acq = min(n_acq, max(1, len(train) - 1)) if len(train) > 1 else len(train)
    acq_idx = rng.permutation(len(train))
    acquisition = [train[i] for i in acq_idx[:n_acq]]
    reference = [train[i] for i in acq_idx[n_acq:]] or [train[0]]

    # fixed audit H from val+test (never in policy state)
    pool = val + test
    if not pool:
        pool = list(train[-max(1, len(train) // 4) :])
    n_audit = max(1, int(round(len(pool) * audit_frac)))
    n_audit = min(n_audit, len(pool))
    audit_idx = rng.permutation(len(pool))[:n_audit]
    audit = [pool[i] for i in audit_idx]

    return SplitManifest(
        seed=seed,
        conditions_train=train,
        conditions_val=val,
        conditions_test=test,
        acquisition_conditions=acquisition,
        reference_conditions=reference,
        audit_conditions=audit,
        train_frac=train_frac,
        val_frac=val_frac,
        test_frac=test_frac,
    )


def write_split_manifest(split: SplitManifest, path: str | Path) -> Path:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", encoding="utf-8") as f:
        json.dump(split.to_dict(), f, indent=2, ensure_ascii=False)
    return path


def load_split_manifest(path: str | Path) -> dict[str, Any]:
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)
