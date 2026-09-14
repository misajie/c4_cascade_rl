"""Build / write dataset_manifest.json (obs keys, conditions, control, counts).

Can operate on synthetic arrays; real h5ad path is optional and never downloaded here.
"""

from __future__ import annotations

import json
from dataclasses import dataclass, asdict, field
from pathlib import Path
from typing import Any

import numpy as np


MANIFEST_SCHEMA_VERSION = "1.0"


@dataclass
class ContextManifest:
    context: str
    n_cells: int
    n_genes: int
    n_conditions: int
    conditions: list[str]
    control_condition: str
    obs_keys: list[str]
    condition_counts: dict[str, int]
    gene_names: list[str] = field(default_factory=list)
    path: str | None = None


@dataclass
class DatasetManifest:
    schema_version: str
    source: ContextManifest
    target: ContextManifest
    notes: list[str] = field(default_factory=list)

    def to_dict(self) -> dict[str, Any]:
        return {
            "schema_version": self.schema_version,
            "source": asdict(self.source),
            "target": asdict(self.target),
            "notes": self.notes,
        }


def build_synthetic_context(
    context: str,
    n_conditions: int = 40,
    n_genes: int = 32,
    cells_per_condition: int = 8,
    control: str = "ctrl",
    seed: int = 0,
) -> tuple[ContextManifest, np.ndarray, np.ndarray, np.ndarray]:
    """Return (manifest, X [n_cells, n_genes], condition_ids, is_control)."""
    rng = np.random.default_rng(seed)
    conds = [control] + [f"{context}_g{i}" for i in range(1, n_conditions)]
    # allocate cells
    cond_ids = np.repeat(np.arange(len(conds)), cells_per_condition)
    n_cells = len(cond_ids)
    # simple additive effects
    base = rng.normal(0, 1, size=(n_genes,))
    effects = rng.normal(0, 0.5, size=(len(conds), n_genes))
    effects[0] = 0.0  # control
    X = base + effects[cond_ids] + rng.normal(0, 0.1, size=(n_cells, n_genes))
    is_control = cond_ids == 0
    counts = {c: int((cond_ids == i).sum()) for i, c in enumerate(conds)}
    gene_names = [f"gene_{j}" for j in range(n_genes)]
    man = ContextManifest(
        context=context,
        n_cells=n_cells,
        n_genes=n_genes,
        n_conditions=len(conds),
        conditions=conds,
        control_condition=control,
        obs_keys=["condition", "context"],
        condition_counts=counts,
        gene_names=gene_names,
        path=None,
    )
    return man, X.astype(np.float64), cond_ids.astype(np.int64), is_control


def build_synthetic_manifest(
    source: str = "k562",
    target: str = "rpe1",
    n_conditions: int = 40,
    n_genes: int = 32,
    seed: int = 0,
) -> tuple[DatasetManifest, dict[str, Any]]:
    src_m, src_X, src_cid, src_ctrl = build_synthetic_context(
        source, n_conditions=n_conditions, n_genes=n_genes, seed=seed
    )
    tgt_m, tgt_X, tgt_cid, tgt_ctrl = build_synthetic_context(
        target, n_conditions=n_conditions, n_genes=n_genes, seed=seed + 1
    )
    # align condition names for paired transfer (shared gene index; matched pert names)
    # keep control + shared g1..g{n-1} names on target renamed to match source gene labels
    shared = ["ctrl"] + [f"shared_g{i}" for i in range(1, n_conditions)]
    src_m.conditions = shared
    tgt_m.conditions = shared
    src_m.condition_counts = {c: int((src_cid == i).sum()) for i, c in enumerate(shared)}
    tgt_m.condition_counts = {c: int((tgt_cid == i).sum()) for i, c in enumerate(shared)}
    src_m.control_condition = "ctrl"
    tgt_m.control_condition = "ctrl"
    manifest = DatasetManifest(
        schema_version=MANIFEST_SCHEMA_VERSION,
        source=src_m,
        target=tgt_m,
        notes=["synthetic stub; replace with real h5ad inventory"],
    )
    arrays = {
        "source_X": src_X,
        "target_X": tgt_X,
        "source_cond_ids": src_cid,
        "target_cond_ids": tgt_cid,
        "source_is_control": src_ctrl,
        "target_is_control": tgt_ctrl,
        "conditions": shared,
    }
    return manifest, arrays


def write_manifest(manifest: DatasetManifest, path: str | Path) -> Path:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", encoding="utf-8") as f:
        json.dump(manifest.to_dict(), f, indent=2, ensure_ascii=False)
    return path


def load_manifest(path: str | Path) -> dict[str, Any]:
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)


def inventory_from_obs_table(
    context: str,
    conditions: list[str],
    condition_counts: dict[str, int],
    obs_keys: list[str],
    n_genes: int,
    control_condition: str = "ctrl",
    path: str | None = None,
    gene_names: list[str] | None = None,
) -> ContextManifest:
    """Build a ContextManifest from a read-only metadata inventory (no expression)."""
    n_cells = int(sum(condition_counts.values()))
    return ContextManifest(
        context=context,
        n_cells=n_cells,
        n_genes=n_genes,
        n_conditions=len(conditions),
        conditions=list(conditions),
        control_condition=control_condition,
        obs_keys=list(obs_keys),
        condition_counts=dict(condition_counts),
        gene_names=list(gene_names or []),
        path=path,
    )
