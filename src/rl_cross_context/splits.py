"""Condition-level splits by gene stem from dataset_manifest.json."""

from __future__ import annotations

import json
import re
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any, Sequence

import random

_CTRL_SUFFIX = re.compile(r"\+ctrl$", re.IGNORECASE)


def gene_stem(condition: str, control: str = "ctrl") -> str | None:
    """Map `GENE+ctrl` → GENE; bare control → None; other strings kept as stem."""
    c = str(condition).strip()
    if c.lower() == str(control).lower():
        return None
    m = _CTRL_SUFFIX.search(c)
    if m:
        return c[: m.start()]
    if "+" in c:
        return c.split("+", 1)[0]
    return c


@dataclass
class SplitManifest:
    seed: int
    control: str
    conditions_train: list[str]
    conditions_val: list[str]
    conditions_test: list[str]
    acquisition_conditions: list[str]
    reference_conditions: list[str]
    audit_conditions: list[str]
    gene_stems_train: list[str]
    gene_stems_val: list[str]
    gene_stems_test: list[str]
    n_overlap: int
    train_frac: float
    val_frac: float
    test_frac: float
    source_context: str | None = None
    target_context: str | None = None
    cell_split_note: str = (
        "cell-level acquisition/reference indices to be filled by CPU job reading h5ad"
    )
    notes: list[str] = field(default_factory=list)

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


def _split_stems(
    stems: Sequence[str],
    train_frac: float,
    val_frac: float,
    seed: int,
) -> tuple[list[str], list[str], list[str]]:
    stems = sorted(set(stems))
    n = len(stems)
    rng = random.Random(seed)
    idx = list(range(n))
    rng.shuffle(idx)
    n_train = int(round(n * train_frac))
    n_val = int(round(n * val_frac))
    if n >= 3:
        n_train = max(1, min(n_train, n - 2))
        n_val = max(1, min(n_val, n - n_train - 1))
    else:
        n_train = max(1, n - 1)
        n_val = 0
    train_i = idx[:n_train]
    val_i = idx[n_train : n_train + n_val]
    test_i = idx[n_train + n_val :]
    if len(test_i) == 0 and n >= 3:
        test_i = val_i[-1:]
        val_i = val_i[:-1]
    train = [stems[i] for i in train_i]
    val = [stems[i] for i in val_i]
    test = [stems[i] for i in test_i]
    return train, val, test


def conditions_for_stems(conditions: Sequence[str], stems: Sequence[str], control: str) -> list[str]:
    stem_set = set(stems)
    out = []
    for c in conditions:
        g = gene_stem(c, control)
        if g is not None and g in stem_set:
            out.append(c)
    return sorted(out)


def make_gene_stem_splits(
    conditions: Sequence[str],
    control: str = "ctrl",
    train_frac: float = 0.8,
    val_frac: float = 0.1,
    test_frac: float = 0.1,
    acquisition_frac: float = 0.5,
    audit_frac: float = 0.25,
    seed: int = 0,
    source_context: str | None = None,
    target_context: str | None = None,
    notes: list[str] | None = None,
) -> SplitManifest:
    """Split by gene stem so all conditions of a gene stay in one partition."""
    assert abs(train_frac + val_frac + test_frac - 1.0) < 1e-6
    stems = []
    for c in conditions:
        g = gene_stem(c, control)
        if g is not None:
            stems.append(g)
    train_s, val_s, test_s = _split_stems(stems, train_frac, val_frac, seed)
    train = conditions_for_stems(conditions, train_s, control)
    val = conditions_for_stems(conditions, val_s, control)
    test = conditions_for_stems(conditions, test_s, control)

    rng = random.Random(seed + 7)
    n_acq = max(1, int(round(len(train) * acquisition_frac)))
    if len(train) > 1:
        n_acq = min(n_acq, len(train) - 1)
    acq_idx = list(range(len(train)))
    rng.shuffle(acq_idx)
    acquisition = [train[i] for i in acq_idx[:n_acq]]
    reference = [train[i] for i in acq_idx[n_acq:]] or (list(train[:1]) if train else [])

    pool = val + test
    if not pool:
        pool = list(train[-max(1, len(train) // 4) :]) if train else []
    n_audit = max(1, int(round(len(pool) * audit_frac))) if pool else 0
    n_audit = min(n_audit, len(pool))
    pool_idx = list(range(len(pool)))
    rng.shuffle(pool_idx)
    audit = [pool[i] for i in pool_idx[:n_audit]] if pool else []

    # leakage check
    def stems_of(cs: list[str]) -> set[str]:
        return {gene_stem(c, control) for c in cs if gene_stem(c, control)}

    assert stems_of(train).isdisjoint(stems_of(val) | stems_of(test))
    assert stems_of(val).isdisjoint(stems_of(test))

    return SplitManifest(
        seed=seed,
        control=control,
        conditions_train=train,
        conditions_val=val,
        conditions_test=test,
        acquisition_conditions=acquisition,
        reference_conditions=reference,
        audit_conditions=audit,
        gene_stems_train=sorted(train_s),
        gene_stems_val=sorted(val_s),
        gene_stems_test=sorted(test_s),
        n_overlap=len({c for c in conditions if gene_stem(c, control)}),
        train_frac=train_frac,
        val_frac=val_frac,
        test_frac=test_frac,
        source_context=source_context,
        target_context=target_context,
        notes=list(notes or []),
    )


def load_dataset_manifest(path: str | Path) -> dict[str, Any]:
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)


def build_split_from_manifest(
    manifest_path: str | Path,
    train_frac: float = 0.8,
    val_frac: float = 0.1,
    test_frac: float = 0.1,
    acquisition_frac: float = 0.5,
    audit_frac: float = 0.25,
    seed: int = 0,
) -> SplitManifest:
    raw = load_dataset_manifest(manifest_path)
    source = raw.get("source") or {}
    target = raw.get("target") or {}
    control = source.get("control_condition") or target.get("control_condition") or "ctrl"
    overlap = raw.get("condition_name_overlap")
    if not overlap:
        sc = set(source.get("conditions") or [])
        tc = set(target.get("conditions") or [])
        overlap = sorted(sc & tc)
    # exclude control from overlap list used for stems
    overlap = [c for c in overlap if str(c).lower() != str(control).lower()]
    notes = [
        f"split on intersection n={len(overlap)}",
        f"source_n_conditions={source.get('n_conditions')}",
        f"target_n_conditions={target.get('n_conditions')}",
        "gene-stem partition: same stem never crosses train/val/test",
    ]
    return make_gene_stem_splits(
        overlap,
        control=control,
        train_frac=train_frac,
        val_frac=val_frac,
        test_frac=test_frac,
        acquisition_frac=acquisition_frac,
        audit_frac=audit_frac,
        seed=seed,
        source_context=source.get("context"),
        target_context=target.get("context"),
        notes=notes,
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


# Back-compat alias used by older tests
def make_condition_splits(*args, **kwargs):
    return make_gene_stem_splits(*args, **kwargs)
