"""Load episode arrays: synthetic or condition_deltas.npz + split_manifest."""

from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np

from .baselines import BaselineState
from .data_manifest import build_synthetic_manifest, write_manifest
from .predictor import condition_means, deltas_from_control
from .replay import AcquisitionQueue
from .splits import make_condition_splits, write_split_manifest


DELTA_NPZ_KEYS = ("conditions", "source_delta", "target_delta")


@dataclass
class EpisodeBundle:
    conditions: list[str]
    source_delta: np.ndarray
    target_delta: np.ndarray
    control: str
    cond_to_idx: dict[str, int]
    audit_ids: list[int]
    acquisition_conditions: list[str]
    split: dict[str, Any]
    source: str = "synthetic"


def save_condition_deltas(
    path: str | Path,
    conditions: list[str],
    source_delta: np.ndarray,
    target_delta: np.ndarray,
    control: str = "ctrl",
    meta: dict[str, Any] | None = None,
) -> Path:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    payload = {
        "conditions": np.asarray(conditions, dtype=object),
        "source_delta": np.asarray(source_delta, dtype=np.float64),
        "target_delta": np.asarray(target_delta, dtype=np.float64),
        "control": np.asarray(control),
    }
    if meta:
        payload["meta_json"] = np.asarray(json.dumps(meta))
    np.savez_compressed(path, **payload)
    return path


def load_condition_deltas(path: str | Path) -> dict[str, Any]:
    path = Path(path)
    z = np.load(path, allow_pickle=True)
    for k in DELTA_NPZ_KEYS:
        if k not in z:
            raise KeyError(f"{path} missing key {k}")
    conditions = [str(c) for c in z["conditions"].tolist()]
    control = str(z["control"]) if "control" in z.files else "ctrl"
    return {
        "conditions": conditions,
        "source_delta": np.asarray(z["source_delta"], dtype=np.float64),
        "target_delta": np.asarray(z["target_delta"], dtype=np.float64),
        "control": control,
    }


def _split_dict_from_obj(sp) -> dict[str, Any]:
    if hasattr(sp, "to_dict"):
        return sp.to_dict()
    return dict(sp)


def load_episode_bundle(
    out_dir: str | Path,
    *,
    seed: int = 0,
    n_conditions: int = 40,
    n_genes: int = 32,
    source_context: str = "k562",
    target_context: str = "rpe1",
    train_frac: float = 0.8,
    val_frac: float = 0.1,
    test_frac: float = 0.1,
    acquisition_frac: float = 0.5,
    audit_frac: float = 0.25,
    arrays_path: str | Path | None = None,
    split_path: str | Path | None = None,
    prefer_synthetic: bool = False,
) -> EpisodeBundle:
    out = Path(out_dir)
    out.mkdir(parents=True, exist_ok=True)
    npz = Path(arrays_path) if arrays_path else out / "condition_deltas.npz"
    split_p = Path(split_path) if split_path else out / "split_manifest.json"

    if not prefer_synthetic and npz.exists():
        arr = load_condition_deltas(npz)
        conditions = arr["conditions"]
        source_delta = arr["source_delta"]
        target_delta = arr["target_delta"]
        control = arr["control"]
        src_tag = "npz"
    else:
        man, arrays = build_synthetic_manifest(
            source=source_context,
            target=target_context,
            n_conditions=n_conditions,
            n_genes=n_genes,
            seed=seed,
        )
        write_manifest(man, out / "dataset_manifest.json")
        conditions = list(arrays["conditions"])
        control = "ctrl"
        n = len(conditions)
        source_delta = deltas_from_control(
            condition_means(arrays["source_X"], arrays["source_cond_ids"], n)
        )
        target_delta = deltas_from_control(
            condition_means(arrays["target_X"], arrays["target_cond_ids"], n)
        )
        save_condition_deltas(
            out / "condition_deltas.npz",
            conditions,
            source_delta,
            target_delta,
            control=control,
            meta={"synthetic": True, "seed": seed},
        )
        src_tag = "synthetic"

    if split_p.exists():
        split = json.loads(split_p.read_text())
    else:
        from .splits import build_split_from_manifest

        man_p = out / "dataset_manifest.json"
        if man_p.exists() and "condition_name_overlap" in json.loads(man_p.read_text()):
            sp = build_split_from_manifest(
                man_p,
                train_frac=train_frac,
                val_frac=val_frac,
                test_frac=test_frac,
                acquisition_frac=acquisition_frac,
                audit_frac=audit_frac,
                seed=seed,
            )
        else:
            sp = make_condition_splits(
                conditions,
                control=control,
                train_frac=train_frac,
                val_frac=val_frac,
                test_frac=test_frac,
                acquisition_frac=acquisition_frac,
                audit_frac=audit_frac,
                seed=seed,
                source_context=source_context,
                target_context=target_context,
            )
        write_split_manifest(sp, split_p)
        split = sp.to_dict()

    cond_to_idx = {c: i for i, c in enumerate(conditions)}
    # drop split conditions missing from arrays (real overlap vs synthetic names)
    def _filter(xs: list[str]) -> list[str]:
        return [c for c in xs if c in cond_to_idx]

    acq = _filter(split.get("acquisition_conditions") or [])
    audit = _filter(split.get("audit_conditions") or [])
    if not acq:
        # synthetic fallback: rebuild split on array conditions
        sp = make_condition_splits(
            conditions,
            control=control,
            train_frac=train_frac,
            val_frac=val_frac,
            test_frac=test_frac,
            acquisition_frac=acquisition_frac,
            audit_frac=audit_frac,
            seed=seed,
        )
        split = sp.to_dict()
        acq = list(sp.acquisition_conditions)
        audit = list(sp.audit_conditions)
        write_split_manifest(sp, split_p)

    audit_ids = [cond_to_idx[c] for c in audit]
    return EpisodeBundle(
        conditions=conditions,
        source_delta=source_delta,
        target_delta=target_delta,
        control=control,
        cond_to_idx=cond_to_idx,
        audit_ids=audit_ids,
        acquisition_conditions=acq,
        split=split,
        source=src_tag,
    )


def bundle_to_state(bundle: EpisodeBundle, seed: int = 0) -> tuple[BaselineState, AcquisitionQueue]:
    state = BaselineState(
        source_delta=bundle.source_delta,
        target_delta=bundle.target_delta,
        revealed_ids=[],
        audit_ids=list(bundle.audit_ids),
        cond_to_idx=dict(bundle.cond_to_idx),
        rng=np.random.default_rng(seed),
    )
    queue = AcquisitionQueue.from_conditions(bundle.acquisition_conditions, seed=seed)
    return state, queue


def assert_no_audit_in_acquisition(bundle: EpisodeBundle) -> None:
    acq = set(bundle.acquisition_conditions)
    audit = set(bundle.split.get("audit_conditions") or [])
    leak = acq & audit
    if leak:
        raise AssertionError(f"audit leaked into acquisition: {sorted(leak)[:5]}")
