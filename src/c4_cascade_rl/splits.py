"""Splits: official_pert.json, GeneTAK parquet, scaffold Morgan, degree E2."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Dict, List, Mapping, Optional, Sequence

import numpy as np
import pandas as pd

GENETAK_COLS = ["cell", "drug", "gene", "y_de", "y_dir", "logfc", "fdr"]

OFFICIAL_TRAIN_COUNTS = {
    "A549": 128293,
    "K562": 102253,
    "MCF7": 154969,
    "PC3": 101707,
    "VCAP": 163748,
}


def load_official_pert(path: Path | str) -> Dict[str, Any]:
    path = Path(path)
    with path.open() as f:
        return json.load(f)


def save_official_pert(data: Mapping[str, Any], path: Path | str) -> Path:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(dict(data), indent=2) + "\n")
    return path


def load_genetak_parquet(path: Path | str) -> pd.DataFrame:
    df = pd.read_parquet(path)
    missing = [c for c in GENETAK_COLS if c not in df.columns]
    if missing:
        raise ValueError(f"GeneTAK missing columns: {missing}")
    return df[GENETAK_COLS].copy()


def train_counts_by_cell(df: pd.DataFrame, split_col: str = "split") -> Dict[str, int]:
    if split_col in df.columns:
        sub = df[df[split_col] == "train"]
    else:
        sub = df
    return {str(k): int(v) for k, v in sub.groupby("cell").size().items()}


def check_a1_counts(counts: Mapping[str, int], tol: float = 0.01) -> bool:
    for cell, target in OFFICIAL_TRAIN_COUNTS.items():
        got = counts.get(cell)
        if got is None:
            return False
        if abs(got - target) / target > tol:
            return False
    return True


def tanimoto_similarity(a: np.ndarray, b: np.ndarray) -> float:
    a = a.astype(np.float64).ravel()
    b = b.astype(np.float64).ravel()
    inter = float(np.minimum(a, b).sum())
    union = float(np.maximum(a, b).sum())
    if union <= 0:
        return 0.0
    return inter / union


def tanimoto_distance(a: np.ndarray, b: np.ndarray) -> float:
    return 1.0 - tanimoto_similarity(a, b)


def pairwise_tanimoto_distance(fps: np.ndarray) -> np.ndarray:
    n = fps.shape[0]
    D = np.zeros((n, n), dtype=np.float64)
    for i in range(n):
        for j in range(i + 1, n):
            d = tanimoto_distance(fps[i], fps[j])
            D[i, j] = D[j, i] = d
    return D


def scaffold_cluster_holdout(
    drug_ids: Sequence[str],
    fingerprints: np.ndarray,
    distance_threshold: float = 0.6,
    target_frac: float = 0.20,
    min_holdout: int = 30,
    recut_threshold: float = 0.5,
    rng: Optional[np.random.Generator] = None,
) -> Dict[str, Any]:
    from scipy.cluster.hierarchy import fcluster, linkage
    from scipy.spatial.distance import squareform

    rng = rng or np.random.default_rng(0)
    drug_ids = list(drug_ids)
    fps = np.asarray(fingerprints)
    assert fps.ndim == 2 and fps.shape[0] == len(drug_ids)

    def _cluster(thresh: float) -> np.ndarray:
        D = pairwise_tanimoto_distance(fps)
        D = np.clip(D, 0.0, 1.0)
        np.fill_diagonal(D, 0.0)
        condensed = squareform(D, checks=False)
        Z = linkage(condensed, method="average")
        return fcluster(Z, t=thresh, criterion="distance")

    labels = _cluster(distance_threshold)
    used_thresh = distance_threshold

    def _holdout_from_labels(labels: np.ndarray) -> List[str]:
        clusters: Dict[int, List[int]] = {}
        for i, lab in enumerate(labels):
            clusters.setdefault(int(lab), []).append(i)
        order = sorted(clusters.keys(), key=lambda c: len(clusters[c]))
        target_n = max(1, int(round(target_frac * len(drug_ids))))
        hold_idx: List[int] = []
        for c in order:
            if len(hold_idx) >= target_n:
                break
            hold_idx.extend(clusters[c])
        return [drug_ids[i] for i in hold_idx]

    holdout = _holdout_from_labels(labels)
    if len(holdout) < min_holdout and distance_threshold != recut_threshold:
        labels = _cluster(recut_threshold)
        used_thresh = recut_threshold
        holdout = _holdout_from_labels(labels)

    hold_set = set(holdout)
    train_drugs = [d for d in drug_ids if d not in hold_set]
    return {
        "holdout_drugs": holdout,
        "train_drugs": train_drugs,
        "n_holdout": len(holdout),
        "n_train": len(train_drugs),
        "distance_threshold": used_thresh,
        "labels": labels.tolist(),
        "target_frac": target_frac,
    }


def morgan_fingerprints_from_smiles(
    smiles_list: Sequence[str],
    radius: int = 2,
    n_bits: int = 2048,
) -> np.ndarray:
    from rdkit import Chem
    from rdkit.Chem import AllChem

    fps = []
    for smi in smiles_list:
        mol = Chem.MolFromSmiles(smi)
        if mol is None:
            fps.append(np.zeros(n_bits, dtype=np.uint8))
            continue
        bv = AllChem.GetMorganFingerprintAsBitVect(mol, radius, nBits=n_bits)
        arr = np.zeros(n_bits, dtype=np.uint8)
        for i in range(n_bits):
            if bv.GetBit(i):
                arr[i] = 1
        fps.append(arr)
    return np.stack(fps, axis=0)


def build_degree_split(
    node_degrees: Mapping[str, float],
    out_path: Path | str,
    quartile: float = 0.25,
) -> Dict[str, Any]:
    items = sorted(node_degrees.items(), key=lambda kv: kv[1])
    out_path = Path(out_path)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    if not items:
        payload = {"e2_nodes": [], "threshold": None, "quartile": quartile}
        out_path.write_text(json.dumps(payload, indent=2) + "\n")
        return payload
    vals = np.array([v for _, v in items], dtype=np.float64)
    thr = float(np.quantile(vals, quartile))
    e2_nodes = [n for n, v in items if v <= thr]
    payload = {
        "e2_nodes": e2_nodes,
        "threshold": thr,
        "quartile": quartile,
        "n_e2": len(e2_nodes),
        "n_total": len(items),
    }
    out_path.write_text(json.dumps(payload, indent=2) + "\n")
    return payload


def e2b_tanimoto_holdout(
    drug_ids: Sequence[str],
    fingerprints: np.ndarray,
    threshold: float = 0.4,
    seed: int = 0,
) -> Dict[str, Any]:
    dist_thresh = 1.0 - float(threshold)
    result = scaffold_cluster_holdout(
        drug_ids,
        fingerprints,
        distance_threshold=dist_thresh,
        target_frac=0.20,
        min_holdout=1,
        recut_threshold=dist_thresh,
        rng=np.random.default_rng(seed),
    )
    result["tanimoto_threshold"] = threshold
    result["split"] = "E2b"
    return result


def write_synthetic_official_pert(path: Path | str) -> Path:
    data = {
        "cells": list(OFFICIAL_TRAIN_COUNTS.keys()),
        "train_counts": dict(OFFICIAL_TRAIN_COUNTS),
        "note": "synthetic placeholder — replace with real official_pert.json",
    }
    return save_official_pert(data, path)
