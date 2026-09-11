"""Splits: official_pert.json, GeneTAK parquet, scaffold Morgan, degree E2."""

from __future__ import annotations

import json
import warnings
from pathlib import Path
from typing import Any, Dict, List, Mapping, Optional, Sequence, Tuple

import numpy as np
import pandas as pd

GENETAK_COLS = ["cell", "drug", "gene", "y_de", "y_dir", "logfc", "fdr"]

# Handbook Gate A1 train pair counts (canonical cell keys)
OFFICIAL_TRAIN_COUNTS = {
    "C32": 128293,
    "HepG2C3A": 102253,
    "HOP62": 154969,
    "Hs766T": 101707,
    "PANC1": 163748,
}

# Disk folder / file aliases → handbook canonical keys
CELL_DIR_ALIASES = {
    "C32": "C32",
    "HepG2_C3A": "HepG2C3A",
    "HepG2C3A": "HepG2C3A",
    "HOP62": "HOP62",
    "Hs_766T": "Hs766T",
    "Hs766T": "Hs766T",
    "PANC-1": "PANC1",
    "PANC1": "PANC1",
}


def canonicalize_cell(name: str) -> str:
    """Map folder/file alias to handbook canonical cell key."""
    key = str(name).strip()
    if key in CELL_DIR_ALIASES:
        return CELL_DIR_ALIASES[key]
    # soft normalize common separators
    soft = key.replace("-", "").replace("_", "")
    for alias, canon in CELL_DIR_ALIASES.items():
        if alias.replace("-", "").replace("_", "") == soft:
            return canon
    return key


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


def discover_genetak_cells(genetak_root: Path | str) -> List[str]:
    """Return disk folder names under GeneTak/ that look like cell dirs."""
    root = Path(genetak_root)
    gt = root / "GeneTak"
    if not gt.is_dir():
        return []
    cells: List[str] = []
    for p in sorted(gt.iterdir()):
        if not p.is_dir():
            continue
        de = list(p.glob("*_DE.csv"))
        di = list(p.glob("*_DIR.csv"))
        if de and di:
            cells.append(p.name)
    return cells


def _read_csv(path: Path) -> pd.DataFrame:
    return pd.read_csv(path)


def load_genetak_cell_csvs(genetak_root: Path | str, cell_folder: str) -> pd.DataFrame:
    """Load `{cell}_DE.csv` + `{cell}_DIR.csv` and merge into GENETAK_COLS (+ split).

    VCWorld DE/DIR schema (prepare.py):
      DE:  pert,gene,label,split  (label 1=DE 0=nonDE)
      DIR: pert,gene,label,split  (label 1=up 0=down → y_dir +1/-1)
    """
    root = Path(genetak_root)
    cell_dir = root / "GeneTak" / cell_folder
    if not cell_dir.is_dir():
        raise FileNotFoundError(f"GeneTak cell dir missing: {cell_dir}")

    de_paths = sorted(cell_dir.glob("*_DE.csv"))
    dir_paths = sorted(cell_dir.glob("*_DIR.csv"))
    if not de_paths or not dir_paths:
        raise FileNotFoundError(f"Need *_DE.csv and *_DIR.csv under {cell_dir}")

    de = _read_csv(de_paths[0])
    di = _read_csv(dir_paths[0])
    for name, df in (("DE", de), ("DIR", di)):
        missing = [c for c in ("pert", "gene", "label", "split") if c not in df.columns]
        if missing:
            raise ValueError(f"{name} CSV missing columns {missing} in {cell_dir}")

    de = de.rename(columns={"pert": "drug", "label": "y_de"})
    di = di.rename(columns={"pert": "drug", "label": "y_dir_raw"})
    di["y_dir"] = di["y_dir_raw"].map(lambda x: 1 if int(x) == 1 else -1)

    merged = de.merge(
        di[["drug", "gene", "split", "y_dir"]],
        on=["drug", "gene", "split"],
        how="inner",
    )
    canon = canonicalize_cell(cell_folder)
    out = pd.DataFrame(
        {
            "cell": canon,
            "drug": merged["drug"].astype(str),
            "gene": merged["gene"].astype(str),
            "y_de": merged["y_de"].astype(int),
            "y_dir": merged["y_dir"].astype(int),
            "logfc": merged["logfc"] if "logfc" in merged.columns else np.nan,
            "fdr": merged["fdr"] if "fdr" in merged.columns else np.nan,
            "split": merged["split"].astype(str),
        }
    )
    return out


def load_all_genetak_csvs(genetak_root: Path | str) -> pd.DataFrame:
    cells = discover_genetak_cells(genetak_root)
    if not cells:
        raise FileNotFoundError(f"No GeneTak cells under {genetak_root}/GeneTak")
    frames = [load_genetak_cell_csvs(genetak_root, c) for c in cells]
    return pd.concat(frames, ignore_index=True)


def build_official_pert_from_genetak(df: pd.DataFrame) -> Dict[str, Dict[str, List[str]]]:
    """Build {cell: {train_drugs, test_drugs}} from merged GeneTAK frame."""
    out: Dict[str, Dict[str, List[str]]] = {}
    for cell, g in df.groupby("cell"):
        train_drugs = sorted(g.loc[g["split"] == "train", "drug"].astype(str).unique().tolist())
        test_drugs = sorted(g.loc[g["split"] == "test", "drug"].astype(str).unique().tolist())
        out[str(cell)] = {"train_drugs": train_drugs, "test_drugs": test_drugs}
    return out


def write_genetak_parquets(df: pd.DataFrame, out_dir: Path | str) -> Dict[str, Path]:
    """Write `{canonical}_{train,test}.parquet` under out_dir."""
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    written: Dict[str, Path] = {}
    cols = GENETAK_COLS + (["split"] if "split" in df.columns else [])
    for cell, g in df.groupby("cell"):
        canon = canonicalize_cell(str(cell))
        for split in ("train", "test"):
            if "split" in g.columns:
                sub = g[g["split"] == split]
            else:
                sub = g if split == "train" else g.iloc[0:0]
            path = out_dir / f"{canon}_{split}.parquet"
            sub2 = sub[[c for c in cols if c in sub.columns]].copy()
            # parquet schema for consumers expects GENETAK_COLS; keep split optional
            keep = [c for c in GENETAK_COLS if c in sub2.columns]
            if "split" in sub2.columns:
                keep = keep + ["split"]
            sub2[keep].to_parquet(path, index=False)
            written[f"{canon}_{split}"] = path
    return written


def gate_a1_counts_from_df(df: pd.DataFrame) -> Dict[str, int]:
    """Count train rows per canonical cell (for Gate A1)."""
    work = df.copy()
    work["cell"] = work["cell"].map(canonicalize_cell)
    if "split" in work.columns:
        work = work[work["split"] == "train"]
    return {str(k): int(v) for k, v in work.groupby("cell").size().items()}


def train_counts_by_cell(df: pd.DataFrame, split_col: str = "split") -> Dict[str, int]:
    if split_col in df.columns:
        sub = df[df[split_col] == "train"]
    else:
        sub = df
    return {canonicalize_cell(str(k)): int(v) for k, v in sub.groupby("cell").size().items()}


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


def random_or_zero_fingerprints(
    n: int,
    n_bits: int = 128,
    rng: Optional[np.random.Generator] = None,
    use_random: bool = True,
) -> np.ndarray:
    """Fallback fps when SMILES/rdkit unavailable (scaffold still runnable)."""
    rng = rng or np.random.default_rng(0)
    if use_random:
        return (rng.random((n, n_bits)) > 0.5).astype(np.uint8)
    warnings.warn("Using zero fingerprints (no SMILES/rdkit); scaffold may be weak", stacklevel=2)
    return np.zeros((n, n_bits), dtype=np.uint8)


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
