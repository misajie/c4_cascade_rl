"""Parquet buffer IO; drug-wise 9:1 split."""

from __future__ import annotations

from pathlib import Path
from typing import Any, Dict, Optional, Sequence, Tuple

import numpy as np
import pandas as pd

BUFFER_COLS = [
    "traj_id",
    "cell",
    "drug",
    "gene",
    "text",
    "ctx",
    "temp",
    "valid",
    "gold_dir",
    "gold_de",
    "r_task",
    "r_muted",
    "r_abl",
    "r_total",
    "hall",
    "dir_ok",
    "length",
    "n_flips",
]


def write_buffer(df: pd.DataFrame, path: Path | str) -> Path:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    out = df.copy()
    for c in BUFFER_COLS:
        if c not in out.columns:
            out[c] = np.nan
    out.to_parquet(path, index=False)
    return path


def read_buffer(path: Path | str) -> pd.DataFrame:
    return pd.read_parquet(path)


def drug_wise_split(
    df: pd.DataFrame,
    train_frac: float = 0.9,
    seed: int = 0,
    drug_col: str = "drug",
) -> Tuple[pd.DataFrame, pd.DataFrame]:
    """Drug-wise 9:1 split — all rows for a drug stay together."""
    drugs = sorted(df[drug_col].astype(str).unique())
    rng = np.random.default_rng(seed)
    rng.shuffle(drugs)
    n_train = max(1, int(round(train_frac * len(drugs))))
    if len(drugs) >= 2:
        n_train = min(n_train, len(drugs) - 1)
    train_drugs = set(drugs[:n_train])
    val_drugs = set(drugs[n_train:])
    train = df[df[drug_col].astype(str).isin(train_drugs)].copy()
    val = df[df[drug_col].astype(str).isin(val_drugs)].copy()
    return train, val


def filter_l1(df: pd.DataFrame) -> pd.DataFrame:
    """L1 filtered BC: DIR ok & R_abl > 0."""
    m = (df["dir_ok"].astype(bool)) & (df["r_abl"].astype(float) > 0)
    return df.loc[m].copy()


def append_buffer(df: pd.DataFrame, path: Path | str) -> Path:
    path = Path(path)
    if path.exists():
        old = read_buffer(path)
        df = pd.concat([old, df], ignore_index=True)
    return write_buffer(df, path)


def summarize_buffer(df: pd.DataFrame) -> Dict[str, Any]:
    return {
        "n": int(len(df)),
        "n_valid": int(df["valid"].sum()) if "valid" in df.columns else len(df),
        "n_drugs": int(df["drug"].nunique()) if "drug" in df.columns else 0,
        "mean_r_abl": float(df["r_abl"].mean()) if "r_abl" in df.columns else 0.0,
        "n_flips": int((df["r_task"] != df["r_muted"]).sum()) if "r_task" in df.columns else 0,
    }
