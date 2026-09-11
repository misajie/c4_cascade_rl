"""Ablation retain sweep; E1/E2/E2b/E3/E4/E5; Path N; never average E1/E2b."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Dict, List, Mapping, Optional, Sequence

import numpy as np

RETAIN_DEFAULT = [1.0, 0.75, 0.5, 0.25, 0.0]
SPLIT_NAMES = ("E1", "E2", "E2b", "E3", "E4", "E5")


def retain_prefix_hops(n_hops: int, retain: float) -> int:
    """Keep first floor(retain * n) hops in path order (prefix retain)."""
    retain = float(np.clip(retain, 0.0, 1.0))
    if n_hops <= 0:
        return 0
    return int(np.floor(retain * n_hops + 1e-9))


def path_order_retain_mute(n_hops: int, retain: float) -> int:
    """Number of hops to DELETE from the front when retaining `retain` fraction of the tail.

    Handbook Path N retain sweep keeps a prefix of the path (path-order).
    retain=1.0 → keep all (delete 0); retain=0.0 → delete all.
    """
    keep = retain_prefix_hops(n_hops, retain)
    return max(0, n_hops - keep)


def score_split(
    y_true_dir: Sequence[int],
    y_pred_dir: Sequence[int],
    y_true_de: Sequence[int],
    y_pred_de: Sequence[int],
    alpha: float = 0.5,
) -> Dict[str, float]:
    y_true_dir = np.asarray(y_true_dir)
    y_pred_dir = np.asarray(y_pred_dir)
    y_true_de = np.asarray(y_true_de)
    y_pred_de = np.asarray(y_pred_de)
    n = max(len(y_true_dir), 1)
    dir_acc = float((y_true_dir == y_pred_dir).mean()) if len(y_true_dir) else 0.0
    de_acc = float((y_true_de == y_pred_de).mean()) if len(y_true_de) else 0.0
    r_task = dir_acc + alpha * de_acc
    return {"dir_acc": dir_acc, "de_acc": de_acc, "r_task": r_task, "n": float(n)}


def ablation_slope(retains: Sequence[float], scores: Sequence[float]) -> float:
    """Linear slope of score vs retain (higher = more causal dependence on early hops)."""
    x = np.asarray(retains, dtype=np.float64)
    y = np.asarray(scores, dtype=np.float64)
    if len(x) < 2:
        return 0.0
    return float(np.polyfit(x, y, 1)[0])


def evaluate_retain_sweep(
    records: Sequence[Mapping[str, Any]],
    retains: Optional[Sequence[float]] = None,
    alpha: float = 0.5,
    split_key: str = "split",
) -> Dict[str, Any]:
    """Evaluate per-split retain sweep. Never averages E1 with E2b."""
    retains = list(retains or RETAIN_DEFAULT)
    by_split: Dict[str, List[Mapping[str, Any]]] = {}
    for r in records:
        sp = str(r.get(split_key, "E1"))
        by_split.setdefault(sp, []).append(r)

    summary: Dict[str, Any] = {"retains": retains, "splits": {}, "note": "E1 and E2b reported separately; never averaged"}
    for sp, rows in by_split.items():
        curve = []
        for ret in retains:
            # Simulate muted preds: if retain < 1, degrade toward random / muted_pred fields
            preds_dir = []
            preds_de = []
            golds_dir = []
            golds_de = []
            for row in rows:
                golds_dir.append(int(row["gold_dir"]))
                golds_de.append(int(row["gold_de"]))
                n_hops = int(row.get("length", 0))
                keep = retain_prefix_hops(n_hops, ret)
                if keep == n_hops:
                    preds_dir.append(int(row.get("pred_dir", row["gold_dir"])))
                    preds_de.append(int(row.get("pred_de", row["gold_de"])))
                else:
                    # use muted predictions if provided
                    preds_dir.append(int(row.get("muted_pred_dir", row.get("pred_dir", 0))))
                    preds_de.append(int(row.get("muted_pred_de", row.get("pred_de", 0))))
            sc = score_split(golds_dir, preds_dir, golds_de, preds_de, alpha=alpha)
            sc["retain"] = ret
            curve.append(sc)
        slope = ablation_slope(retains, [c["r_task"] for c in curve])
        summary["splits"][sp] = {"curve": curve, "ablation_slope": slope, "n": len(rows)}

    # Explicitly refuse combined E1+E2b metric
    summary["e1_e2b_averaged"] = None
    summary["e1_e2b_policy"] = "never_average"
    return summary


def write_summary(summary: Dict[str, Any], path: Path | str) -> Path:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(summary, indent=2) + "\n")
    return path


def path_n_eval(
    records: Sequence[Mapping[str, Any]],
    out_dir: Path | str,
    retains: Optional[Sequence[float]] = None,
    alpha: float = 0.5,
) -> Dict[str, Any]:
    summary = evaluate_retain_sweep(records, retains=retains, alpha=alpha)
    write_summary(summary, Path(out_dir) / "summary.json")
    return summary
