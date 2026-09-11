"""Gate A1/A2/B/C/D/E decision functions writing JSON under runs/."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Dict, Mapping, Optional, Sequence

# Official GeneTAK train counts (Gate A1 targets, ±1%) — handbook canonical cells
A1_TARGETS = {
    "C32": 128293,
    "HepG2C3A": 102253,
    "HOP62": 154969,
    "Hs766T": 101707,
    "PANC1": 163748,
}
A1_TOLERANCE = 0.01


def _write_json(path: Path, payload: Dict[str, Any]) -> Path:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n")
    return path


def gate_a1(
    train_counts: Mapping[str, int],
    runs_dir: Path | str,
    week: str = "W1",
    tolerance: float = A1_TOLERANCE,
) -> Dict[str, Any]:
    """Gate A1: train counts within 1% of official targets."""
    details = {}
    ok = True
    for cell, target in A1_TARGETS.items():
        got = int(train_counts.get(cell, -1))
        if got < 0:
            details[cell] = {"target": target, "got": None, "pass": False}
            ok = False
            continue
        rel = abs(got - target) / float(target)
        passed = rel <= tolerance
        details[cell] = {"target": target, "got": got, "rel_err": rel, "pass": passed}
        ok = ok and passed
    payload = {"gate": "A1", "pass": ok, "details": details}
    _write_json(Path(runs_dir) / week / "gate_A1.json", payload)
    return payload


def gate_a2(
    n_scaffold_holdout_drugs: int,
    degree_e2_ready: bool,
    runs_dir: Path | str,
    week: str = "W1",
    min_holdout: int = 30,
) -> Dict[str, Any]:
    """Gate A2: scaffold split + degree bottom-quartile E2 ready."""
    scaffold_ok = n_scaffold_holdout_drugs >= min_holdout
    ok = scaffold_ok and bool(degree_e2_ready)
    payload = {
        "gate": "A2",
        "pass": ok,
        "n_scaffold_holdout_drugs": n_scaffold_holdout_drugs,
        "min_holdout": min_holdout,
        "scaffold_ok": scaffold_ok,
        "degree_e2_ready": bool(degree_e2_ready),
    }
    _write_json(Path(runs_dir) / week / "gate_A2.json", payload)
    return payload


# Alias kept for callers / docs
gate_a2_scaffold = gate_a2


def gate_b(
    n_flips: int,
    min_flip: int = 3000,
    runs_dir: Path | str = "runs",
    week: str = "W2",
    k_hops: int = 2,
    recommend_k1: bool = True,
) -> Dict[str, Any]:
    """Gate B: >= min_flip trajectories flip under path-order mute."""
    ok = int(n_flips) >= int(min_flip)
    payload = {
        "gate": "B",
        "pass": ok,
        "n_flips": int(n_flips),
        "min_flip": int(min_flip),
        "k_hops": int(k_hops),
        "action": "proceed" if ok else ("retry_k1" if recommend_k1 and k_hops > 1 else "fail"),
    }
    _write_json(Path(runs_dir) / week / "gate_B.json", payload)
    return payload


def gate_c(
    bc_accuracy: float,
    bc_loss: float,
    runs_dir: Path | str,
    week: str = "W3",
    min_acc: float = 0.5,
    max_loss: float = 2.0,
) -> Dict[str, Any]:
    """Gate C: L0 BC legal-mask accuracy / loss thresholds."""
    ok = (bc_accuracy >= min_acc) and (bc_loss <= max_loss)
    payload = {
        "gate": "C",
        "pass": ok,
        "bc_accuracy": float(bc_accuracy),
        "bc_loss": float(bc_loss),
        "min_acc": min_acc,
        "max_loss": max_loss,
    }
    _write_json(Path(runs_dir) / week / "gate_C.json", payload)
    return payload


def gate_d(
    l0_ablation_slope: float,
    l1_ablation_slope: float,
    runs_dir: Path | str,
    week: str = "W3",
) -> Dict[str, Any]:
    """Gate D: L1 filtered-BC improves ablation slope vs L0 (more positive / steeper drop on mute)."""
    # Higher slope magnitude on retain→0 means stronger causal hops; require L1 > L0.
    improved = float(l1_ablation_slope) > float(l0_ablation_slope)
    payload = {
        "gate": "D",
        "pass": improved,
        "l0_ablation_slope": float(l0_ablation_slope),
        "l1_ablation_slope": float(l1_ablation_slope),
        "delta": float(l1_ablation_slope) - float(l0_ablation_slope),
        "shaping_unlocked": improved,  # handbook: no shaping until Gate B AND L1 moved slope
    }
    _write_json(Path(runs_dir) / week / "gate_D.json", payload)
    return payload


def gate_e(
    policy_entropy: float,
    kl_to_l0: float,
    runs_dir: Path | str,
    week: str = "W4",
    min_entropy: float = 0.05,
    max_kl: float = 10.0,
) -> Dict[str, Any]:
    """Gate E: IQL π entropy / KL-to-L0 sanity."""
    ok = (policy_entropy >= min_entropy) and (kl_to_l0 <= max_kl) and (kl_to_l0 >= 0.0)
    payload = {
        "gate": "E",
        "pass": ok,
        "policy_entropy": float(policy_entropy),
        "kl_to_l0": float(kl_to_l0),
        "min_entropy": min_entropy,
        "max_kl": max_kl,
    }
    _write_json(Path(runs_dir) / week / "gate_E.json", payload)
    return payload


def load_gate(runs_dir: Path | str, week: str, name: str) -> Optional[Dict[str, Any]]:
    p = Path(runs_dir) / week / f"gate_{name}.json"
    if not p.exists():
        return None
    return json.loads(p.read_text())
