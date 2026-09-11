"""Collect rollouts via injectable VCWorld adapter; 4 ctx x 2 temps; Gate B."""

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Callable, Dict, List, Optional, Protocol, Sequence

import numpy as np
import pandas as pd
import yaml

from c4_cascade_rl.buffer import summarize_buffer, write_buffer
from c4_cascade_rl.gates import gate_b
from c4_cascade_rl.reward import (
    RewardConfig,
    compute_reward_with_muted_preds,
    path_order_mute_flips,
)
from c4_cascade_rl.schema import Trajectory, parse_trajectory


class VCWorldAdapter(Protocol):
    def retrieve(self, query: Dict[str, Any]) -> Dict[str, Any]: ...
    def prompt(self, ctx: str, query: Dict[str, Any], retrieved: Dict[str, Any]) -> str: ...
    def infer(self, prompt: str, temperature: float) -> str: ...


@dataclass
class StubAdapter:
    """Synthetic adapter for dry runs / tests (no VCWorld)."""

    flip_rate: float = 0.4
    rng: np.random.Generator = field(default_factory=lambda: np.random.default_rng(0))

    def retrieve(self, query: Dict[str, Any]) -> Dict[str, Any]:
        return {"edges": [], "query": query}

    def prompt(self, ctx: str, query: Dict[str, Any], retrieved: Dict[str, Any]) -> str:
        return f"CTX={ctx}|drug={query.get('drug')}|gene={query.get('gene')}"

    def infer(self, prompt: str, temperature: float) -> str:
        # Deterministic-ish synthetic hop path + stop
        h = abs(hash((prompt, round(temperature, 2)))) % 10000
        src, mid, dst = f"N{h%50}", f"N{(h//50)%50}", f"N{(h//2500)%50}"
        de = h % 2
        direction = 1 if (h // 2) % 2 == 0 else -1
        return (
            f"HOP={src}|regulates|{mid}|+1\n"
            f"HOP={mid}|regulates|{dst}|-1\n"
            f"HOP={dst}|affects|G0|0\n"
            f"STOP|DE={de}|DIR={direction:+d}"
        )


def mute_text_path_order(text: str, k: int) -> str:
    """Delete first k HOP lines in path order; keep STOP."""
    lines = [ln for ln in text.strip().splitlines() if ln.strip()]
    hops = [ln for ln in lines if ln.startswith("HOP=")]
    stops = [ln for ln in lines if ln.startswith("STOP")]
    kept = hops[k:] + stops
    return "\n".join(kept)


def collect_rollouts(
    queries: Sequence[Dict[str, Any]],
    adapter: VCWorldAdapter,
    cfg: Optional[Dict[str, Any]] = None,
    reward_cfg: Optional[RewardConfig] = None,
    cap: int = 20000,
    runs_dir: Path | str = "runs",
    week: str = "W2",
    buffer_path: Optional[Path | str] = None,
    mute_infer: Optional[Callable[[str, float], str]] = None,
) -> pd.DataFrame:
    """4 ctx x 2 temps stratified collect; write buffer; call Gate B.

    For Gate B flips: re-infer STOP after path-order mute of first k hops
    (or use stub mute_infer). Stratified across ctx/temp up to `cap`.
    """
    cfg = cfg or {}
    contexts = [str(x).lower() if x is not False else "false" for x in cfg.get("graph_ctx", ["true", "shuffle", "degree", "empty"])]
    # YAML may parse bare `true` as bool — normalize to string mode names
    contexts = [("true" if x in ("true", "1") else x) for x in contexts]
    temps = list(cfg.get("temps", [0.2, 0.8]))
    k_hops = int(cfg.get("k_hops", 2))
    min_flip = int(cfg.get("gate_B_min_flip", 3000))
    reward_cfg = reward_cfg or RewardConfig(
        alpha=float(cfg.get("alpha", 0.5)),
        k_hops=k_hops,
        lambda_abl=float(cfg.get("lambda_abl", 1.0)),
        lambda_len=float(cfg.get("lambda_len", 0.05)),
        lambda_hall=float(cfg.get("lambda_hall", 1.0)),
    )

    n_slots = max(1, len(contexts) * len(temps))
    per_slot = max(1, cap // n_slots)
    rows: List[Dict[str, Any]] = []
    n_flips = 0
    tid = 0

    for ctx in contexts:
        for temp in temps:
            for q in list(queries)[:per_slot]:
                retrieved = adapter.retrieve(q)
                prompt = adapter.prompt(ctx, q, retrieved)
                text = adapter.infer(prompt, float(temp))
                traj = parse_trajectory(text)
                gold_dir = int(q.get("gold_dir", q.get("y_dir", 1)))
                gold_de = int(q.get("gold_de", q.get("y_de", 1)))

                # Path-order mute then (re)score muted STOP
                muted_text = mute_text_path_order(text, k_hops)
                if mute_infer is not None:
                    muted_text = mute_infer(muted_text, float(temp))
                elif hasattr(adapter, "infer") and traj.valid and traj.length > k_hops:
                    # Re-prompt stub: infer from muted prefix marker
                    muted_text = adapter.infer(prompt + "|MUTED", float(temp))
                    # Prefer path-order deletion of hops but keep regenerated STOP if present
                    mt = parse_trajectory(muted_text)
                    if mt.valid and mt.stop is not None:
                        muted_pred_dir, muted_pred_de = mt.stop.direction, mt.stop.de
                    else:
                        muted_pred_dir, muted_pred_de = gold_dir, gold_de
                else:
                    mt = parse_trajectory(muted_text)
                    if mt.valid and mt.stop is not None:
                        muted_pred_dir, muted_pred_de = mt.stop.direction, mt.stop.de
                    else:
                        muted_pred_dir, muted_pred_de = (
                            (traj.stop.direction if traj.stop else gold_dir),
                            (traj.stop.de if traj.stop else gold_de),
                        )

                # For stub: force some flips based on hash for Gate B testing
                if isinstance(adapter, StubAdapter):
                    if adapter.rng.random() < adapter.flip_rate and traj.valid and traj.stop:
                        muted_pred_dir = -traj.stop.direction
                        muted_pred_de = 1 - traj.stop.de

                rb = compute_reward_with_muted_preds(
                    traj,
                    gold_dir,
                    gold_de,
                    muted_pred_dir,
                    muted_pred_de,
                    cfg=reward_cfg,
                )
                flip = path_order_mute_flips(rb.r_task, rb.r_muted)
                if flip:
                    n_flips += 1
                rows.append(
                    {
                        "traj_id": tid,
                        "cell": q.get("cell", ""),
                        "drug": q.get("drug", ""),
                        "gene": q.get("gene", ""),
                        "text": text,
                        "ctx": ctx,
                        "temp": float(temp),
                        "valid": bool(traj.valid),
                        "gold_dir": gold_dir,
                        "gold_de": gold_de,
                        "r_task": rb.r_task,
                        "r_muted": rb.r_muted,
                        "r_abl": rb.r_abl,
                        "r_total": rb.r_total,
                        "hall": rb.hall,
                        "dir_ok": rb.dir_ok,
                        "length": rb.length,
                        "n_flips": int(flip),
                    }
                )
                tid += 1
                if len(rows) >= cap:
                    break
            if len(rows) >= cap:
                break
        if len(rows) >= cap:
            break

    df = pd.DataFrame(rows)
    if buffer_path is not None:
        write_buffer(df, buffer_path)

    gate_b(
        n_flips=int(df["n_flips"].sum()) if len(df) else n_flips,
        min_flip=min_flip,
        runs_dir=runs_dir,
        week=week,
        k_hops=k_hops,
    )
    return df


def load_config(path: Path | str) -> Dict[str, Any]:
    with open(path) as f:
        return yaml.safe_load(f)


def build_adapter(cfg=None, stub: bool = False):
    """Factory re-export — see c4_cascade_rl.vcworld_adapter.build_adapter."""
    from c4_cascade_rl.vcworld_adapter import build_adapter as _build

    return _build(cfg or {}, stub=stub)
