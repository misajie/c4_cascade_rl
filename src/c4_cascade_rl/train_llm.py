"""LoRA SFT / DPO stubs: L5a verbalization+tail, L5b DPO R_abl, L5c task-only.

No PPO/GRPO. Dry-run without GPU/PEFT.
"""

from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, List, Optional, Sequence

import numpy as np


@dataclass
class LoRAConfig:
    rank: int = 16
    alpha: int = 32
    dropout: float = 0.05
    lr: float = 1e-4
    epochs: int = 1
    max_steps: int = 10


def _loss_on_tail_only(full_ids: Sequence[int], prompt_len: int) -> Dict[str, Any]:
    """Mask: loss only on verbalization + tail tokens after prompt_len."""
    n = len(full_ids)
    mask = [0] * min(prompt_len, n) + [1] * max(0, n - prompt_len)
    return {"mask": mask, "n_supervised": sum(mask)}


def sft_l5a_dry_run(
    examples: Sequence[Dict[str, Any]],
    cfg: Optional[LoRAConfig] = None,
    out_dir: Path | str = "runs/W5",
) -> Dict[str, Any]:
    """L5a SFT: loss on verbalization + tail only. Dry loop."""
    cfg = cfg or LoRAConfig()
    losses = []
    for i, ex in enumerate(examples[: cfg.max_steps]):
        text = ex.get("text", "")
        verbal = ex.get("verbalization", text)
        prompt = ex.get("prompt", "")
        # Fake token ids
        prompt_ids = list(range(len(prompt.split()) or 1))
        full = prompt_ids + list(range(100, 100 + max(1, len(verbal.split()))))
        info = _loss_on_tail_only(full, len(prompt_ids))
        # Synthetic decreasing loss
        losses.append(1.0 / (1 + i) + 0.01 * (1 - info["n_supervised"] / max(len(full), 1)))
    payload = {
        "stage": "L5a",
        "lora_rank": cfg.rank,
        "dry_run": True,
        "steps": len(losses),
        "mean_loss": float(np.mean(losses)) if losses else 0.0,
        "note": "loss on verbalization+tail only; PEFT not loaded",
    }
    out = Path(out_dir)
    out.mkdir(parents=True, exist_ok=True)
    (out / "l5a_sft.json").write_text(json.dumps(payload, indent=2) + "\n")
    return payload


def dpo_pair_loss(pref_logp: float, dispref_logp: float, beta: float = 0.1) -> float:
    """Simple scalar DPO surrogate for dry-run."""
    return float(-np.log(1.0 / (1.0 + np.exp(-beta * (pref_logp - dispref_logp)))))


def dpo_l5b_dry_run(
    pairs: Sequence[Dict[str, Any]],
    cfg: Optional[LoRAConfig] = None,
    out_dir: Path | str = "runs/W5",
    beta: float = 0.1,
) -> Dict[str, Any]:
    """L5b DPO on R_abl preference pairs."""
    cfg = cfg or LoRAConfig()
    losses = []
    for i, p in enumerate(pairs[: cfg.max_steps]):
        # Prefer higher r_abl
        pref = float(p.get("chosen_logp", 0.0))
        disp = float(p.get("rejected_logp", -1.0))
        losses.append(dpo_pair_loss(pref, disp, beta=beta))
    payload = {
        "stage": "L5b",
        "objective": "DPO_R_abl",
        "lora_rank": cfg.rank,
        "dry_run": True,
        "steps": len(losses),
        "mean_loss": float(np.mean(losses)) if losses else 0.0,
    }
    out = Path(out_dir)
    out.mkdir(parents=True, exist_ok=True)
    (out / "l5b_dpo.json").write_text(json.dumps(payload, indent=2) + "\n")
    return payload


def dpo_l5c_dry_run(
    pairs: Sequence[Dict[str, Any]],
    cfg: Optional[LoRAConfig] = None,
    out_dir: Path | str = "runs/W5",
    beta: float = 0.1,
) -> Dict[str, Any]:
    """L5c task-only DPO (no R_abl)."""
    cfg = cfg or LoRAConfig()
    losses = []
    for i, p in enumerate(pairs[: cfg.max_steps]):
        pref = float(p.get("chosen_logp", 0.0))
        disp = float(p.get("rejected_logp", -1.0))
        losses.append(dpo_pair_loss(pref, disp, beta=beta))
    payload = {
        "stage": "L5c",
        "objective": "DPO_task_only",
        "lora_rank": cfg.rank,
        "dry_run": True,
        "steps": len(losses),
        "mean_loss": float(np.mean(losses)) if losses else 0.0,
    }
    out = Path(out_dir)
    out.mkdir(parents=True, exist_ok=True)
    (out / "l5c_dpo.json").write_text(json.dumps(payload, indent=2) + "\n")
    return payload


def train_llm(
    stage: str,
    examples: Sequence[Dict[str, Any]],
    dry_run: bool = True,
    lora_rank: int = 16,
    out_dir: Path | str = "runs/W5",
) -> Dict[str, Any]:
    cfg = LoRAConfig(rank=lora_rank)
    if not dry_run:
        # Real PEFT path — optional; fall back to dry if missing
        try:
            import peft  # noqa: F401
            import transformers  # noqa: F401
        except ImportError:
            dry_run = True
    if stage == "L5a":
        return sft_l5a_dry_run(examples, cfg=cfg, out_dir=out_dir)
    if stage == "L5b":
        return dpo_l5b_dry_run(examples, cfg=cfg, out_dir=out_dir)
    if stage == "L5c":
        return dpo_l5c_dry_run(examples, cfg=cfg, out_dir=out_dir)
    raise ValueError(f"unknown stage {stage}; expected L5a/L5b/L5c (no PPO/GRPO)")
