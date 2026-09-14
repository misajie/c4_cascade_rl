"""Risk–budget curve helpers."""

from __future__ import annotations

from typing import Iterable, Sequence

import numpy as np


def risk_at_budgets(
    losses_by_step: Sequence[float],
    budgets: Sequence[int],
) -> dict[int, float]:
    """Map budget B -> loss after B acquisitions (1-indexed steps)."""
    out: dict[int, float] = {}
    n = len(losses_by_step)
    for b in budgets:
        if n == 0:
            out[int(b)] = float("nan")
        else:
            idx = min(int(b), n) - 1
            idx = max(0, idx)
            out[int(b)] = float(losses_by_step[idx])
    return out


def cumulative_reward_curve(rewards: Sequence[float]) -> list[float]:
    total = 0.0
    curve = []
    for r in rewards:
        total += float(r)
        curve.append(total)
    return curve


def area_under_budget_curve(budgets: Sequence[int], risks: Sequence[float]) -> float:
    """Trapezoidal AUBC (lower risk better — returns negative area of risk for ranking)."""
    if len(budgets) < 2:
        return float(-risks[0]) if risks else 0.0
    x = np.asarray(budgets, dtype=np.float64)
    y = np.asarray(risks, dtype=np.float64)
    return float(-np.trapezoid(y, x))


def summarize_method_curves(
    method_histories: dict[str, list[dict]],
    budgets: Sequence[int],
) -> dict[str, dict]:
    """Build per-method risk-budget summary from run_episode histories."""
    summary: dict[str, dict] = {}
    for name, hist in method_histories.items():
        losses = [h["loss"] for h in hist]
        rewards = [h["reward"] for h in hist]
        rb = risk_at_budgets(losses, budgets)
        summary[name] = {
            "risk_at_budget": rb,
            "cumulative_reward": cumulative_reward_curve(rewards),
            "aubc": area_under_budget_curve(list(rb.keys()), list(rb.values())),
            "final_loss": float(losses[-1]) if losses else float("nan"),
        }
    return summary


def paired_compare(summary: dict[str, dict], metric: str = "aubc") -> list[tuple[str, float]]:
    ranked = sorted(((m, float(s[metric])) for m, s in summary.items()), key=lambda x: -x[1])
    return ranked
