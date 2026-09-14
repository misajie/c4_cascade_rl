"""Fixed acquisition queue: reveal without replacement (paired across methods)."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Iterable, Sequence

import numpy as np


@dataclass
class AcquisitionQueue:
    """Paired queue: same order for every method; reveal without replacement."""

    items: list[str]
    revealed: list[str] = field(default_factory=list)
    _remaining: list[str] = field(default_factory=list)

    def __post_init__(self) -> None:
        if not self._remaining and self.items:
            self._remaining = list(self.items)

    @classmethod
    def from_conditions(cls, conditions: Sequence[str], seed: int = 0) -> "AcquisitionQueue":
        rng = np.random.default_rng(seed)
        order = list(conditions)
        rng.shuffle(order)
        return cls(items=order, _remaining=list(order))

    @property
    def remaining(self) -> list[str]:
        return list(self._remaining)

    @property
    def n_remaining(self) -> int:
        return len(self._remaining)

    @property
    def legal_mask(self) -> np.ndarray:
        """Boolean mask over original `items` order: True iff not yet revealed."""
        revealed_set = set(self.revealed)
        return np.array([c not in revealed_set for c in self.items], dtype=bool)

    def reveal(self, condition: str) -> str:
        if condition not in self._remaining:
            raise ValueError(f"Cannot reveal {condition!r}; not remaining")
        self._remaining.remove(condition)
        self.revealed.append(condition)
        return condition

    def reveal_index(self, idx: int) -> str:
        """Reveal by index into original `items` (must be legal)."""
        cond = self.items[idx]
        return self.reveal(cond)

    def reveal_next_random(self, rng: np.random.Generator | None = None) -> str:
        rng = rng or np.random.default_rng()
        if not self._remaining:
            raise RuntimeError("Queue exhausted")
        pick = str(rng.choice(self._remaining))
        return self.reveal(pick)

    def clone(self) -> "AcquisitionQueue":
        return AcquisitionQueue(
            items=list(self.items),
            revealed=list(self.revealed),
            _remaining=list(self._remaining),
        )


def paired_queues(conditions: Sequence[str], seed: int, n_methods: int) -> list[AcquisitionQueue]:
    """Same shuffled order for every method (paired acquisition)."""
    base = AcquisitionQueue.from_conditions(conditions, seed=seed)
    return [base.clone() for _ in range(n_methods)]
