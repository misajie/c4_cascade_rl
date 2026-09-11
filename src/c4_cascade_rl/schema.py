"""Hop-schema parser: HOP=SRC|REL|DST|SIGN and STOP|DE={0,1}|DIR={+1,-1}.

Invalid line drops the whole trajectory. No LLM repair.
"""

from __future__ import annotations

import re
from dataclasses import dataclass, field
from typing import List, Optional, Sequence, Union

_HOP_RE = re.compile(
    r"^HOP=(?P<src>[^|]+)\|(?P<rel>[^|]+)\|(?P<dst>[^|]+)\|(?P<sign>\+1|-1|0)$"
)
_STOP_RE = re.compile(
    r"^STOP\|DE=(?P<de>0|1)\|DIR=(?P<dir>\+1|-1)$"
)

VALID_SIGNS = {"+1", "-1", "0"}


@dataclass(frozen=True)
class Hop:
    src: str
    rel: str
    dst: str
    sign: int  # +1, -1, or 0

    def to_line(self) -> str:
        sign_s = f"{self.sign:+d}" if self.sign != 0 else "0"
        return f"HOP={self.src}|{self.rel}|{self.dst}|{sign_s}"


@dataclass(frozen=True)
class Stop:
    de: int  # 0 or 1
    direction: int  # +1 or -1

    def to_line(self) -> str:
        return f"STOP|DE={self.de}|DIR={self.direction:+d}"


Line = Union[Hop, Stop]


@dataclass
class Trajectory:
    hops: List[Hop] = field(default_factory=list)
    stop: Optional[Stop] = None
    raw_lines: List[str] = field(default_factory=list)
    valid: bool = True
    drop_reason: Optional[str] = None

    @property
    def length(self) -> int:
        return len(self.hops)

    def path_order_hops(self) -> List[Hop]:
        """Hops in schema path order (as written)."""
        return list(self.hops)

    def muted(self, k: int) -> "Trajectory":
        """Delete first k hops in path order; keep stop."""
        k = max(0, int(k))
        return Trajectory(
            hops=list(self.hops[k:]),
            stop=self.stop,
            raw_lines=[],
            valid=self.valid,
            drop_reason=self.drop_reason,
        )

    def to_text(self) -> str:
        lines = [h.to_line() for h in self.hops]
        if self.stop is not None:
            lines.append(self.stop.to_line())
        return "\n".join(lines)


def parse_line(line: str) -> Optional[Line]:
    """Parse a single schema line. Returns None if invalid."""
    s = line.strip()
    if not s:
        return None
    m = _HOP_RE.match(s)
    if m:
        sign_tok = m.group("sign")
        sign = 0 if sign_tok == "0" else int(sign_tok)
        return Hop(
            src=m.group("src").strip(),
            rel=m.group("rel").strip(),
            dst=m.group("dst").strip(),
            sign=sign,
        )
    m = _STOP_RE.match(s)
    if m:
        return Stop(de=int(m.group("de")), direction=int(m.group("dir")))
    return None


def parse_trajectory(text: str) -> Trajectory:
    """Parse multi-line trajectory. Any invalid line → valid=False, hops cleared."""
    raw = [ln for ln in text.strip().splitlines() if ln.strip()]
    hops: List[Hop] = []
    stop: Optional[Stop] = None
    for ln in raw:
        parsed = parse_line(ln)
        if parsed is None:
            return Trajectory(
                hops=[],
                stop=None,
                raw_lines=raw,
                valid=False,
                drop_reason=f"invalid_line:{ln!r}",
            )
        if isinstance(parsed, Stop):
            if stop is not None:
                return Trajectory(
                    hops=[],
                    stop=None,
                    raw_lines=raw,
                    valid=False,
                    drop_reason="multiple_stop",
                )
            stop = parsed
        else:
            if stop is not None:
                return Trajectory(
                    hops=[],
                    stop=None,
                    raw_lines=raw,
                    valid=False,
                    drop_reason="hop_after_stop",
                )
            hops.append(parsed)
    if stop is None:
        return Trajectory(
            hops=[],
            stop=None,
            raw_lines=raw,
            valid=False,
            drop_reason="missing_stop",
        )
    return Trajectory(hops=hops, stop=stop, raw_lines=raw, valid=True)


def parse_trajectories(texts: Sequence[str]) -> List[Trajectory]:
    return [parse_trajectory(t) for t in texts]


def is_valid_sign(sign: int) -> bool:
    return sign in (+1, -1, 0)
