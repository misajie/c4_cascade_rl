"""Graph environment: load graph, legal actions, apply hop, context modes."""

from __future__ import annotations

import json
import random
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Dict, List, Optional, Sequence, Set, Tuple

import numpy as np

from c4_cascade_rl.schema import Hop, Stop, Trajectory

Edge = Tuple[str, str, str, int]  # src, rel, dst, sign


@dataclass
class GraphData:
    nodes: List[str]
    edges: List[Edge]
    node_degree: Dict[str, float] = field(default_factory=dict)
    edge_set: Set[Edge] = field(default_factory=set)
    adj: Dict[str, List[Edge]] = field(default_factory=dict)

    def __post_init__(self):
        if not self.edge_set:
            self.edge_set = set(self.edges)
        if not self.adj:
            adj: Dict[str, List[Edge]] = {}
            for e in self.edges:
                adj.setdefault(e[0], []).append(e)
            self.adj = adj
        if not self.node_degree:
            deg: Dict[str, float] = {n: 0.0 for n in self.nodes}
            for s, _, d, _ in self.edges:
                deg[s] = deg.get(s, 0.0) + 1.0
                deg[d] = deg.get(d, 0.0) + 1.0
            self.node_degree = deg


def load_graph(path: Path | str) -> GraphData:
    """Load graph from JSON: {nodes: [...], edges: [[src,rel,dst,sign], ...]}."""
    path = Path(path)
    raw = json.loads(path.read_text())
    nodes = list(raw["nodes"])
    edges: List[Edge] = []
    for e in raw["edges"]:
        if len(e) == 3:
            src, rel, dst = e
            sign = 0
        else:
            src, rel, dst, sign = e[0], e[1], e[2], int(e[3])
        edges.append((str(src), str(rel), str(dst), int(sign)))
    return GraphData(nodes=nodes, edges=edges)


def save_graph(graph: GraphData, path: Path | str) -> Path:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    payload = {
        "nodes": graph.nodes,
        "edges": [list(e) for e in graph.edges],
    }
    path.write_text(json.dumps(payload) + "\n")
    return path


CTX_MODES = ("true", "shuffle", "degree", "empty")


def render_context(
    graph: GraphData,
    mode: str,
    focus_nodes: Optional[Sequence[str]] = None,
    rng: Optional[random.Random] = None,
    max_edges: int = 64,
) -> List[Edge]:
    """Return edge list under context mode: true/shuffle/degree/empty."""
    if mode not in CTX_MODES:
        raise ValueError(f"unknown graph_ctx mode: {mode}")
    rng = rng or random.Random(0)
    if mode == "empty":
        return []
    edges = list(graph.edges)
    if focus_nodes:
        focus = set(focus_nodes)
        focused = [e for e in edges if e[0] in focus or e[2] in focus]
        edges = focused or edges
    if mode == "true":
        return edges[:max_edges]
    if mode == "shuffle":
        edges = edges[:]
        rng.shuffle(edges)
        return edges[:max_edges]
    if mode == "degree":
        degs = graph.node_degree
        if not degs:
            return edges[:max_edges]
        vals = np.array(list(degs.values()), dtype=np.float64)
        thr = float(np.quantile(vals, 0.25))
        low = {n for n, v in degs.items() if v <= thr}
        deg_edges = [e for e in edges if e[0] in low or e[2] in low]
        if not deg_edges:
            deg_edges = sorted(
                edges,
                key=lambda e: -(degs.get(e[0], 0) + degs.get(e[2], 0)),
            )
        return deg_edges[:max_edges]
    return edges[:max_edges]


@dataclass
class GraphEnv:
    graph: GraphData
    horizon: int = 6
    ctx_mode: str = "true"
    rng: random.Random = field(default_factory=lambda: random.Random(0))

    current: List[Hop] = field(default_factory=list)
    done: bool = False
    focus: List[str] = field(default_factory=list)

    def reset(self, start_nodes: Optional[Sequence[str]] = None) -> Dict[str, Any]:
        self.current = []
        self.done = False
        self.focus = list(start_nodes) if start_nodes else (self.graph.nodes[:1] if self.graph.nodes else [])
        return self.observe()

    def observe(self) -> Dict[str, Any]:
        ctx_edges = render_context(self.graph, self.ctx_mode, self.focus, self.rng)
        return {
            "hops": list(self.current),
            "legal": self.legal_actions(),
            "ctx_edges": ctx_edges,
            "t": len(self.current),
            "done": self.done,
        }

    def legal_actions(self) -> List[Edge]:
        if self.done or len(self.current) >= self.horizon:
            return []
        if not self.current:
            seeds = self.focus or self.graph.nodes
            acts: List[Edge] = []
            for s in seeds:
                acts.extend(self.graph.adj.get(s, []))
            return acts
        last = self.current[-1].dst
        return list(self.graph.adj.get(last, []))

    def legal_mask(self, all_edges: Sequence[Edge]) -> np.ndarray:
        legal = set(self.legal_actions())
        return np.array([1.0 if e in legal else 0.0 for e in all_edges], dtype=np.float32)

    def apply_hop(self, edge: Edge) -> Dict[str, Any]:
        if self.done:
            raise RuntimeError("episode done")
        if edge not in set(self.legal_actions()):
            raise ValueError(f"illegal hop: {edge}")
        hop = Hop(src=edge[0], rel=edge[1], dst=edge[2], sign=int(edge[3]))
        self.current.append(hop)
        self.focus = [hop.dst]
        if len(self.current) >= self.horizon:
            self.done = True
        return self.observe()

    def stop(self, de: int, direction: int) -> Trajectory:
        self.done = True
        return Trajectory(hops=list(self.current), stop=Stop(de=de, direction=direction), valid=True)

    def edge_set_for_hall(self) -> Set:
        return set(self.graph.edges) | {(s, r, d) for s, r, d, _ in self.graph.edges}


def load_kg_json(kg_dir: Path | str) -> Optional[GraphData]:
    """Best-effort load of VCWorld KG from nodes/edges/graph JSON under kg_dir.

    Looks for nodes.json + edges.json, or a combined graph.json.
    Returns None if files are missing or unreadable.
    """
    kg_dir = Path(kg_dir)
    graph_path = kg_dir / "graph.json"
    nodes_path = kg_dir / "nodes.json"
    edges_path = kg_dir / "edges.json"
    try:
        if graph_path.is_file():
            return load_graph(graph_path)
        if nodes_path.is_file() and edges_path.is_file():
            nodes_raw = json.loads(nodes_path.read_text())
            edges_raw = json.loads(edges_path.read_text())
            if isinstance(nodes_raw, dict) and "nodes" in nodes_raw:
                nodes = list(nodes_raw["nodes"])
            elif isinstance(nodes_raw, list):
                nodes = [str(n.get("id", n) if isinstance(n, dict) else n) for n in nodes_raw]
            else:
                nodes = [str(k) for k in nodes_raw]
            edges: List[Edge] = []
            edge_list = edges_raw["edges"] if isinstance(edges_raw, dict) and "edges" in edges_raw else edges_raw
            for e in edge_list:
                if isinstance(e, dict):
                    src = str(e.get("src", e.get("source", e.get("from", ""))))
                    rel = str(e.get("rel", e.get("relation", e.get("type", "rel"))))
                    dst = str(e.get("dst", e.get("target", e.get("to", ""))))
                    sign = int(e.get("sign", e.get("weight", 0)) or 0)
                    edges.append((src, rel, dst, sign))
                elif isinstance(e, (list, tuple)):
                    if len(e) == 3:
                        edges.append((str(e[0]), str(e[1]), str(e[2]), 0))
                    else:
                        edges.append((str(e[0]), str(e[1]), str(e[2]), int(e[3])))
            if not nodes:
                nodes = sorted({s for s, _, d, _ in edges} | {d for s, _, d, _ in edges})
            return GraphData(nodes=nodes, edges=edges)
    except Exception:
        return None
    return None
