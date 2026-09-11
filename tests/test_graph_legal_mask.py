"""Legal mask + context modes."""
import numpy as np
from c4_cascade_rl.graph_env import GraphData, GraphEnv, render_context


def _tiny_graph():
    nodes = ["A", "B", "C", "D"]
    edges = [
        ("A", "r", "B", 1),
        ("B", "r", "C", -1),
        ("B", "r", "D", 0),
        ("C", "r", "D", 1),
    ]
    return GraphData(nodes=nodes, edges=edges)


def test_legal_actions_and_mask():
    g = _tiny_graph()
    env = GraphEnv(graph=g, horizon=6, ctx_mode="true")
    env.reset(["A"])
    legal = env.legal_actions()
    assert ("A", "r", "B", 1) in legal
    mask = env.legal_mask(g.edges)
    assert mask.shape == (len(g.edges),)
    assert mask[0] == 1.0
    env.apply_hop(("A", "r", "B", 1))
    legal2 = env.legal_actions()
    assert set(legal2) == {("B", "r", "C", -1), ("B", "r", "D", 0)}


def test_illegal_hop_raises():
    env = GraphEnv(graph=_tiny_graph())
    env.reset(["A"])
    import pytest
    with pytest.raises(ValueError):
        env.apply_hop(("C", "r", "D", 1))


def test_ctx_modes():
    g = _tiny_graph()
    assert render_context(g, "empty") == []
    assert len(render_context(g, "true")) > 0
    sh = render_context(g, "shuffle")
    assert len(sh) > 0
    deg = render_context(g, "degree")
    assert isinstance(deg, list)
