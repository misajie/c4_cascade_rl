"""Flexible KG JSON shapes for load_kg_json / parse_kg_payload."""
import json
from pathlib import Path

from c4_cascade_rl.graph_env import load_kg_json, parse_kg_payload


def test_parse_weird_graph_dict_shape():
    """Dict with node_list / edge_list and {source,target,relation} edges."""
    raw = {
        "node_list": [{"id": "A"}, {"id": "B"}, {"id": "C"}],
        "edge_list": [
            {"source": "A", "target": "B", "relation": "activates", "sign": 1},
            {"h": "B", "r": "inhibits", "t": "C", "weight": -1},
            {"from": "A", "to": "C", "type": "assoc"},
        ],
    }
    g = parse_kg_payload(raw)
    assert set(g.nodes) >= {"A", "B", "C"}
    assert len(g.edges) == 3
    assert g.node_degree["A"] >= 1
    assert g.node_degree["B"] >= 1
    assert sum(g.node_degree.values()) > 0


def test_load_kg_json_weird_graph_json(tmp_path):
    kg = tmp_path / "KG"
    kg.mkdir()
    payload = {
        "data": {
            "nodes": ["X", "Y", "Z"],
            "edges": [
                {"source": "X", "target": "Y", "relation": "r1"},
                {"h": "Y", "r": "r2", "t": "Z"},
            ],
        }
    }
    (kg / "graph.json").write_text(json.dumps(payload))
    g = load_kg_json(kg)
    assert g is not None
    assert set(g.nodes) >= {"X", "Y", "Z"}
    assert len(g.edges) == 2
    assert g.node_degree["Y"] >= 1


def test_load_kg_json_nodes_edges_fallback(tmp_path):
    kg = tmp_path / "KG2"
    kg.mkdir()
    # Intentionally unreadable rigid shape in graph.json → fallback
    (kg / "graph.json").write_text(json.dumps({"foo": "bar"}))
    (kg / "nodes.json").write_text(json.dumps([{"id": "n1"}, {"id": "n2"}]))
    (kg / "edges.json").write_text(
        json.dumps([{"from": "n1", "to": "n2", "type": "link", "sign": 1}])
    )
    g = load_kg_json(kg)
    # graph.json fails flexible parse (no edges key) → nodes+edges fallback
    assert g is not None
    assert set(g.nodes) >= {"n1", "n2"}
    assert len(g.edges) == 1
    assert g.node_degree["n1"] >= 1


def test_node_entity_id_keys():
    raw = {
        "nodes": [
            {"entity_id": "DDX1", "ensembl_id": "ENSG1", "node_index": 0},
            {"entity_id": "TP53", "node_index": 1},
        ],
        "edges": [{"source": "DDX1", "target": "TP53", "relation": "regulates", "sign": 1}],
    }
    g = parse_kg_payload(raw)
    assert "DDX1" in g.nodes and "TP53" in g.nodes
    assert g.node_degree["DDX1"] >= 1


def test_edge_src_id_dst_id():
    raw = {
        "nodes": [
            {"entity_id": "DDX1", "node_index": 0},
            {"entity_id": "TP53", "node_index": 1},
        ],
        "edges": [
            {"src_id": "DDX1", "dst_id": "TP53", "rel_type": "regulates", "src_type": "gene", "dst_type": "gene", "sign": 1}
        ],
    }
    g = parse_kg_payload(raw)
    assert len(g.edges) == 1
    assert g.edges[0][0] == "DDX1" and g.edges[0][2] == "TP53"
