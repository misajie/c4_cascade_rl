"""Stub still works; CLI adapter constructs without calling network."""
from pathlib import Path

from c4_cascade_rl.collect import StubAdapter, build_adapter
from c4_cascade_rl.vcworld_adapter import CLIVCWorldAdapter, build_adapter as build_adapter_vc


def test_stub_adapter_infer():
    a = StubAdapter(flip_rate=0.0)
    text = a.infer(a.prompt("true", {"drug": "D0", "gene": "G0"}, a.retrieve({"drug": "D0"})), 0.2)
    assert "STOP" in text
    assert "HOP=" in text


def test_build_adapter_stub():
    a = build_adapter({}, stub=True)
    assert isinstance(a, StubAdapter)


def test_cli_adapter_constructs(tmp_path):
    vc = tmp_path / "VCWorld" / "src" / "cli_pipeline"
    vc.mkdir(parents=True)
    (vc / "cli.py").write_text("# stub cli\n")
    graph = tmp_path / "graph" / "VCWorld"
    graph.mkdir(parents=True)
    cfg = {
        "paths": {
            "vcworld_root": str(tmp_path / "VCWorld"),
            "zenodo_graph": str(tmp_path / "graph"),
            "beta_model": "dummy-model",
            "runs_dir": str(tmp_path / "runs"),
        }
    }
    a = build_adapter_vc(cfg, stub=False)
    assert isinstance(a, CLIVCWorldAdapter)
    assert a.cli_py == vc / "cli.py"
    # retrieve/prompt should not raise just from construction; retrieve may fail CLI but returns payload
    retrieved = a.retrieve({"drug": "d", "gene": "g", "cell": "C32"})
    assert "query" in retrieved
    prompt = a.prompt("true", {"drug": "d", "gene": "g", "cell": "C32"}, retrieved)
    assert isinstance(prompt, str) and len(prompt) > 0
