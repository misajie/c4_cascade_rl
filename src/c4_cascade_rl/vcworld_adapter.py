"""VCWorld CLI adapter — wraps GENTEL-lab/VCWorld without forking it.

Single-query path (Week2 collect):
  Prefer `python cli.py single prompt ...` when available, then `infer`.
Batch path (full Week2):
  `python cli.py de|dir retrieve|prompt|infer ...` over CSV batches.

Dry / no-GPU: construction is cheap; `infer` raises a clear error if the
CLI subprocess fails (unit tests mock subprocess).
"""

from __future__ import annotations

import json
import subprocess
import sys
from pathlib import Path
from typing import Any, Dict, List, Mapping, Optional, Sequence


def _first_existing(candidates: Sequence[Path]) -> Optional[Path]:
    for p in candidates:
        if p.is_file():
            return p
    return None


def resolve_vcworld_jsons(graph_dir: Path | str) -> Dict[str, Optional[Path]]:
    """Map drug-sim / gene-sim / drug-desc / gene-desc under graph_dir/VCWorld/."""
    root = Path(graph_dir) / "VCWorld"

    def _glob(pattern: str):
        if not root.is_dir():
            return []
        return sorted(root.glob(pattern))

    return {
        "drug_sim": _first_existing(
            [
                root / "combined_similarity_sorted.json",
                root / "combined_similarity_sorted",
                *_glob("*combined_similarity_sorted*.json"),
            ]
        ),
        "gene_sim": _first_existing(
            [
                root / "results_close_gene.json",
                root / "results_close_gene",
                *_glob("*results_close_gene*.json"),
            ]
        ),
        "drug_desc": _first_existing(
            [
                root / "drug_simp.json",
                root / "drug_simp",
                *_glob("*drug_simp*.json"),
            ]
        ),
        "gene_desc": _first_existing(
            [
                root / "gene_output.json",
                root / "gene_output",
                *_glob("*gene_output*.json"),
            ]
        ),
        "kg_nodes": _first_existing([root / "KG" / "nodes.json"]),
        "kg_edges": _first_existing([root / "KG" / "edges.json"]),
        "kg_graph": _first_existing([root / "KG" / "graph.json"]),
    }


class CLIVCWorldAdapter:
    """Subprocess adapter calling VCWorld `src/cli_pipeline/cli.py`."""

    def __init__(
        self,
        vcworld_root: Path | str,
        graph_dir: Path | str,
        beta_model: str,
        work_dir: Path | str,
        python_exe: Optional[str] = None,
    ):
        self.vcworld_root = Path(vcworld_root)
        self.graph_dir = Path(graph_dir)
        self.beta_model = beta_model
        self.work_dir = Path(work_dir)
        self.work_dir.mkdir(parents=True, exist_ok=True)
        self.python_exe = python_exe or sys.executable
        self.cli_dir = self.vcworld_root / "src" / "cli_pipeline"
        self.cli_py = self.cli_dir / "cli.py"
        self.jsons = resolve_vcworld_jsons(self.graph_dir)
        self._last_retrieve: Dict[str, Any] = {}

    def _base_cmd(self) -> List[str]:
        return [self.python_exe, str(self.cli_py)]

    def _run(self, args: Sequence[str], timeout: Optional[float] = 600) -> subprocess.CompletedProcess:
        if not self.cli_py.is_file():
            raise FileNotFoundError(
                f"VCWorld CLI not found at {self.cli_py}. "
                "Set paths.vcworld_root to the VCWorld checkout."
            )
        cmd = self._base_cmd() + list(args)
        try:
            return subprocess.run(
                cmd,
                cwd=str(self.cli_dir),
                capture_output=True,
                text=True,
                timeout=timeout,
                check=False,
            )
        except FileNotFoundError as e:
            # Missing interpreter / cli — return a failed CompletedProcess-like result
            return subprocess.CompletedProcess(cmd, returncode=127, stdout="", stderr=str(e))
        except subprocess.TimeoutExpired as e:
            raise RuntimeError(f"VCWorld CLI timed out: {' '.join(cmd)}") from e

    def _task_flag(self, query: Mapping[str, Any]) -> str:
        # Prefer DE unless query explicitly asks for dir-only
        task = str(query.get("task", query.get("mode", "de"))).lower()
        return "dir" if task in ("dir", "direction") else "de"

    def retrieve(self, query: Dict[str, Any]) -> Dict[str, Any]:
        """Call `cli.py <de|dir> retrieve` (batch-style) or synthesize path refs."""
        task = self._task_flag(query)
        out_json = self.work_dir / f"retrieve_{task}_{query.get('drug','q')}.json"
        args = [
            task,
            "retrieve",
            "--drug",
            str(query.get("drug", "")),
            "--gene",
            str(query.get("gene", "")),
            "--out",
            str(out_json),
        ]
        # Attach similarity / desc paths when present
        for flag, key in (
            ("--drug-sim", "drug_sim"),
            ("--gene-sim", "gene_sim"),
            ("--drug-desc", "drug_desc"),
            ("--gene-desc", "gene_desc"),
        ):
            p = self.jsons.get(key)
            if p is not None:
                args.extend([flag, str(p)])

        proc = self._run(args)
        payload: Dict[str, Any] = {
            "query": dict(query),
            "task": task,
            "returncode": proc.returncode,
            "stdout": proc.stdout,
            "stderr": proc.stderr,
            "paths": {k: (str(v) if v else None) for k, v in self.jsons.items()},
        }
        if out_json.is_file():
            try:
                payload["retrieved"] = json.loads(out_json.read_text())
            except json.JSONDecodeError:
                payload["retrieved"] = {"raw": out_json.read_text()[:2000]}
        elif proc.returncode != 0:
            # Best-effort: keep paths for prompt stage even if retrieve CLI differs
            payload["retrieved"] = {"edges": [], "note": "retrieve CLI failed or unsupported"}
        else:
            payload["retrieved"] = {"edges": [], "stdout": proc.stdout}
        self._last_retrieve = payload
        return payload

    def prompt(self, ctx: str, query: Dict[str, Any], retrieved: Dict[str, Any]) -> str:
        """Prefer `cli.py single prompt` when available; else `<de|dir> prompt`."""
        task = self._task_flag(query)
        out_path = self.work_dir / f"prompt_{task}.txt"
        # Try single-query entry first
        single_args = [
            "single",
            "prompt",
            "--ctx",
            str(ctx),
            "--drug",
            str(query.get("drug", "")),
            "--gene",
            str(query.get("gene", "")),
            "--cell",
            str(query.get("cell", "")),
            "--out",
            str(out_path),
        ]
        proc = self._run(single_args)
        if proc.returncode != 0:
            args = [
                task,
                "prompt",
                "--ctx",
                str(ctx),
                "--drug",
                str(query.get("drug", "")),
                "--gene",
                str(query.get("gene", "")),
                "--out",
                str(out_path),
            ]
            proc = self._run(args)
        if out_path.is_file():
            return out_path.read_text()
        if proc.stdout.strip():
            return proc.stdout
        # Fallback local prompt (still valid for dry wiring)
        return (
            f"CTX={ctx}|cell={query.get('cell')}|drug={query.get('drug')}|"
            f"gene={query.get('gene')}|retrieved_keys={list((retrieved or {}).keys())}"
        )

    def infer(self, prompt: str, temperature: float) -> str:
        """Call `cli.py ... infer` with model path. Raises clear error on failure."""
        prompt_file = self.work_dir / "infer_prompt.txt"
        out_file = self.work_dir / "infer_out.txt"
        prompt_file.write_text(prompt)
        # Prefer single infer, then de infer
        attempts = [
            [
                "single",
                "infer",
                "--prompt-file",
                str(prompt_file),
                "--model",
                self.beta_model,
                "--temperature",
                str(temperature),
                "--out",
                str(out_file),
            ],
            [
                "de",
                "infer",
                "--prompt-file",
                str(prompt_file),
                "--model",
                self.beta_model,
                "--temperature",
                str(temperature),
                "--out",
                str(out_file),
            ],
        ]
        last: Optional[subprocess.CompletedProcess] = None
        for args in attempts:
            last = self._run(args)
            if out_file.is_file() and out_file.stat().st_size > 0:
                return out_file.read_text()
            if last.returncode == 0 and last.stdout.strip():
                return last.stdout
        err = (last.stderr if last else "") or (last.stdout if last else "")
        raise RuntimeError(
            "VCWorld CLI infer failed (GPU/model may be unavailable). "
            f"model={self.beta_model!r} cli={self.cli_py} detail={err[:800]!r}"
        )


def build_adapter(cfg: Mapping[str, Any], stub: bool = False):
    """Factory: StubAdapter (tests/dry) or CLIVCWorldAdapter from config paths."""
    if stub:
        from c4_cascade_rl.collect import StubAdapter

        return StubAdapter()
    paths = cfg.get("paths", {}) if isinstance(cfg, Mapping) else {}
    work = Path(paths.get("runs_dir", "runs")) / "vcworld_work"
    return CLIVCWorldAdapter(
        vcworld_root=paths.get("vcworld_root", "VCWorld"),
        graph_dir=paths.get("zenodo_graph", "data/graph"),
        beta_model=paths.get("beta_model", "Qwen/Qwen2.5-7B-Instruct"),
        work_dir=work,
        python_exe=str(cfg.get("python_exe", paths.get("python_exe", "python"))),
    )
