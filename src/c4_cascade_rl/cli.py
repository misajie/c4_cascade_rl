"""CLI entry points: week1 / collect / train / eval."""

from __future__ import annotations

import json
import warnings
from pathlib import Path
from typing import Optional

import click
import yaml


def _load_cfg(config: str):
    with open(config) as f:
        return yaml.safe_load(f)


def _genetak_dir_exists(cfg) -> bool:
    paths = cfg.get("paths", {}) or {}
    root = Path(paths.get("genetak_root", "data/genetak"))
    sub = paths.get("genetak_csv_subdir", "GeneTak")
    return (root / sub).is_dir()


@click.group()
def main():
    """c4_cascade_rl — Candidate 4 cascade RL."""
    pass


@main.command("week1")
@click.option("--config", default="configs/default.yaml", show_default=True)
@click.option(
    "--synthetic/--no-synthetic",
    default=None,
    help="Use synthetic Gate A data. Default: False if GeneTak/ exists else True.",
)
def week1_cmd(config: str, synthetic: Optional[bool]):
    """Week 1 prepare: splits + Gate A1/A2."""
    from c4_cascade_rl.gates import gate_a1, gate_a2, A1_TARGETS
    from c4_cascade_rl.splits import (
        OFFICIAL_TRAIN_COUNTS,
        build_degree_split,
        build_official_pert_from_genetak,
        discover_genetak_cells,
        gate_a1_counts_from_df,
        genetak_load_diagnostics,
        load_all_genetak_csvs,
        random_or_zero_fingerprints,
        save_official_pert,
        scaffold_cluster_holdout,
        write_genetak_parquets,
        write_synthetic_official_pert,
    )
    import numpy as np

    cfg = _load_cfg(config)
    paths = cfg.get("paths", {}) or {}
    runs = Path(paths.get("runs_dir", "runs"))
    splits_dir = Path(paths.get("splits_dir", "data/splits"))
    genetak_root = Path(paths.get("genetak_root", "data/genetak"))
    splits_dir.mkdir(parents=True, exist_ok=True)
    official = Path(paths.get("official_pert", splits_dir / "official_pert.json"))

    if synthetic is None:
        synthetic = not _genetak_dir_exists(cfg)

    a1_details_echo = {}

    if synthetic:
        write_synthetic_official_pert(official)
        counts = dict(OFFICIAL_TRAIN_COUNTS)
        a1 = gate_a1(counts, runs_dir=runs, week="W1")
        # Synthetic scaffold: enough drugs + distinct clusters so holdout >= 30
        n_drugs = 150
        rng = np.random.default_rng(0)
        fps = np.zeros((n_drugs, 128), dtype=np.uint8)
        fps[:75, :64] = (rng.random((75, 64)) > 0.5).astype(np.uint8)
        fps[75:, 64:] = (rng.random((75, 64)) > 0.5).astype(np.uint8)
        drugs = [f"D{i}" for i in range(n_drugs)]
        sc = scaffold_cluster_holdout(
            drugs, fps, distance_threshold=0.6, min_holdout=30, target_frac=0.25
        )
        deg = build_degree_split({f"N{i}": float(i) for i in range(40)}, splits_dir / "degree.json")
    else:
        cells = discover_genetak_cells(genetak_root)
        if not cells:
            raise click.ClickException(
                f"GeneTak/ not found or empty under {genetak_root}; use --synthetic"
            )
        df = load_all_genetak_csvs(genetak_root)
        write_genetak_parquets(df, splits_dir)
        pert = build_official_pert_from_genetak(df)
        counts = gate_a1_counts_from_df(df)
        diag = genetak_load_diagnostics(genetak_root, df)
        save_official_pert(
            {
                "cells": list(OFFICIAL_TRAIN_COUNTS.keys()),
                "train_counts": counts,
                "targets": dict(OFFICIAL_TRAIN_COUNTS),
                "per_cell": pert,
                "diagnostics": diag,
            },
            official,
        )
        a1 = gate_a1(counts, runs_dir=runs, week="W1")
        a1_details_echo = a1.get("details", {})

        # Scaffold from drug list — random fps if no SMILES/rdkit
        all_drugs = sorted({d for cell in pert.values() for d in cell.get("train_drugs", [])})
        if len(all_drugs) < 30:
            all_drugs = sorted(df["drug"].astype(str).unique().tolist())
        try:
            from c4_cascade_rl.splits import morgan_fingerprints_from_smiles

            # No SMILES column in GeneTAK CSVs — fall through
            raise ImportError("no smiles")
        except Exception:
            warnings.warn(
                "No SMILES/rdkit for scaffold; using random fingerprints",
                stacklevel=1,
            )
            fps = random_or_zero_fingerprints(len(all_drugs), n_bits=128, use_random=True)
        sc = scaffold_cluster_holdout(
            all_drugs,
            fps,
            distance_threshold=0.6,
            min_holdout=min(30, max(1, len(all_drugs) // 5)),
            target_frac=0.25,
        )
        (splits_dir / "scaffold.json").write_text(json.dumps({
            "n_holdout": sc["n_holdout"],
            "n_train": sc["n_train"],
            "holdout_drugs": sc["holdout_drugs"],
            "train_drugs": sc["train_drugs"],
        }, indent=2) + "\n")

        # Degree from KG nodes/edges if present
        deg = None
        from c4_cascade_rl.graph_env import load_kg_json

        attempted_kg: list[str] = []
        kg_dir = paths.get("graph_kg")
        if kg_dir:
            attempted_kg.append(str(Path(kg_dir)))
            g = load_kg_json(kg_dir)
            if g is not None and g.node_degree:
                deg = build_degree_split(g.node_degree, splits_dir / "degree.json")
        if deg is None:
            # also try zenodo_graph/VCWorld/KG
            zg = Path(paths.get("zenodo_graph", "data/graph")) / "VCWorld" / "KG"
            attempted_kg.append(str(zg))
            g = load_kg_json(zg)
            if g is not None and g.node_degree:
                deg = build_degree_split(g.node_degree, splits_dir / "degree.json")
        if deg is None:
            click.echo("KG not found; attempted paths:")
            for p in attempted_kg:
                click.echo(f"  - {p}")
            warnings.warn("KG not found; using synthetic degree split", stacklevel=1)
            deg = build_degree_split(
                {f"N{i}": float(i) for i in range(40)}, splits_dir / "degree.json"
            )

    a2 = gate_a2(sc["n_holdout"], degree_e2_ready=deg["n_e2"] > 0, runs_dir=runs, week="W1")

    # Echo A1 pass/fail with per-cell got vs target (+ DE/DIR merge diagnostics)
    lines = [f"A1 pass={a1['pass']}"]
    details = a1.get("details") or a1_details_echo
    diag_echo = {}
    if not synthetic:
        try:
            diag_echo = genetak_load_diagnostics(genetak_root)
        except Exception:
            diag_echo = {}
    for cell, target in A1_TARGETS.items():
        d = details.get(cell, {})
        got = d.get("got")
        extra = diag_echo.get(cell, {})
        if extra:
            lines.append(
                f"  {cell}: got={got} target={target} pass={d.get('pass')} "
                f"n_de_train={extra.get('n_de_train')} "
                f"n_dir_train={extra.get('n_dir_train')} "
                f"n_merged_train={extra.get('n_merged_train')}"
            )
        else:
            lines.append(f"  {cell}: got={got} target={target} pass={d.get('pass')}")
    click.echo("\n".join(lines))
    click.echo(json.dumps({"A1": a1["pass"], "A2": a2["pass"], "n_holdout": sc["n_holdout"], "synthetic": synthetic}))


@main.command("collect")
@click.option("--config", default="configs/default.yaml")
@click.option("--n", default=200, show_default=True)
@click.option("--stub/--no-stub", default=True)
def collect_cmd(config: str, n: int, stub: bool):
    """Collect rollouts (stub adapter by default)."""
    from c4_cascade_rl.collect import StubAdapter, collect_rollouts, build_adapter
    import pandas as pd

    cfg = _load_cfg(config)
    paths = cfg.get("paths", {}) or {}
    splits_dir = Path(paths.get("splits_dir", "data/splits"))

    queries = None
    # Prefer real GeneTAK train parquet if present
    train_parquets = sorted(splits_dir.glob("*_train.parquet"))
    if train_parquets:
        frames = [pd.read_parquet(p) for p in train_parquets]
        dfq = pd.concat(frames, ignore_index=True)
        # sample up to n unique-ish rows
        if len(dfq) > n:
            dfq = dfq.sample(n=n, random_state=0)
        queries = []
        for _, r in dfq.iterrows():
            queries.append(
                {
                    "cell": str(r.get("cell", "")),
                    "drug": str(r.get("drug", "")),
                    "gene": str(r.get("gene", "")),
                    "gold_dir": int(r.get("y_dir", 1)),
                    "gold_de": int(r.get("y_de", 1)),
                }
            )

    if not queries:
        queries = [
            {
                "cell": "C32",
                "drug": f"D{i % 20}",
                "gene": f"G{i % 50}",
                "gold_dir": 1 if i % 2 == 0 else -1,
                "gold_de": i % 2,
            }
            for i in range(n)
        ]

    if stub:
        adapter = StubAdapter(flip_rate=0.5)
    else:
        adapter = build_adapter(cfg, stub=False)

    buf = Path(paths.get("buffer_dir", "data/buffer")) / "rollouts.parquet"
    runs = paths.get("runs_dir", "runs")
    df = collect_rollouts(
        queries,
        adapter,
        cfg=cfg,
        cap=n,
        runs_dir=runs,
        buffer_path=buf,
    )
    click.echo(f"wrote {len(df)} rows → {buf}; flips={int(df['n_flips'].sum())}")


@main.command("train")
@click.option("--config", default="configs/default.yaml")
@click.option("--algo", type=click.Choice(["bc", "l1", "awac", "iql", "llm"]), default="bc")
@click.option("--level", default="L0")
@click.option("--dry-run/--no-dry-run", default=True)
def train_cmd(config: str, algo: str, level: str, dry_run: bool):
    """Train hop policy or LLM LoRA stages."""
    import torch
    from c4_cascade_rl.algo_bc import BCConfig, train_bc, train_l1_filtered, train_l2_awac
    from c4_cascade_rl.algo_iql import IQLConfig, train_iql
    from c4_cascade_rl.models_hop import build_default_model
    from c4_cascade_rl.train_llm import train_llm

    cfg = _load_cfg(config)
    runs = Path(cfg.get("paths", {}).get("runs_dir", "runs"))

    if algo == "llm":
        examples = [{"prompt": "p", "verbalization": "v", "text": "t", "chosen_logp": 0.0, "rejected_logp": -1.0}] * 8
        out = train_llm(level if level.startswith("L5") else "L5a", examples, dry_run=dry_run, lora_rank=cfg.get("lora_rank", 16), out_dir=runs / "W5")
        click.echo(json.dumps(out))
        return

    B, L, A = 64, 4, 16
    node_ids = torch.randint(0, 50, (B, L))
    actions = torch.randint(0, A, (B,))
    legal = torch.ones(B, A)
    model = build_default_model(n_nodes=50, n_actions=A, d_model=32)
    bc_cfg = BCConfig(epochs=2, batch_size=16)

    if algo == "bc" or level == "L0":
        metrics = train_bc(model, node_ids, actions, legal, cfg=bc_cfg)
        from c4_cascade_rl.gates import gate_c
        gate_c(metrics["accuracy"], metrics["loss"], runs_dir=runs, week="W3")
    elif algo == "l1" or level == "L1":
        dir_ok = torch.ones(B)
        r_abl = torch.rand(B)
        metrics = train_l1_filtered(model, node_ids, actions, legal, dir_ok, r_abl, cfg=bc_cfg)
    elif algo == "awac" or level == "L2":
        adv = torch.randn(B)
        metrics = train_l2_awac(model, node_ids, actions, legal, adv, eta=1.0, cfg=bc_cfg)
    elif algo == "iql":
        l0 = build_default_model(n_nodes=50, n_actions=A, d_model=32)
        l0.load_state_dict(model.state_dict())
        rewards = torch.randn(B)
        next_ids = torch.randint(0, 50, (B, L))
        dones = torch.zeros(B)
        metrics = train_iql(model, l0, node_ids, actions, rewards, next_ids, dones, legal, cfg=IQLConfig(epochs=2, batch_size=16), runs_dir=str(runs))
    else:
        raise click.ClickException(f"unknown algo {algo}")
    click.echo(json.dumps(metrics))


@main.command("eval")
@click.option("--config", default="configs/default.yaml")
@click.option("--retain", default="1.0,0.75,0.5,0.25,0.0")
def eval_cmd(config: str, retain: str):
    """Ablation retain sweep → summary.json."""
    from c4_cascade_rl.eval_ablation import path_n_eval

    cfg = _load_cfg(config)
    runs = Path(cfg.get("paths", {}).get("runs_dir", "runs")) / "W6"
    retains = [float(x) for x in retain.split(",")]
    records = []
    for split in ("E1", "E2", "E2b", "E3", "E4", "E5"):
        for i in range(20):
            records.append(
                {
                    "split": split,
                    "gold_dir": 1,
                    "gold_de": 1,
                    "pred_dir": 1 if i % 3 else -1,
                    "pred_de": 1 if i % 2 else 0,
                    "muted_pred_dir": -1,
                    "muted_pred_de": 0,
                    "length": 4,
                }
            )
    summary = path_n_eval(records, runs, retains=retains, alpha=cfg.get("alpha", 0.5))
    click.echo(json.dumps({"splits": list(summary["splits"].keys()), "out": str(runs / "summary.json")}))


if __name__ == "__main__":
    main()
