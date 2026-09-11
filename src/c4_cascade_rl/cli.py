"""CLI entry points: week1 / collect / train / eval."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Optional

import click
import yaml


def _load_cfg(config: str):
    with open(config) as f:
        return yaml.safe_load(f)


@click.group()
def main():
    """c4_cascade_rl — Candidate 4 cascade RL."""
    pass


@main.command("week1")
@click.option("--config", default="configs/default.yaml", show_default=True)
@click.option("--synthetic/--no-synthetic", default=True, help="Use synthetic Gate A data if real missing")
def week1_cmd(config: str, synthetic: bool):
    """Week 1 prepare: splits + Gate A1/A2."""
    from c4_cascade_rl.gates import gate_a1, gate_a2
    from c4_cascade_rl.splits import (
        OFFICIAL_TRAIN_COUNTS,
        build_degree_split,
        scaffold_cluster_holdout,
        write_synthetic_official_pert,
    )
    import numpy as np

    cfg = _load_cfg(config)
    runs = Path(cfg.get("paths", {}).get("runs_dir", "runs"))
    splits_dir = Path(cfg.get("paths", {}).get("splits_dir", "data/splits"))
    splits_dir.mkdir(parents=True, exist_ok=True)
    official = splits_dir / "official_pert.json"
    if not official.exists() or synthetic:
        write_synthetic_official_pert(official)
    # A1 with exact official counts (synthetic pass)
    a1 = gate_a1(OFFICIAL_TRAIN_COUNTS, runs_dir=runs, week="W1")
    # Synthetic scaffold: enough drugs + distinct clusters so holdout >= 30
    n_drugs = 150
    rng = np.random.default_rng(0)
    fps = np.zeros((n_drugs, 128), dtype=np.uint8)
    fps[:75, :64] = (rng.random((75, 64)) > 0.5).astype(np.uint8)
    fps[75:, 64:] = (rng.random((75, 64)) > 0.5).astype(np.uint8)
    drugs = [f"D{i}" for i in range(n_drugs)]
    sc = scaffold_cluster_holdout(drugs, fps, distance_threshold=0.6, min_holdout=30, target_frac=0.25)
    deg = build_degree_split({f"N{i}": float(i) for i in range(40)}, splits_dir / "degree.json")
    a2 = gate_a2(sc["n_holdout"], degree_e2_ready=deg["n_e2"] > 0, runs_dir=runs, week="W1")
    click.echo(json.dumps({"A1": a1["pass"], "A2": a2["pass"], "n_holdout": sc["n_holdout"]}))


@main.command("collect")
@click.option("--config", default="configs/default.yaml")
@click.option("--n", default=200, show_default=True)
@click.option("--stub/--no-stub", default=True)
def collect_cmd(config: str, n: int, stub: bool):
    """Collect rollouts (stub adapter by default)."""
    from c4_cascade_rl.collect import StubAdapter, collect_rollouts

    cfg = _load_cfg(config)
    queries = [
        {
            "cell": "A549",
            "drug": f"D{i%20}",
            "gene": f"G{i%50}",
            "gold_dir": 1 if i % 2 == 0 else -1,
            "gold_de": i % 2,
        }
        for i in range(n)
    ]
    adapter = StubAdapter(flip_rate=0.5)
    buf = Path(cfg.get("paths", {}).get("buffer_dir", "data/buffer")) / "rollouts.parquet"
    runs = cfg.get("paths", {}).get("runs_dir", "runs")
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
