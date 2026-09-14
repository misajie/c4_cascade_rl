"""click CLI: manifest | split | train-baselines | train-dqn | report."""

from __future__ import annotations

import json
from pathlib import Path

import click
import numpy as np

from .baselines import BASELINE_REGISTRY, BaselineState, run_episode
from .config import load_config
from .data_manifest import build_synthetic_manifest, write_manifest
from .dqn import train_dqn_synthetic
from .eval_curves import summarize_method_curves
from .predictor import condition_means, deltas_from_control
from .replay import AcquisitionQueue, paired_queues
from .splits import make_condition_splits, write_split_manifest


def _outdir(cfg) -> Path:
    p = Path(cfg.output_dir)
    p.mkdir(parents=True, exist_ok=True)
    return p


@click.group()
@click.option("--config", "config_path", default=None, type=click.Path())
@click.option("--seed", default=None, type=int)
@click.pass_context
def main(ctx: click.Context, config_path: str | None, seed: int | None) -> None:
    overrides = {}
    if seed is not None:
        overrides["seed"] = seed
    cfg = load_config(config_path, **overrides)
    ctx.ensure_object(dict)
    ctx.obj["cfg"] = cfg


@main.command("manifest")
@click.option("--synthetic/--real", default=True, help="v1 tests use synthetic; real needs server h5ad")
@click.pass_context
def manifest_cmd(ctx: click.Context, synthetic: bool) -> None:
    cfg = ctx.obj["cfg"]
    out = _outdir(cfg)
    if not synthetic:
        raise click.ClickException("Real h5ad inventory is SCHE-only; use --synthetic locally")
    man, arrays = build_synthetic_manifest(
        source=cfg.source_context,
        target=cfg.target_context,
        n_conditions=cfg.n_conditions,
        n_genes=cfg.n_genes,
        seed=cfg.seed,
    )
    path = write_manifest(man, out / "dataset_manifest.json")
    np.savez_compressed(
        out / "synthetic_arrays.npz",
        source_X=arrays["source_X"],
        target_X=arrays["target_X"],
        source_cond_ids=arrays["source_cond_ids"],
        target_cond_ids=arrays["target_cond_ids"],
        conditions=np.array(arrays["conditions"]),
    )
    click.echo(f"wrote {path}")


@main.command("split")
@click.pass_context
def split_cmd(ctx: click.Context) -> None:
    cfg = ctx.obj["cfg"]
    out = _outdir(cfg)
    man_path = out / "dataset_manifest.json"
    if not man_path.exists():
        # build synthetic on the fly
        man, _ = build_synthetic_manifest(
            source=cfg.source_context,
            target=cfg.target_context,
            n_conditions=cfg.n_conditions,
            n_genes=cfg.n_genes,
            seed=cfg.seed,
        )
        write_manifest(man, man_path)
        conditions = man.source.conditions
        control = man.source.control_condition
    else:
        data = json.loads(man_path.read_text())
        conditions = data["source"]["conditions"]
        control = data["source"]["control_condition"]
    split = make_condition_splits(
        conditions,
        control=control,
        train_frac=cfg.train_frac,
        val_frac=cfg.val_frac,
        test_frac=cfg.test_frac,
        acquisition_frac=cfg.acquisition_frac,
        audit_frac=cfg.audit_frac,
        seed=cfg.seed,
    )
    path = write_split_manifest(split, out / "split_manifest.json")
    click.echo(f"wrote {path}")


def _load_state(cfg) -> tuple[BaselineState, AcquisitionQueue, list[str]]:
    out = _outdir(cfg)
    man, arrays = build_synthetic_manifest(
        source=cfg.source_context,
        target=cfg.target_context,
        n_conditions=cfg.n_conditions,
        n_genes=cfg.n_genes,
        seed=cfg.seed,
    )
    conditions = arrays["conditions"]
    control = "ctrl"
    split = make_condition_splits(
        conditions,
        control=control,
        train_frac=cfg.train_frac,
        val_frac=cfg.val_frac,
        test_frac=cfg.test_frac,
        acquisition_frac=cfg.acquisition_frac,
        audit_frac=cfg.audit_frac,
        seed=cfg.seed,
    )
    write_manifest(man, out / "dataset_manifest.json")
    write_split_manifest(split, out / "split_manifest.json")

    n_cond = len(conditions)
    src_means = condition_means(arrays["source_X"], arrays["source_cond_ids"], n_cond)
    tgt_means = condition_means(arrays["target_X"], arrays["target_cond_ids"], n_cond)
    src_delta = deltas_from_control(src_means, 0)
    tgt_delta = deltas_from_control(tgt_means, 0)
    cond_to_idx = {c: i for i, c in enumerate(conditions)}
    audit_ids = [cond_to_idx[c] for c in split.audit_conditions]
    queue = AcquisitionQueue.from_conditions(split.acquisition_conditions, seed=cfg.seed)
    state = BaselineState(
        source_delta=src_delta,
        target_delta=tgt_delta,
        revealed_ids=[],
        audit_ids=audit_ids,
        cond_to_idx=cond_to_idx,
        rng=np.random.default_rng(cfg.seed),
    )
    return state, queue, split.acquisition_conditions


@main.command("train-baselines")
@click.pass_context
def train_baselines_cmd(ctx: click.Context) -> None:
    cfg = ctx.obj["cfg"]
    out = _outdir(cfg)
    state, queue_tmpl, acq = _load_state(cfg)
    budget = max(cfg.budgets)
    names = list(BASELINE_REGISTRY.keys())
    queues = paired_queues(acq, seed=cfg.seed, n_methods=len(names))
    histories = {}
    results = {}
    for name, q in zip(names, queues):
        st = BaselineState(
            source_delta=state.source_delta,
            target_delta=state.target_delta,
            revealed_ids=[],
            audit_ids=list(state.audit_ids),
            cond_to_idx=dict(state.cond_to_idx),
            rng=np.random.default_rng(cfg.seed),
        )
        policy = BASELINE_REGISTRY[name]()
        res = run_episode(policy, q, st, budget=budget, alpha=cfg.ridge_alpha)
        histories[name] = res["history"]
        results[name] = {"final_loss": res["final_loss"], "cumulative_reward": res["cumulative_reward"]}
    curves = summarize_method_curves(histories, cfg.budgets)
    payload = {"results": results, "curves": curves, "budgets": cfg.budgets}
    path = out / "baselines_report.json"
    path.write_text(json.dumps(payload, indent=2))
    click.echo(f"wrote {path}")


@main.command("train-dqn")
@click.option("--dry-run", is_flag=True, default=False, help="CPU synthetic; random policy if no torch opt")
@click.pass_context
def train_dqn_cmd(ctx: click.Context, dry_run: bool) -> None:
    cfg = ctx.obj["cfg"]
    out = _outdir(cfg)
    state, queue, _ = _load_state(cfg)
    res = train_dqn_synthetic(
        state,
        queue,
        budgets=cfg.budgets,
        episodes=cfg.dqn_episodes,
        seed=cfg.seed,
        hidden=cfg.dqn_hidden,
        lr=cfg.dqn_lr,
        dry_run=dry_run,
    )
    path = out / "dqn_report.json"
    path.write_text(json.dumps(res, indent=2))
    click.echo(f"wrote {path} dry_run={res['dry_run']} torch={res['torch_available']}")


@main.command("report")
@click.pass_context
def report_cmd(ctx: click.Context) -> None:
    cfg = ctx.obj["cfg"]
    out = _outdir(cfg)
    parts = {}
    for name in ("baselines_report.json", "dqn_report.json", "dataset_manifest.json", "split_manifest.json"):
        p = out / name
        if p.exists():
            parts[name] = json.loads(p.read_text())
    path = out / "summary_report.json"
    path.write_text(json.dumps({"seed": cfg.seed, "parts": list(parts.keys()), "data": parts}, indent=2))
    click.echo(f"wrote {path} keys={list(parts.keys())}")


if __name__ == "__main__":
    main()
