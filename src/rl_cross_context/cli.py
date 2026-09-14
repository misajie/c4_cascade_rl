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
from .episode_data import assert_no_audit_in_acquisition, bundle_to_state, load_episode_bundle
from .eval_curves import summarize_method_curves
from .replay import paired_queues
from .splits import build_split_from_manifest, make_condition_splits, write_split_manifest


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
@click.option("--manifest", "manifest_path", default=None, type=click.Path(exists=True))
@click.option("--out", "out_path", default=None, type=click.Path())
@click.pass_context
def split_cmd(ctx: click.Context, manifest_path: str | None, out_path: str | None) -> None:
    cfg = ctx.obj["cfg"]
    out = _outdir(cfg)
    man_path = Path(manifest_path) if manifest_path else out / "dataset_manifest.json"
    if man_path.exists() and "condition_name_overlap" in json.loads(man_path.read_text()):
        split = build_split_from_manifest(
            man_path,
            train_frac=cfg.train_frac,
            val_frac=cfg.val_frac,
            test_frac=cfg.test_frac,
            acquisition_frac=cfg.acquisition_frac,
            audit_frac=cfg.audit_frac,
            seed=cfg.seed,
        )
    elif man_path.exists():
        data = json.loads(man_path.read_text())
        conditions = data["source"]["conditions"]
        control = data["source"].get("control_condition", "ctrl")
        split = make_condition_splits(
            conditions,
            control=control,
            train_frac=cfg.train_frac,
            val_frac=cfg.val_frac,
            test_frac=cfg.test_frac,
            acquisition_frac=cfg.acquisition_frac,
            audit_frac=cfg.audit_frac,
            seed=cfg.seed,
            source_context=data["source"].get("context"),
            target_context=(data.get("target") or {}).get("context"),
        )
    else:
        man, _ = build_synthetic_manifest(
            source=cfg.source_context,
            target=cfg.target_context,
            n_conditions=cfg.n_conditions,
            n_genes=cfg.n_genes,
            seed=cfg.seed,
        )
        write_manifest(man, out / "dataset_manifest.json")
        split = make_condition_splits(
            man.source.conditions,
            control=man.source.control_condition,
            train_frac=cfg.train_frac,
            val_frac=cfg.val_frac,
            test_frac=cfg.test_frac,
            acquisition_frac=cfg.acquisition_frac,
            audit_frac=cfg.audit_frac,
            seed=cfg.seed,
            source_context=man.source.context,
            target_context=man.target.context,
        )
    dest = Path(out_path) if out_path else out / "split_manifest.json"
    path = write_split_manifest(split, dest)
    click.echo(
        json.dumps(
            {
                "path": str(path),
                "n_train": len(split.conditions_train),
                "n_val": len(split.conditions_val),
                "n_test": len(split.conditions_test),
                "n_audit": len(split.audit_conditions),
                "n_stems_train": len(split.gene_stems_train),
            }
        )
    )


def _load_state(cfg, arrays_path: str | None = None, prefer_synthetic: bool = False):
    out = _outdir(cfg)
    bundle = load_episode_bundle(
        out,
        seed=cfg.seed,
        n_conditions=cfg.n_conditions,
        n_genes=cfg.n_genes,
        source_context=cfg.source_context,
        target_context=cfg.target_context,
        train_frac=cfg.train_frac,
        val_frac=cfg.val_frac,
        test_frac=cfg.test_frac,
        acquisition_frac=cfg.acquisition_frac,
        audit_frac=cfg.audit_frac,
        arrays_path=arrays_path,
        prefer_synthetic=prefer_synthetic,
    )
    assert_no_audit_in_acquisition(bundle)
    state, queue = bundle_to_state(bundle, seed=cfg.seed)
    return state, queue, bundle


@main.command("train-baselines")
@click.option("--arrays", "arrays_path", default=None, type=click.Path(exists=True))
@click.option("--synthetic", is_flag=True, default=False, help="Force synthetic arrays")
@click.pass_context
def train_baselines_cmd(ctx: click.Context, arrays_path: str | None, synthetic: bool) -> None:
    cfg = ctx.obj["cfg"]
    out = _outdir(cfg)
    state, _, bundle = _load_state(cfg, arrays_path=arrays_path, prefer_synthetic=synthetic)
    budget = max(cfg.budgets)
    names = list(BASELINE_REGISTRY.keys())
    queues = paired_queues(bundle.acquisition_conditions, seed=cfg.seed, n_methods=len(names))
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
        results[name] = {
            "final_loss": res["final_loss"],
            "cumulative_reward": res["cumulative_reward"],
        }
    curves = summarize_method_curves(histories, cfg.budgets)
    payload = {
        "results": results,
        "curves": curves,
        "budgets": cfg.budgets,
        "data_source": bundle.source,
        "n_acquisition": len(bundle.acquisition_conditions),
        "n_audit": len(bundle.audit_ids),
    }
    path = out / "baselines_report.json"
    path.write_text(json.dumps(payload, indent=2))
    click.echo(f"wrote {path} source={bundle.source}")


@main.command("train-dqn")
@click.option("--dry-run", is_flag=True, default=False)
@click.option("--horizon", default=3, type=int, show_default=True, help="1=myopic; ≥3 discounted")
@click.option("--arrays", "arrays_path", default=None, type=click.Path(exists=True))
@click.option("--synthetic", is_flag=True, default=False)
@click.pass_context
def train_dqn_cmd(
    ctx: click.Context,
    dry_run: bool,
    horizon: int,
    arrays_path: str | None,
    synthetic: bool,
) -> None:
    cfg = ctx.obj["cfg"]
    out = _outdir(cfg)
    state, queue, bundle = _load_state(cfg, arrays_path=arrays_path, prefer_synthetic=synthetic)
    res = train_dqn_synthetic(
        state,
        queue,
        budgets=cfg.budgets,
        episodes=cfg.dqn_episodes,
        seed=cfg.seed,
        hidden=cfg.dqn_hidden,
        lr=cfg.dqn_lr,
        dry_run=dry_run,
        horizon=horizon,
        alpha=cfg.ridge_alpha,
        eval_every=getattr(cfg, "dqn_eval_every", 25),
        eps_start=cfg.dqn_eps_start,
        eps_end=cfg.dqn_eps_end,
        eps_decay=cfg.dqn_eps_decay,
        buffer_size=cfg.dqn_buffer_size,
    )
    res["data_source"] = bundle.source
    path = out / f"dqn_report_h{horizon}.json"
    path.write_text(json.dumps(res, indent=2))
    # also write canonical name for horizon=3
    if horizon == 3:
        (out / "dqn_report.json").write_text(json.dumps(res, indent=2))
    click.echo(
        f"wrote {path} dry_run={res['dry_run']} torch={res['torch_available']} source={bundle.source}"
    )


@main.command("report")
@click.pass_context
def report_cmd(ctx: click.Context) -> None:
    cfg = ctx.obj["cfg"]
    out = _outdir(cfg)
    parts = {}
    for name in (
        "baselines_report.json",
        "dqn_report.json",
        "dqn_report_h1.json",
        "dataset_manifest.json",
        "split_manifest.json",
    ):
        p = out / name
        if p.exists():
            parts[name] = json.loads(p.read_text())
    path = out / "summary_report.json"
    path.write_text(json.dumps({"seed": cfg.seed, "parts": list(parts.keys()), "data": parts}, indent=2))
    click.echo(f"wrote {path} keys={list(parts.keys())}")


if __name__ == "__main__":
    main()
