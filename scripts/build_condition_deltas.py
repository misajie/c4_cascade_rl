#!/usr/bin/env python3
"""Build condition_deltas.npz from K562/RPE1 h5ad (CPU job only — never on login node).

Schema:
  conditions: object[str]
  source_delta, target_delta: float64 [n_cond, n_genes]  (mean expr − control)
  control: str

Example (sbatch CPU):
  python scripts/build_condition_deltas.py \
    --source-h5ad ~/VGAE/data/replogle_k562_essential/perturb_processed.h5ad \
    --target-h5ad ~/VGAE/data/replogle_rpe1_essential/perturb_processed.h5ad \
    --split runs/rl_cross_context/split_manifest.json \
    --out runs/rl_cross_context/condition_deltas.npz
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np

# allow `python scripts/...` without install
_ROOT = Path(__file__).resolve().parents[1]
_SRC = _ROOT / "src"
if str(_SRC) not in sys.path:
    sys.path.insert(0, str(_SRC))

from rl_cross_context.h5ad_means import means_by_condition  # noqa: E402


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--source-h5ad", required=True)
    ap.add_argument("--target-h5ad", required=True)
    ap.add_argument("--split", required=True, help="split_manifest.json")
    ap.add_argument("--out", required=True)
    ap.add_argument("--condition-col", default="condition")
    ap.add_argument("--control", default="ctrl")
    ap.add_argument("--max-genes", type=int, default=0, help="0=all; else unsupported for backed")
    args = ap.parse_args()

    import anndata as ad

    split = json.loads(Path(args.split).read_text())
    pools = []
    for k in (
        "conditions_train",
        "conditions_val",
        "conditions_test",
        "acquisition_conditions",
        "reference_conditions",
        "audit_conditions",
    ):
        pools.extend(split.get(k) or [])
    conditions = sorted(set(pools))
    if args.control not in conditions:
        conditions = [args.control] + conditions

    src = ad.read_h5ad(args.source_h5ad, backed="r")
    tgt = ad.read_h5ad(args.target_h5ad, backed="r")
    if args.max_genes and args.max_genes > 0:
        raise SystemExit("--max-genes with backed h5ad not supported; omit for full run")

    src_means = means_by_condition(src, args.condition_col, conditions)
    tgt_means = means_by_condition(tgt, args.condition_col, conditions)
    ctrl_i = conditions.index(args.control)
    source_delta = src_means - src_means[ctrl_i]
    target_delta = tgt_means - tgt_means[ctrl_i]

    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(
        out,
        conditions=np.asarray(conditions, dtype=object),
        source_delta=source_delta.astype(np.float64),
        target_delta=target_delta.astype(np.float64),
        control=np.asarray(args.control),
        meta_json=np.asarray(
            json.dumps(
                {
                    "n_conditions": len(conditions),
                    "n_genes": int(source_delta.shape[1]),
                    "source_h5ad": args.source_h5ad,
                    "target_h5ad": args.target_h5ad,
                }
            )
        ),
    )
    print(
        json.dumps(
            {
                "out": str(out),
                "n_conditions": len(conditions),
                "n_genes": int(source_delta.shape[1]),
            }
        )
    )


if __name__ == "__main__":
    main()
