#!/usr/bin/env python3
import json
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
import click
import yaml
from c4_cascade_rl.train_llm import train_llm

@click.command()
@click.option("--config", default="configs/default.yaml")
@click.option("--stage", default="L5a")
@click.option("--dry-run/--no-dry-run", default=True)
def main(config, stage, dry_run):
    with open(config) as f:
        cfg = yaml.safe_load(f)
    examples = [{"prompt": "graph", "verbalization": "path", "text": "t",
                 "chosen_logp": 0.2, "rejected_logp": -0.5}] * 16
    out = train_llm(
        stage, examples, dry_run=dry_run, lora_rank=cfg.get("lora_rank", 16),
        out_dir=Path(cfg.get("paths", {}).get("runs_dir", "runs")) / "W5",
    )
    print(json.dumps(out))

if __name__ == "__main__":
    main()
