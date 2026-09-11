# c4_cascade_rl — Candidate 4 Cascade RL

Offline hop-schema cascade RL research package wrapping **VCWorld** (does not fork it).
Hard locks: top-k = first k schema hops in **path order**; no potential-based shaping / MI until Gate B + L1 ablation slope; E2b = Morgan/Tanimoto hold-out (never average with E1).

## Day 0 setup (external data — do not commit)

1. Clone GENTEL-lab/VCWorld into a sibling path (default `paths.vcworld_root` in `configs/default.yaml`).
2. Download GeneTAK (Drive) → `data/genetak/`.
3. Download Zenodo graph `10.5281/zenodo.18513982` → `data/graph/`.
4. Pin base LLM: `Qwen/Qwen2.5-7B-Instruct` or `meta-llama/Llama-3-8B-Instruct`.

This package wraps the VCWorld CLI via an injectable adapter; it does **not** fork VCWorld.

## Install

```bash
cd /workspace/c4_cascade_rl
pip install -e ".[dev]"
# or: pip install -r requirements.txt && export PYTHONPATH=src
```

## Week commands

```bash
# Week 1 — prepare splits / Gate A
python scripts/week1_prepare.py --config configs/default.yaml

# Week 2 — collect rollouts (needs VCWorld adapter)
python scripts/collect.py --config configs/default.yaml --n 20000

# Week 3 — train hop policy (BC / filtered BC / AWAC)
python scripts/train_hop.py --config configs/default.yaml --algo bc --level L0

# Week 4–5 — LLM LoRA SFT / DPO (use --dry-run without GPU)
python scripts/train_llm.py --config configs/default.yaml --stage L5a --dry-run

# Week 6–7 — ablation eval
python scripts/eval.py --config configs/default.yaml --retain 1.0,0.75,0.5,0.25,0.0
```

Or via CLI: `c4 week1|collect|train|eval ...`

## Do-not list (from handbook)

- Do **not** use GAT attention / Integrated Gradients for top-k ablation.
- Do **not** add potential-based shaping, mutual information, or per-hop rewards until Gate B passes **and** L1 moves the ablation slope.
- Do **not** average E1 with E2b.
- Do **not** clone/fork VCWorld into this repo; inject an adapter.
- Do **not** LLM-repair invalid hop lines — drop the whole trajectory.
- Do **not** run PPO/GRPO for LLM stages (SFT + DPO only).
- Default `k_hops=2`; drop to `k=1` only if Gate B fails.

## Tests

```bash
cd /workspace/c4_cascade_rl && pytest -q
```

Synthetic fixtures only; rdkit/GPU/GeneTAK/VCWorld optional (tests skip if missing).

## License

MIT — research scaffolding.
