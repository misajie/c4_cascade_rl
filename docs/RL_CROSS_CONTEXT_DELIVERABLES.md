# RL Cross-Context Deliverables (K562 ↔ RPE1)

Source task: `docs/RL_CROSS_CONTEXT_TASK.md` (local / uncommitted on Mac may be newer).

Hard rules (v1):
- **RL is core**; no LLM / diffusion / KG / extra cell actions
- **cost = 1** per perturbation
- **reward = L_t − L_{t+1}** on fixed audit set **H**; **H never enters policy state**
- **Paired acquisition queue** across methods (same reveal order / shared pool shuffle)
- Do **not** touch GeneTAK 118G / deprep unless user names it

## Six deliverables

| # | Deliverable | Artifact / module |
|---|-------------|-------------------|
| 1 | **Metadata inventory** | Read-only obs keys / conditions / control / counts (SCHE on login). Feeds manifest. |
| 2 | **`dataset_manifest.json`** | Schema + write path via `data_manifest.py` / `cli manifest` |
| 3 | **`split_manifest.json`** | Condition-level 80/10/10 + acquisition / reference + fixed audit **H** (`splits.py`) |
| 4 | **Baselines (paired queue)** | random, uncertainty, greedy-VOI stub, contextual-bandit stub, open-loop (`baselines.py`, `replay.py`) |
| 5 | **Double DQN + legal action mask** | `dqn.py` / `cli train-dqn` (`--dry-run` CPU synthetic; torch optional) |
| 6 | **Risk–budget curves + report** | `eval_curves.py` / `cli report` → `baselines_report.json`, `dqn_report.json`, `summary_report.json` |

## Suggested run order

```bash
pip install -e ".[dev]"
python -m rl_cross_context.cli --config configs/rl_cross_context.yaml manifest
python -m rl_cross_context.cli --config configs/rl_cross_context.yaml split
python -m rl_cross_context.cli --config configs/rl_cross_context.yaml train-baselines
python -m rl_cross_context.cli --config configs/rl_cross_context.yaml train-dqn --dry-run
python -m rl_cross_context.cli --config configs/rl_cross_context.yaml report
pytest -q tests/rl_cross_context
```

Output dir suggestion: `~/vcrl/c4_cascade_rl/runs/rl_cross_context/` (see `botenv/PATHS.md`).
