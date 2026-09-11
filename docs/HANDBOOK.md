# Candidate 4 Cascade RL — Experiment Handbook (Runbook)

## Hard locks

1. **Top-k ablation** = first *k* schema hops in **path order** (not GAT attention / IG). Default `k=2`; `k=1` only if Gate B fails.
2. **No** potential-based shaping / MI / per-hop reward until Gate B passes **and** L1 moved the ablation slope.
3. **E2b** = Morgan/Tanimoto drug cluster hold-out; **never** average E1 with E2b.

## Schema

- Hop line: `HOP=SRC|REL|DST|SIGN` with `SIGN ∈ {+1,-1,0}`.
- Stop line: `STOP|DE={0,1}|DIR={+1,-1}`.
- Invalid line → **drop whole trajectory**. No LLM repair.

## Reward (STOP only)

```
R_task = 1[DIR correct] + α · 1[DE correct]
R_abl  = ReLU(R_task − R_muted)   # muted = delete first k hops in path order
R      = R_task + λ_abl·R_abl − λ_len·len − λ_hall·1[hall]
```

Defaults: `α=0.5`, `λ_abl=1.0`, `λ_len=0.05`, `λ_hall=1.0`, horizon `H=6`.

## Contexts & temperatures

`graph_ctx ∈ {true, shuffle, degree, empty}` × `temps ∈ {0.2, 0.8}`.

## Gates

| Gate | Meaning |
|------|---------|
| A1 | Official pert train counts within 1% of 128293 / 102253 / 154969 / 101707 / 163748 |
| A2 | Scaffold split + degree bottom-quartile E2 ready |
| B | ≥ `gate_B_min_flip` (3000) trajectories flip under path-order mute |
| C | L0 BC legal-mask accuracy / loss thresholds |
| D | L1 filtered-BC improves ablation slope vs L0 |
| E | IQL π entropy / KL-to-L0 sanity |

## Splits

- `official_pert.json` cell/drug/gene partitions.
- GeneTAK parquet: `cell,drug,gene,y_de,y_dir,logfc,fdr`.
- Scaffold: Morgan r=2, 2048 bits, average-linkage, Tanimoto distance 0.6 (~20% drugs); recut 0.5 if <30 hold-out drugs.
- `degree.json`: bottom-quartile degree → E2.
- E2b: Tanimoto threshold 0.4 cluster hold-out.

## Weeks 1–7

### Week 1 — Prepare
Load GeneTAK + Zenodo graph; build official splits; Gate A1/A2; write under `runs/W1/`.

### Week 2 — Collect
Injectable VCWorld adapter: retrieve → prompt → infer. 4 ctx × 2 temps, stratified 20k cap. Buffer parquet + Gate B.

### Week 3 — Hop policies
- L0: BC on all legal trajectories.
- L1: filtered BC (`DIR` ok & `R_abl>0`).
- L2: AWAC with `η ∈ {0.5,1,2}`.
Gate C / D.

### Week 4 — IQL
Expectile V, Q backup, π with BC KL to L0. Gate E entropy check. `runs/W4/`.

### Week 5 — LLM LoRA
- L5a: SFT loss on verbalization + tail only (`lora_rank=16`).
- L5b: DPO on `R_abl`.
- L5c: task-only DPO.
No PPO/GRPO. `--dry-run` without GPU.

### Week 6 — Ablation eval
Retain fractions `{1.0,0.75,0.5,0.25,0.0}` on Path N. Metrics E1/E2/E2b/E3/E4/E5 → `summary.json`. **Never average E1 with E2b.**

### Week 7 — Path N + report
Freeze hop policy → 8B verbalize adapter; drop extra edges; final tables under `runs/W7/`.

## Path N

Greedy hops → freeze subgraph → verbalize with LoRA adapter → evaluate retain sweep.

## Adapter contract (VCWorld)

```python
class VCWorldAdapter:
    def retrieve(self, query) -> dict: ...
    def prompt(self, ctx, query) -> str: ...
    def infer(self, prompt, temperature) -> str: ...
```

Inject at collect time; package never vendors VCWorld source.
