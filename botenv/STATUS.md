# 当前状态（可变；更新时改本文件）

**更新**: 2026-09-14

## 活跃任务（主线）
- **跨场景 RL（K562↔RPE1）** — 见 `docs/RL_CROSS_CONTEXT_TASK.md` / `docs/RL_CROSS_CONTEXT_DELIVERABLES.md`
- 代码脚手架：`src/rl_cross_context/` + `configs/rl_cross_context.yaml`
- 建议输出：`~/vcrl/c4_cascade_rl/runs/rl_cross_context/`

### 跨场景 RL checklist
- [ ] metadata inventory（obs keys / conditions / control；SCHE 只读）
- [ ] `dataset_manifest.json`
- [ ] splits（condition-level 80/10/10 + acquisition/reference）→ `split_manifest.json`
- [ ] baselines（random / uncertainty / greedy-VOI / contextual-bandit / open-loop；paired acquisition queue）
- [ ] Double DQN（legal action mask；reward = ΔL on fixed audit set H）
- [ ] risk–budget curves + report

## 暂停中（保持 pause；勿擅自恢复）
- Day0 半小时监控例程 `c4-day0`：**paused**
- anndata/scanpy 官方 C32 `de prepare`：**停装，等用户指示**（禁止本机下 wheel）
- 勿擅自重启 deprep / 改安装策略
- **跨场景第一版禁止碰 GeneTAK 118G / 恢复 deprep**，除非用户点名

## 已完成要点（C4 Day0 线，归档参考）
- 代码已上 GitHub（含 KG `entity_id` / `src_id`/`dst_id` 等修复）；week1 曾到 **synthetic=false**
- pytest 曾 **37 passed, 1 skipped**
- A1 仍 false：shipped DE train ≪ Table 5（C32 44708 vs 128293）；等官方 prepare 对比或冻结 target
- GeneTAK / graph 已落盘并摊平；zip 内无更大 CSV

## 恢复 Day0/deprep 时检查清单（仅用户明确要求后）
1. 读完 `CONSTRAINTS.md`
2. 问用户：anndata/scanpy 怎么装？
3. 用户同意后再 `sbatch`；装完核对 torch 未变
4. 出 `prepared_official` 对比表 → 丢 coding → 再决定 A1 target
5. 用户说开监控再 `resume` 例程 `c4-day0`

## RL cross-context — split (2026-09-14)
- `runs/rl_cross_context/split_manifest.json` written from `dataset_manifest.json` (seed=0)
- gene-stem 80/10/10 on 849 overlap conditions → train 678 / val 85 / test 85 stems
- acquisition 339 / reference 339 / audit 42 (audit from val∪test; never in acquisition)
- cell-level acquisition/reference indices still for SCHE CPU h5ad job

## RL cross-context — baselines/DQN wiring (2026-09-14)
- `episode_data.py`: load `condition_deltas.npz` + `split_manifest.json` (else synthetic)
- VOI baseline no longer reads audit H labels
- Double DQN is candidate-wise shared MLP; `--horizon 1|3`
- SCHE: CPU `scripts/build_condition_deltas.py` → then `train-baselines` / `train-dqn`
