# 跨场景 RL 任务（K562 ↔ RPE1）

> 若 Mac 上有未提交的更新版，以 Mac 本地为准；本文件保证仓库链接可解析。

## 目标
在 Replogle K562 essential ↔ RPE1 essential 之间做 **主动获取（acquisition）RL**：用有限扰动预算提升跨场景 delta 预测，核心是 RL，不是再堆 LLM/diffusion/KG。

## 数据（已落盘 VGAE；勿下载）
- `~/VGAE/data/replogle_k562_essential/perturb_processed.h5ad`
- `~/VGAE/data/replogle_rpe1_essential/perturb_processed.h5ad`
- Norman 后续：`~/VGAE/data/norman19/perturb_processed.h5ad`

## Hard rules
1. RL is core；v1 **禁止** LLM / diffusion / KG / extra cell actions
2. cost = 1 per perturbation
3. reward = L_t − L_{t+1} on fixed audit set **H**；**H never in policy state**
4. paired acquisition queue across methods
5. 第一版禁止碰 GeneTAK 118G / 恢复 deprep，除非用户点名

## 流水线 checklist
metadata inventory → dataset_manifest → splits → baselines → DQN → risk–budget report

详见 `docs/RL_CROSS_CONTEXT_DELIVERABLES.md`。
