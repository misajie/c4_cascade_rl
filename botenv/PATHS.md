# 路径与入口

## 入口
- 优先 Mac → `ssh myserver`（Host 见 `~/.ssh/config`）
- Windows `DESKTOP-IEFI6BG` 可作备用；Mac Shell 异常时再换

## 项目根
- Mac: `/Users/jj/projects/vcrl/c4_cascade_rl`
- myserver: `/public/home/jjtianhkuhpc1/vcrl/c4_cascade_rl`（`~/vcrl/c4_cascade_rl`）
- VCWorld 代码: `~/vcrl/VCWorld`
- 图数据: `~/vcrl/c4_cascade_rl/data/graph/VCWorld/`（KG 在 `.../KG/`）
- GeneTAK: `~/vcrl/c4_cascade_rl/data/genetak/`（`GeneTak/` CSV + `data/*.h5ad`）
- 配置: `configs/server.yaml`（服务器绝对路径）
- β 模型: `/public/home/jjtianhkuhpc1/mnt/models/Qwen/Qwen2.5-7B-Instruct`

## Replogle / Norman（跨场景 RL；VGAE 已落盘）
- K562 essential:
  - `~/VGAE/data/replogle_k562_essential/perturb_processed.h5ad`
  - `=/public/home/jjtianhkuhpc1/VGAE/data/replogle_k562_essential/perturb_processed.h5ad`
- RPE1 essential:
  - `~/VGAE/data/replogle_rpe1_essential/perturb_processed.h5ad`
  - `=/public/home/jjtianhkuhpc1/VGAE/data/replogle_rpe1_essential/perturb_processed.h5ad`
- Norman（后续，非 v1 必做）:
  - `~/VGAE/data/norman19/perturb_processed.h5ad`
  - `=/public/home/jjtianhkuhpc1/VGAE/data/norman19/perturb_processed.h5ad`
- 跨场景 RL 输出建议: `~/vcrl/c4_cascade_rl/runs/rl_cross_context/`

## venv
- `source ~/venv/bin/activate`
- `export PYTHONNOUSERSITE=1`
- torch 期望: `2.4.0+cu121` under `~/venv/.../site-packages/torch`
