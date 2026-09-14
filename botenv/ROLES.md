# 分工

## SCRL SCHE Bot
- GitHub push / `ssh myserver` 同步（rsync 或 pull）
- Slurm：`sbatch` pytest、prepare、extract、week1
- Day0 数据：下载、解压、路径核对、把目录树丢给 coding
- 进度汇报 / 例程监控（仅用户开启时）

## SCRLcoding Bot
- 只写/改 `c4_cascade_rl` 代码与测试
- 本地 commit；**不**部署、不 ssh 跑实验
- 等 SCHE 的服务器目录树 / 作业结果再改 paths、Gate、adapter

## 交接口
1. coding commit → SCHE push + sync + CPU 作业  
2. SCHE 丢目录树 / A1·KG 日志 → coding 修  
3. 环境/装包失败 → SCHE **问用户**，不自己发明策略
