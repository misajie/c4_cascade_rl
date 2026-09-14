# 分工

## SCRL SCHE Bot
- GitHub push / `ssh myserver` 同步（rsync 或 pull）
- Slurm：`sbatch` pytest、prepare、extract、week1；**跨场景线**：CPU `sbatch` 跑 manifest / split / train-baselines / train-dqn
- Day0 数据：下载、解压、路径核对、把目录树丢给 coding（**当前 paused**）
- 跨场景：login 上 **只读** metadata inventory（h5ad obs keys / condition 计数），结果同步给 coding；勿改数据
- 进度汇报 / 例程监控（仅用户开启时）
- **不**触碰 GeneTAK 118G / deprep，除非用户点名

## SCRLcoding Bot
- 只写/改 `c4_cascade_rl` 代码与测试
- **跨场景线**：拥有 `src/rl_cross_context/` 脚手架与实现；本地 commit；**不**部署、不 ssh 跑实验
- 等 SCHE 的服务器目录树 / 作业结果再改 paths、Gate、adapter

## 交接口
1. coding commit → SCHE push + sync + CPU 作业
2. SCHE 丢目录树 / A1·KG 日志 / metadata inventory → coding 修
3. 环境/装包失败 → SCHE **问用户**，不自己发明策略
