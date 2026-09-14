# Hard constraints（违反即错）

## myserver / Slurm
1. **禁止**在 login / mgmt 节点跑代码、`pip install`、`pytest`、训练。login 只做 git sync / `rsync` / `sbatch`。
2. 测试与需要算力的安装走 **CPU 分区作业**：`partition=cpu`，`account=zhangqingpeng`，`qos=user_jjtianhkuhpc1`。
3. **禁止**修改或重装环境里的 torch（勿 `pip install/overwrite` site CUDA torch）。torch 保持 optional；用 `--no-deps` / 现有 `~/venv` torch。装完核对 `torch.__version__` 与 `torch.__file__` 未变。
4. 真实 GeneTAK / Zenodo / VCWorld 数据只在服务器；不要假设本机有 data env。
5. **一个稳定共享 venv**：`~/venv`。禁止为 c4 再建会重装 torch 的 per-project `.venv`。禁止使用 `~/LLMWEDAN/.venv`（那是 ServerBot/VGAE 的）。

## 本机下载 / 装包策略
6. **禁止**为解决服务器环境问题，在 Mac/本机下载 wheel、数据集或改离线安装策略后擅自推进。  
   环境装不上（例如 anndata/scanpy/llvmlite 在计算节点 PyPI 卡住）→ **先告诉用户，等指示**。不要自作主张换安装通道。
7. Day0 数据下载：若用代理流式上传服务器，也须符合用户当时指示；默认不要把大数据落盘在 Mac。

## Bot 行为
8. Repo push、myserver 同步、pytest 作业、Day0 下载/解压 = **SCHE**；coding bot 只交代码 commit/patch。
9. 用户说「停 / 暂停监控」→ 立刻停作业推进与例程监控，等明确恢复。
10. 开干前读 `botenv/`；刷新上下文后仍以本目录为准，不以聊天摘要猜。
