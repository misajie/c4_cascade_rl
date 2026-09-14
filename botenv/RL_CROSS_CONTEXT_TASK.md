# 任务：跨场景迁移的 RL 主动扰动选择

**任务类型**：课程作业主线，后续可扩展为论文。

**核心约束**：RL 必须是核心方法；不得把结果改写成普通主动学习论文。第一版不引入 LLM、扩散模型、知识图谱或追加细胞动作。

## 目标

在源细胞背景已有扰动图谱、目标细胞背景只有少量已测扰动的情况下，训练 RL 策略逐轮选择下一个 target perturbation，使固定 held-out target 条件的响应预测误差在相同预算下尽快下降。

建议标题：`RL for Adaptive Perturbation Selection under Cross-Context Transfer`。

## 第一阶段数据

1. 主数据：Replogle K562 essential 与 RPE1 essential，先做 K562→RPE1，再做 RPE1→K562。
2. 预测器：ridge/linear source→target delta predictor；优先复用现有实现，但重新确认输入和 split 没有泄漏。
3. 暂不处理 GeneTAK/Tahoe 全量 118 GB，也不恢复暂停中的 anndata/scanpy 安装。
4. Norman2019 和 GeneTAK 只在第一版闭环通过后扩展。

## MDP 定义

扰动候选集合为 \(P\)，总实验预算为 \(B\)，初始 pilot 已消耗 \(b_{pilot}\)。第 \(t\) 步剩余预算：

\[
b_t = B-b_{pilot}-\sum_{i<t} c(a_i).
\]

第一版每个 perturbation 成本统一为 1，因此 \(b_t=B-b_{pilot}-t\)。状态：

\[
s_t=(D_t, X_s, X_t^{ctrl}, \hat f_t, U_t, b_t),
\]

其中 \(D_t\) 是已经获得的 target 反馈，\(X_s\) 是 source 响应，\(X_t^{ctrl}\) 是 target control，\(\hat f_t\) 是当前迁移预测器，\(U_t\) 是允许使用的不确定性摘要，\(b_t\) 是剩余预算。

动作：

\[
a_t\in P\setminus D_t,\qquad c(a_t)=1.
\]

只有 \(b_t\ge1\) 时允许选择动作；预算耗尽后 episode 终止。动作选择后，从预先固定的 target acquisition pool 无放回揭示该 perturbation 的一批细胞，并更新 \(D_t\) 与 \(\hat f_t\)。

固定、永不反馈给 test policy 的审计集合 \(H\) 定义风险：

\[
L_t=|H|^{-1}\sum_{p\in H}\ell(Y^{ref}_{t,p},\hat f_t(p)).
\]

训练阶段 reward：

\[
r_t=L_t-L_{t+1},\qquad
G=\sum_{t=0}^{T-1}r_t=L_0-L_T.
\]

\(H\) 的 reference cells 只能在 meta-train 用于评分；validation 用于选超参；test reference 完全封存。策略状态、候选排序和停止规则不能读取 H 标签。

## RL 实现要求

第一版使用离散动作的 Double DQN：

- 合法动作 mask，禁止超过剩余预算；
- replay buffer、target network、epsilon schedule、固定训练 seed；
- candidate-wise encoder 或 MLP，不能用 gene ID 记忆收益；
- predictor 更新规则在所有 baseline 和 RL 中相同；
- episode 的 acquisition 队列预先随机化，各方法使用同一队列做 paired replay。

每轮逻辑：

```text
observe source + target pilot + purchased feedback + remaining budget
encode state and legal candidate actions
select perturbation with masked Q
reveal next target acquisition batch without replacement
update predictor and state
repeat until budget is exhausted
score fixed audit set only after the episode
```

## 必须实现的 baseline

- random selection；
- uncertainty sampling；
- greedy one-step VOI；
- contextual bandit（同一状态特征和近似网络容量，只优化即时 reward）；
- open-loop：开始时一次性选完全部扰动；
- RL horizon=1 与 horizon≥3。

所有方法固定 source 数据、target pilot、候选池、预算、预测器、更新频率、预处理和 acquisition 队列。

## 训练/验证/测试划分

- 按 perturbation 或 gene/family 做 condition-level 外层划分；同一条件所有细胞必须在同一分区。
- 推荐 meta-train/validation/test = 80/10/10；具体数量必须由服务器 metadata 重新核对，不得直接使用旧摘要数字。
- 每个条件内部再切 acquisition/reference cells；sample split 不是生物重复。
- 若有 well/guide/batch 字段，优先按独立实验单位切分；否则只能称 cell-budget replay。
- 预处理、HVG、PCA、标准化和超参不得使用 test target reference。

## 评估与 RL 发表扩展门槛

主指标：固定预算下 test audit 的 delta-expression MSE；报告风险-预算曲线和最终风险。辅助指标：Pearson/Spearman delta、DE top-k overlap、校准误差。统计按 perturbation/family 做 paired bootstrap，不按 gene 行或 cell 行伪造独立样本。

课程作业验收：RL 在至少一个 source→target 方向、多个预算点上优于 random，并完整复现训练/测试 mask。

论文扩展验收：

1. RL 在双向迁移和至少一个额外留出背景上稳定优于最强非 RL baseline；
2. horizon≥3 的优势超过 contextual bandit/greedy VOI，horizon=1 不应凭网络容量获得虚假优势；
3. 打乱反馈、open-loop、去掉 target feedback 的消融证明策略确实使用了序贯信息；
4. 提供跨背景迁移误差分析和至少一个可复核的生物学解释；机制结论必须有独立数据/实验支持；
5. 报告计算成本、参考池误差、数据筛选和所有历史结果重新审计情况。

## 服务器执行纪律

开始任何服务器任务前必须读取本目录全部约束。登录节点只做目录/metadata 检查、同步和提交 sbatch；代码、测试、数据处理和训练走 Slurm CPU/GPU 分区。使用共享 `~/venv`，禁止改 torch、禁止使用 `~/LLMWEDAN/.venv`，禁止把真实服务器数据下载到 Mac。当前 anndata/scanpy 安装与 c4-day0 监控保持暂停，除非用户明确恢复。

## 交付物顺序

1. `dataset_manifest.json`：obs keys、condition、control、cell counts、well/guide/batch、condition overlap；
2. `split_manifest.json`：外层 condition-level 划分和 episode seed；
3. replay cache 与无泄漏访问测试；
4. predictor + random/uncertainty/VOI/bandit baseline 曲线；
5. Double DQN 训练日志、评估曲线和 horizon/feedback 消融；
6. `REPORT.md`：方法、预算定义、结果、失败案例、是否达到论文扩展门槛。
