# 当前状态（可变；更新时改本文件）

**更新**: 2026-09-14

## 暂停中
- Day0 半小时监控例程 `c4-day0`：**paused**
- anndata/scanpy 官方 C32 `de prepare`：**停装，等用户指示**（禁止本机下 wheel）
- 勿擅自重启 deprep / 改安装策略

## 已完成要点
- 代码已上 GitHub（含 KG `entity_id` / `src_id`/`dst_id` 等修复）；week1 曾到 **synthetic=false**
- pytest 曾 **37 passed, 1 skipped**
- A1 仍 false：shipped DE train ≪ Table 5（C32 44708 vs 128293）；等官方 prepare 对比或冻结 target
- GeneTAK / graph 已落盘并摊平；zip 内无更大 CSV

## 恢复时检查清单
1. 读完 `CONSTRAINTS.md`
2. 问用户：anndata/scanpy 怎么装？
3. 用户同意后再 `sbatch`；装完核对 torch 未变
4. 出 `prepared_official` 对比表 → 丢 coding → 再决定 A1 target
5. 用户说开监控再 `resume` 例程 `c4-day0`
