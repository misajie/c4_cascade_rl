# botenv — bot 固定约束（不要靠上下文记忆）

本目录给 **SCRL SCHE Bot / SCRLcoding Bot**（及后续接手者）用：把会反复踩坑的规则写成文件。  
**每次开干 Day0 / 服务器 / 同步 / 测试之前，先读本目录。**

| 文件 | 用途 |
|------|------|
| `CONSTRAINTS.md` | 硬约束（禁止项） |
| `ROLES.md` | SCHE vs coding 分工 |
| `PATHS.md` | myserver / Mac 路径与账号 |
| `STATUS.md` | 当前暂停点 / 待用户指示项（可变） |

改约束时：先改这里，再同步到 myserver，必要时 commit。

当前主线：**跨场景 RL（K562↔RPE1）**（见根目录 `docs/RL_CROSS_CONTEXT_TASK.md`）。C4 Day0 / anndata deprep 保持 paused。

