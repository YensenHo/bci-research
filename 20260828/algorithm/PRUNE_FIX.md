# ⚠️ 剪枝方案修正（2026-08-29 19:30 父代理修正）

channel_importance_analysis.py 的剪枝逻辑曾有 bug（order 从高到低但 keep/drop 方向反了），
已修正：**保留高重要性通道，剪掉低重要性通道**。

修正后的方案（channel_importance.json 已同步更新）：

- **保 16ch**：剪掉 6 个最低重要性 = `['C2','FC1','FCz','FC3','FC2','Fz']`（额区），保留中央运动区+顶枕区 16ch
- **保 12ch**：再剪掉 `['CP1','C1','FC4','P2']`，保留核心中央区 `['C5','C3','Cz','C4','C6','CP3','CPz','CP2','CP4','P1','Pz','POz']`

**重训时必须用修正后的 keep 通道列表**（从 channel_importance.json 的 prune_keep16/prune_keep12 读，
或重跑修正后的 channel_importance_analysis.py）。不要用旧的错误方案（曾把 CP3/CP4/Cz 列为剪除对象）。
