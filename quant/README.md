# 海陆栅格重建的单图空间诊断

## 0. 怎么跑

```powershell
$py = "$env:USERPROFILE\.cache\codex-runtimes\codex-primary-runtime\dependencies\python\python.exe"
& $py quantify.py        # 计算全部诊断指标
& $py make_report.py     # 生成 outputs/report.md
```

输入只有两类，都在仓库内，随附 SHA-256：

- `refs/*.csv` —— 参照矩阵 R（由 Natural Earth 1:10m 栅格化得到，0/1）
- `submissions/*.png` —— 模型提交图 P

输出：`outputs/metrics.csv`、`metrics.json`、`report.md`、`hashes.json`、
`decode_log.json`、`reference_sensitivity.csv`。

## 1. 复现性规则

1. **每个指标都是 `(R, P)` 的确定性函数。** 唯一的随机源是分块自助法，
   种子写死在 `CONFIG["bootstrap_seed"] = 20261016`。
2. **参数预先写死在 `CONFIG`。** 不允许看到结果之后再改；任何改动都必须重跑全部提交。
3. **PNG 解码必须先 `convert("L")` 再按 `>127.5` 二值化。** 这一步是必须的：
   1-bit 灰度 PNG（PIL 里是 `mode "1"`）的数组是 `bool`，直接拿它和 127.5
   比会静默得到全 0 矩阵。`decode_log.json` 记录了每份文件的真实模式、
   位深和色阶，可复核。
4. **逐格 i.i.d. 不成立**（相邻格共享同一条规则），所以置信区间一律用
   **分块自助法**（块大小 5 格），不能按格点独立样本计算。
5. 全部输入与输出的哈希见 `hashes.json`。

## 2. 诊断指标

本目录只报告相互独立的诊断量，不把多个指标加权成跨维度综合分。

| 指标 | 含义 |
|---|---|
| 逐点准确率 `acc` | 与参照逐格一致的比例 |
| 面积加权准确率 `acc_w` | 按余弦纬度权重计算的一致率 |
| HSS | Heidke skill score，扣除随机一致后的分类技巧 |
| 平衡准确率 | `(TPR + TNR) / 2`，用于观察多数类影响 |
| 边界 F1@d | 在 d 格容差下的海岸边界 F1；主报告展示 d=1 |
| 岛屿 IoU 中位数 | 参照陆地连通分量的 IoU 中位数（每个分量等权） |
| 岛屿 IoU 面积加权 | 以参照分量面积为权重的 IoU 均值 |
| IoU≥0.5 比例 | IoU ≥ 0.5 的参照陆地分量比例 |
| 岛屿召回 | IoU ≥ 0.30 的参照陆地连通分量比例 |
| 近陆窄水体保留率 | 参照水域四连通分量中，所有格点到最近陆地的四邻接距离均不超过 2 格时，这些水域格点仍被正确判为水的比例；该定义不检验与开放海洋的连通性 |
| 质心偏移 | 按纬度面积加权的陆地质心距离 |
| 陆地格点比 | 预测陆地格点数 / 参照陆地格点数；未按纬度加权，不等同于物理面积比 |
| 边界距离 | 预测边界与参照边界之间的平均/95 分位距离 |

这些指标分别描述整体一致率、类别平衡、边界几何、岛屿结构和窄水体保留，
应结合任务分辨率、参照变体和置信区间解释，不应相加或重新归一化为单一分数。

## 3. 已发现的数据问题

1. **重复提交**：`gemini-4-argon_seto.png` 与
   `handwritten-baseline_seto.png` 的 SHA-256 完全相同。报告会保留
   `duplicate_of` 标记，便于在分析时排除重复证据。
2. **1-bit PNG 陷阱**：`gpt-6.1-sol.png` 和 `gpt-6.1-sol_seto.png`
   是 1-bit 灰度图。本流水线已正确处理并在 `decode_log.json` 记录。
3. **参照敏感性**：全球任务提供两个标签变体（里海改判水域、浮动冰架改判
   陆地），逐模型复评结果见 `reference_sensitivity.csv`。这说明分数对制图
   约定有依赖，不能只报一个数。

## 4. 文件

| 文件 | 说明 |
|---|---|
| `quantify.py` | 主程序，计算单图诊断指标 |
| `make_report.py` | 由 metrics.json 生成 report.md |
| `refs/` | 参照矩阵与变体、不确定性格图层 |
| `submissions/` | 历史待评提交（含手工基线） |
| `outputs/` | 全部诊断计算结果 |

