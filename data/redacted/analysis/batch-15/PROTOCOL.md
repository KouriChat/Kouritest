# 已知偏移检出实验：H / L / H-sham

## 研究问题与有限主张

在冻结地图任务、提示词、工具链、输出上限与评分器的条件下，同一模型仅改变请求级 `reasoning.effort` 时，多维输出诊断能否识别出**超过同配置自然运行波动**的可复现行为偏移？

本实验检验的是请求级配置操纵的检测有效性，不是供应商内部预算识别，也不是通用智力或“降智”测量。结论只覆盖本实验的六幅地图、一个声明模型、一个服务路径与执行时段。

## 设计

- 正式目标服务路径：`channel-F`，声明模型 `gpt-6.1-sol`。路径选择与失败记录见文末“执行记录”；所有失败路径的归档均隔离保留，不纳入正式质量配对。
- 任务：`global,csc,dzh,seto,arctic01,hecate`。
- 三个运行组：
  - H：标准配置，`reasoning.effort=high`；
  - L：低推理配置，`reasoning.effort=low`；
  - H-sham：与 H 完全相同的 `high` 配置，但使用独立归档根和重复编号，用于估计同配置自然波动。
- 每图每组 2 次独立运行：6 × 3 × 2 = 36 个目标模型运行。
- 每幅地图内部随机交错 6 个调用（3 组 × 2 重复），整体地图顺序也随机化；固定种子 `20261029`。
- H/L 按地图和重复编号配对；H-sham 仅用于自然波动对照，不与处理效应混为一谈。
- 唯一预期变化是请求体 `reasoning.effort`。提示词来自现有 `experiment_config/tasks.json`，所有调用使用同一 `api-tools` 后端、`direct` 传输、`max_output_tokens=12000` 与相同工具集。
- 新实验写入独立目录 `experiments/known_shift_20261029/`；正式归档位于 `runs_S2/`；不覆盖、不改写既有 provider 归档和论文源文件。

## 成本估算与停止规则

依据历史 channel-OR `openai/gpt-6.1-sol` 响应中可观测的实际 `usage.cost`（n=185）：均值 0.0689 USD，中位数 0.0695，95 分位 0.1344，最大值 0.3002。

- 预算估算：36 × 0.1344 × 1.15 = 5.56 USD。
- 硬上限：10 USD；达到 9 USD 时停止新增调用，保留不平衡结果并明确报告。
- 失败、超时、重试和部分交付产生的费用全部计入；不把失败当作缺失而隐去费用。
- 所有调用的响应 JSON 保存原始 usage/cost；若服务未返回 cost，则记录 `cost=None`，不以 0 替代。
- 不因单价或结果异常而删除特定地图；若预算不足，按全图、全组平衡缩减重复数。

## 主要与次要终点

- 共同主要终点：边界 F1@1、参考陆地分量 IoU 中位数。
- 次要终点：准确率、HSS、分量格点加权 IoU、陆地召回率/精确率、交付有效率。
- 探索性终点：近陆水域保留率，并报告其有效格点数。
- 不构造事后加权总分。

## 分析方案

1. 报告每模型、每运行组的调用数、完整率、费用、token 与延迟。
2. 对 H-L 在同一地图、同一重复编号上的差值，逐地图展示点估计和原始运行值；同时给出六图等权平均差。
3. 用 H-sham 的同配置差值形成自然波动分布，报告其绝对值中位数、95 分位和最大值，并与 H-L 的效应大小和方向一致性比较。由于 sham 只有 12 个配对，这仅作为经验参照，不用于校准生产级阈值。
4. 以地图为分析单位展示异质性；不以 6 幅地图外推所有任务。
5. 同时报告成功交付质量和端到端交付可用性。质量只在 `status=ok` 且有完整评分时计算；失败/部分不填补为质量值。
6. 不以同一批数据调阈值后宣称校准检出率。结果至多支持“初步偏移检出证据”。
7. 费用以新增 API 实际 cost 为准；本地评分计算与模型推理费用分列。

## 预设判读

若以下条件同时满足，则称本实验提供了“有限的支持信号”：

- 两个共同主要终点的六图平均 H-L 差值方向为负；
- 至少 5/6 地图在两个主要终点的方向上一致为负；
- 平均效应至少有一个主要终点超过 H-sham 绝对差值的 95 分位。

若仅部分满足，报告为混合/不确定性结果；若方向不稳定或效应落在 sham 波动内，则不主张检测有效。不得用单个 `p < 0.05` 替代效应方向和自然波动对照。

## 排除与失败处理

- `status=ok` 且存在完整 `map_metrics` 的运行进入质量分析。
- `partial`、`failed`、超时和无完整矩阵的运行进入交付与费用分析，不作质量填补。
- runner 内部自动重试保留在 `request_retries`；不人工挑选成功重试。
- 若某个地图/组未完成，仍报告设计不平衡，不删除其他地图来制造平衡。

## 复现

```powershell
$py = "$env:USERPROFILE\.cache\codex-runtimes\codex-primary-runtime\dependencies\python\python.exe"
& $py .\experiments\known_shift_20261029\audit_existing.py
& $py .\experiments\known_shift_20261029\scorer_stress.py
& $py .\experiments\known_shift_20261029\analyze_existing_evidence.py
& $py .\experiments\known_shift_20261029\run_known_shift.py --estimate
& $py .\experiments\known_shift_20261029\run_known_shift.py --dry-run
& $py .\experiments\known_shift_20261029\run_known_shift.py --run --provider "channel-F" --model "gpt-6.1-sol" --workers 4
& $py .\experiments\known_shift_20261029\analyze_known_shift.py
```
## 执行记录（2026-10-29）

| 路径 | 尝试 | 结果 | 处理 |
|---|---:|---|---|
| channel-OR / `openai/gpt-6.1-sol` | 1 次试运行 | 第 4 轮 HTTP 402（Payment Required） | 保留于 `runs/`，不进入正式分析 |
| channel-C / `gpt-6.1-sol` / `api-tools` | 1 次试运行 | partial；13 轮后仍未完成工具交付，矩阵可恢复 | 保留于 `runs_S1/`，不进入正式分析 |
| channel-C / `gpt-6.1-sol` / `api` | 1 次试运行 | HTTP 502（上游暂不可用） | 保留于 `runs_S1_api/` |
| channel-F / `gpt-6.1-sol` / `api-tools` | 1 次试运行 | 长时间无响应，人工终止 | 保留于 `runs_S2/` |
| channel-F / `gpt-6.1-sol` / `api` | 1 次试运行 | ok，完整交付 | 作为可行性确认；正式批次继续使用该路径 |
| channel-F / `gpt-6.1-sol` / `api` | 正式批次 10 次归档尝试 | ok 4，partial 6；出现 SSL 超时、部分交付和 50 分钟超时 | 保留于 `runs_S2_api/`；未形成任何 H–L 配对 |
| channel-A / `gpt-6.1-sol-codex` / `api` | 1 次试运行 | `stream ended before response.completed` | 保留于 `runs_channel-A_api/` |

正式 36 次平衡实验因服务端不稳定而未完成。按预注册停止规则，不重跑或挑选已有成功运行来制造平衡，也不对不完整样本作确认性方向检验。