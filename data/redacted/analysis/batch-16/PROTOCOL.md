# 负相关实验：推理预算约束与地图质量

## 研究问题

在模型声明、地图任务、冻结提示词、工具集、输出上限、后端与评分器均保持不变的条件下，提高请求级限制强度 `d`（等价于降低推理预算）是否导致地图质量下降？

## 操作化

- 预算条件：`high`、`medium`、`low`。
- 限制强度编码：`d=0`（high）、`d=1`（medium）、`d=2`（low）。
- 唯一处理变量：Responses API 请求体中的 `reasoning.effort`。
- 不修改 `run_map_experiments.py`、`quant/quantify.py`、任务提示词、工具权限、地图尺寸、输出上限或人工修补流程。
- 质量越高越好。方向性假设为：`low - high < 0`，且质量对 `d` 的组内斜率为负。

## 设计

- 服务条件：
  - channel-C / `gpt-6.1-sol`
  - channel-A / `gpt-6.1-sol-codex`（名义 GPT 6.1 Sol 的可比变体，纳入 provider 内配对并在报告中注明）
  - channel-F / `gpt-6.1-sol`
- 地图任务：`seto`（濑户内海）与 `hecate`（赫卡特海峡）。
- 重复：每个 provider × task 有 2 个独立 group。
- 计划运行数：3 × 2 × 3 × 2 = 36。
- 随机化：每个 group 随机排列三个预算水平；第二个 group 使用反向顺序，以平衡时间漂移。
- 后端：`api-tools`；传输：`direct`；最多输出 token：12000；每次运行超时：45 分钟。
- 每次运行使用冻结提示词、同一评分器与同一工具循环。各预算使用隔离归档根，避免同名归档相互覆盖。

## 主要结局

1. 准确率（`accuracy_pct / 100`）
2. HSS
3. 边界 F1@1（`boundary_f1_d1`）
4. 近陆窄水保留率（`narrow_water_keep`）

次要结局：岛屿 IoU 中位数、岛屿 IoU 面积加权、岛屿 IoU≥0.5 比例、分量召回率。

## 统计方法

- 仅将 `status=ok` 且有完整地图指标的运行纳入质量分析；失败、跳过和缺失均报告，不做成功样本填补。
- 配对键：`provider × task × group`。主分析使用三个预算均有效的完整三元组。
- 主对比：`low - high`。同时报告 `medium - high` 作为剂量检查。
- 效应量：均值配对差、配对 Cohen's dz、下降配对数/完整配对数。
- 95% 置信区间：以完整三元组为独立单位的 block bootstrap，20000 次，固定随机种子 20261028。
- 方向检验：单侧配对符号翻转检验（H1: 差值 < 0）。n≤18 时精确枚举，否则使用 100000 次固定种子蒙特卡洛。
- 多重比较：对四个主要结局的 `low - high` 单侧 p 值做 Holm 校正。
- 剂量趋势：每个完整三元组计算质量对 `d∈{0,1,2}` 的最小二乘斜率；报告平均斜率、bootstrap CI、负斜率比例及单侧符号翻转检验。
- 本实验检验的是请求级预算参数的因果方向性，不等同于识别供应商内部实际使用的推理预算。

## 复现

```powershell
$py = "$env:USERPROFILE\.cache\codex-runtimes\codex-primary-runtime\dependencies\python\python.exe"
& $py .\experiments\negative_correlation_20261028\run_negative_correlation.py
& $py .\experiments\negative_correlation_20261028\analyze_negative_correlation.py
```

只计划、不调用模型：

```powershell
& $py .\experiments\negative_correlation_20261028\run_negative_correlation.py --dry-run
```

脚本始终把原始响应、请求记录、归档和 runner 日志写入本目录，不触碰主实验目录。
