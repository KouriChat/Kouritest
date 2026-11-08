# 负相关实验收束报告

生成时间：2026-10-28（Asia/Shanghai）

## 1. 结论先行

预算实验去重后为 **20 次归档尝试、15 次有效运行、1 次 failed、4 次 partial**，形成 **1 个完整 high–medium–low 单元**和 **5 个组内 low–high 配对**。5 个配对的 low−high 平均差为：准确率 −8.36、HSS −16.98、边界 F1 −4.94、按格点数加权的分量 IoU −8.94 个百分点；四项均有 4/5 个配对下降。近陆水域保留率平均差为 −4.01 个百分点，仅 3/5 个配对下降，95% 配对自助区间跨零。

该结果支持“低请求级推理预算与总体覆盖及部分结构指标的下降方向一致”，并显示指标与任务异质性；它不支持跨任务、跨指标的普遍负相关，也不支持稳定线性剂量关系。

## 2. 统计口径

- 主配对键：`provider × task × group`。
- 主对比：low − high。
- 准确率、HSS、边界 F1 与近陆水域保留率组成四指标检验族；面积加权分量 IoU 作为附加结构指标。
- 5 个配对下，准确率、HSS、边界 F1 的单侧配对符号翻转检验原始 p=0.0625；Holm 校正后 p=0.25。近陆水域保留率原始 p=0.4375。
- 区间为以配对单元为重抽样单位的 20,000 次 percentile bootstrap。
- 唯一完整三水平单元为 channel-C / hecate / g1；medium 的准确率高于 high，low 低于两者，呈非单调响应。
- OR 的 5 high / 3 low 为跨批次非配对试点，不并入 5 个组内配对。

## 3. 关键结果

| 条件/任务 | 设计 | n | ΔAcc | ΔHSS | ΔF1@1 | Δ近陆水 | ΔIoU格点加权 |
|---|---|---:|---:|---:|---:|---:|---:|
| OR / 濑户内海 | 非配对 | 5 vs 3 | −2.07 | −4.14 | −9.40 | −47.88 | +0.38 |
| channel-C / 濑户内海 | 配对 | 2 | −8.37 | −17.40 | −7.27 | −43.18 | −5.12 |
| channel-C / 赫卡特 | 配对 | 1 | −10.27 | −18.67 | −8.61 | +46.83 | −11.16 |
| channel-F / 濑户内海 | 配对 | 1 | +2.61 | +4.70 | +1.99 | −9.09 | −13.72 |
| channel-F / 赫卡特 | 配对 | 1 | −17.38 | −36.12 | −3.57 | +28.57 | −9.56 |
| 5 个配对均值 | 配对 | 5 | −8.36 | −16.98 | −4.94 | −4.01 | −8.94 |

## 4. 科学解释

1. **总体与边界指标方向较一致。** channel-C 的 3 个配对在准确率、HSS 与边界 F1 上全部为负；channel-F的 2 个配对在赫卡特为负，在濑户内海为正。
2. **准确率改善可与结构损失并存。** channel-F / 濑户内海配对准确率提高 2.61 个百分点，按格点数加权的分量 IoU 下降 13.72 个百分点。
3. **近陆水域指标方向不稳定。** 两个赫卡特配对的近陆水域保留率上升，但准确率与边界 F1 下降；该指标应与陆地召回率、精确率和格点比联合判读。
4. **三水平响应非单调。** channel-C / 赫卡特 / g1 的 high、medium、low 准确率分别为 73.84%、76.84%、63.57%，medium 高于 high、low 低于两者。
5. **成功交付是质量分析的样本条件。** 低预算组包含更多 partial；质量效应与交付失效应作为两个结果面分别报告。

## 5. 复现

```powershell
$py = "$env:USERPROFILE\.cache\codex-runtimes\codex-primary-runtime\dependencies\python\python.exe"
& $py .\experiments\negative_correlation_20261028\analyze_negative_correlation.py --bootstrap 20000
& $py .\experiments\negative_correlation_20261028\describe_completed.py
```

## 6. 文件索引

- `PROTOCOL.md`：实验设计与分析口径
- `AMENDMENT_20261028.md`：channel-A 排除与运行中修订
- `raw_runs.csv`：20 条归档尝试及状态
- `valid_runs.csv`：15 条有效运行
- `complete_triplets.csv`：完整三水平单元
- `paired_effects.csv`：组内配对效应与区间
- `primary_tests.csv`：四指标检验族与 Holm 校正
- `trend_effects.csv`：完整三水平单元的探索性斜率
- `heterogeneity.csv`：provider 与 provider×task 的配对异质性
- `cell_summary.csv`：单元均值与有效运行数
- `direction_descriptors.csv`：high/medium/low 方向描述
- `analysis_summary.json`、`experiment_audit.json`：机器可读汇总与审计
