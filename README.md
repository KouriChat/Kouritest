# Kouritest — 基于海陆栅格重建的大语言模型行为偏移评测

本仓库是论文《基于海陆栅格重建的大语言模型行为偏移评测方法研究》的测试脚本与公开数据。

仓库地址：<[Kouritest](https://github.com/KouriChat/Kouritest)>

论文用离线海陆栅格重建测量同一声明模型在不同服务条件下的输出偏移。模型写脚本生成二值海陆图，评价器读取交付文件并计算空间指标。整套流程在个人计算机上可以跑完。

本仓库只包含实验脚本与论文需要公开的数据。运行目录、渲染中间产物、稿件草稿和本地绝对路径都不在发布集中。渠道名已统一替换为代码（`OR`、`A`–`G`、`S1`/`S2`），真实渠道名、端点地址、密钥与本地路径均已移除。

## 0. 论文表图与仓库文件对照

论文共 11 幅图、11 张表。下表给出各自的源文件。

### 图

| 论文产出 | 图源文件 |
|---|---|
| 图 1 六幅参照矩阵 | `paper_framework/figs/fig2_tasks.png` |
| 图 2 准确率与结构指标 | `paper_framework/figs/fig4_metric_complementarity.png` |
| 图 3 受控输出扰动 | `paper_framework/figs/fig8_controlled_corruption.png` |
| 图 4 重复测量一致性 | `paper_framework/figs/fig6_reliability.png` |
| 图 5 跨条件覆盖与三图准确率 | `paper_framework/figs/fig_provider_overview.png` |
| 图 6 跨条件退化指纹 | `paper_framework/figs/fig_provider_fingerprint.png` |
| 图 7 准确率差与分量 IoU 差 | `paper_framework/figs/fig_provider_structure.png` |
| 图 8 顺序增量 R² | `paper_framework/figs/fig_provider_variance.png` |
| 图 9 行为可分性 | `paper_framework/figs/fig_provider_separability.png` |
| 图 10 逐图配对差 | `paper_framework/figs/fig_budget_task_profiles_20261105.svg` |
| 图 11 配置比较与同配置重复 | `paper_framework/figs/fig_budget_effect_comparison_20261105.svg` |

### 表

| 论文产出 | 数据文件 |
|---|---|
| 表 1 联合判读规则 | 论文正文 |
| 表 2 成本与 36 次测量外推 | 接口账单窗口（未公开明细） |
| 表 3 服务条件覆盖 | `paper_framework/analysis/completion_rates.csv` |
| 表 4 Pearson 相关系数 | `paper_framework/analysis/metric_pearson.csv` |
| 表 5 跨条件任务配对差 | `paper_framework/analysis/provider_condition_aggregate.csv` |
| 表 6 顺序增量 R² | `paper_framework/analysis/provider_variance_decomposition.csv` |
| 表 7 预算配对比较 | `paper_framework/analysis/budget_contrast_pairs.csv` |
| 表 8 扩展测量覆盖 | `paper_framework/supplement_20261105/coverage_verified.csv` |
| 表 9 六图配对结果 | `paper_framework/supplement_20261105/effects_verified.csv` |
| 表 10 两图扩展 | `paper_framework/supplement_20261105/pairs_verified.csv` |
| 表 11 预算配对比较 | `paper_framework/analysis/budget_contrast_pairs.csv` |

`paper_framework/analysis/` 里的 CSV 都可复核，字段说明见 `data/redacted/README.md`。

## 1. 研究问题

论文回答三个问题。

1. 边界 F1、分量 IoU 与近陆窄水保留率，如何响应边界平移、岛屿删除和窄水填塞？
2. 同一声明模型在不同服务条件下，质量差异如何分布到地图与指标？
3. 请求级推理配置的变化，能否与同配置重复区分开？

## 2. 探针设计

### 2.1 任务与参照

探针是六幅离线地图。每幅图规定经纬范围、网格尺寸、二值约定、图像方向和交付文件。

| 任务 | 地图 | 步长 | 网格 | 格点数 |
|---|---|---|---|---|
| `global` | 全球 | 2° | 90 × 180 | 16,200 |
| `csc` | 南海 | 0.2° | 111 × 141 | 15,651 |
| `dzh` | 地中海 | 0.2° | 71 × 231 | 16,401 |
| `seto` | 濑户内海 | 2/63° | 64 × 253 | 16,192 |
| `arctic01` | 北极航道 | 0.1° | 61 × 241 | 14,701 |
| `hecate` | 赫卡特海峡 | 2/126° | 127 × 127 | 16,129 |

参照矩阵在 `quant/refs/`，由 Natural Earth 1:10m 陆地与湖泊数据栅格化得到。

模型只接收任务说明，不读取参照矩阵。评分只读提交文件与参照矩阵。生成与评分分开，同一份提交换一台机器也能复算。

### 2.2 五个结果面

评测不只统计错格，还记录错在哪里。五个结果面各管一件事。

- **总体分类**：格点判断是否正确，类别比例是否失衡。
- **边界位置**：预测海岸线是否落在参照海岸线附近。
- **陆地分量**：大陆、岛屿和碎片陆地是否保留为对应的连通分量。
- **近陆水体**：近岸窄水是否仍被判为水。
- **交付状态**：文件是否完整、尺寸是否正确、是否二值、方向是否正确、CSV 与 PNG 是否一致。

前四项在完整交付上计算。交付状态单独统计。

### 2.3 诊断向量与方向指标

六项质量指标组成诊断向量。

| 记号 | 指标 | 回答的问题 |
|---|---|---|
| Q_cell | 逐格准确率 | 总体错格有多少 |
| Q_HSS | Heidke 技巧评分 | 扣除类别先验后还剩多少分类技巧 |
| Q_edge | 边界 F1@1 | 海岸线位置与形状是否对齐 |
| Q_comp_med | 分量 IoU 中位数 | 中小陆地单元是否保留 |
| Q_comp_aw | 分量 IoU 面积加权 | 大型陆块是否保留 |
| Q_nearland | 近陆窄水保留率 | 窄水是否被填成陆地 |

三项方向指标不进入质量向量，只用来判断误差方向。

| 记号 | 指标 | 含义 |
|---|---|---|
| Q_area | 陆地面积比 | 大于 1 偏陆，小于 1 偏水 |
| Q_recall | 陆地召回率 | 参照陆地被保留的程度 |
| Q_precision | 陆地精确率 | 预测陆地的可信程度 |

交付有效性 V 作为独立运行字段记录，不参与质量向量。

### 2.4 联合判读

单项指标会给出误导性结论。近陆窄水保留率高，可能只是整体偏水。论文表 1 给出判读规则。

| 指标组合 | 误差方向 | 解释 |
|---|---|---|
| Q_nearland 高，Q_recall 低，Q_area < 1 | 整体偏水 | 近岸水域保留多，但参照陆地被漏画 |
| Q_nearland 低，Q_recall 高，Q_area > 1 | 整体偏陆 | 陆地覆盖充分，窄水被填塞 |
| Q_nearland 高，Q_recall 高，Q_precision 高，Q_area ≈ 1 | 水陆协调 | 近岸水域与陆地结构一致 |
| Q_cell 高，Q_comp_med 低 | 小尺度结构缺失 | 总体覆盖好，中小陆地单元不足 |
| Q_cell 高，Q_edge 低 | 边界偏移 | 内部格点对，海岸线有系统误差 |

判读时先看方向指标，再看结构指标。四项质量指标不能互相替代。

## 3. 指标口径

### 3.1 基础分与准确率

评分器沿用历史口径：

```text
得分 = 该图格点数 − 与参照矩阵不一致的格子数
```

逐格准确率等于得分除以格点数。区域图另给归一化到 16,200 的分数，方便与全球图并排。

### 3.2 逐格准确率与 HSS

设混淆矩阵的四个计数为 TP、TN、FP、FN。

```text
Q_cell = (TP + TN) / (TP + TN + FP + FN)
Q_HSS  = (Q_cell - Q_exp) / (1 - Q_exp)
```

Q_exp 由类别边际计算。HSS 校正类别先验，避免多数类预测拿到虚高分。

### 3.3 边界 F1

对矩阵 X 取四邻接边界集合 B(X)。容差 d 下的边界精确率和召回率分别按距离集合计算，边界 F1 为二者的调和平均。主报告取 d = 1，敏感性分析取 d = 0 与 d = 2。容差以格点为单位，跨图比较时只作相对诊断量。

### 3.4 陆地分量 IoU

参照陆地取四连通分量，每个分量至少两格。对每个参照分量 C_j，在预测分量中找重叠最大的一个。

```text
phi_j = |C_j ∩ C_j_hat| / |C_j ∪ C_j_hat|
```

中位数对中小单元等权。面积加权均值按分量格点数加权，偏向大陆与大型岛屿。匹配是单向的，指标描述参照分量的保留程度。

### 3.5 近陆窄水保留率

对参照水域的每个连通分量，计算格点到最近参照陆地的四邻接距离。最大距离不超过 r（论文取 r = 2）的分量并入近陆窄水集合。

```text
Q_nearland = count(P(g) = 0, g in N_r(R)) / count(N_r(R))
```

该指标只测窄水是否判对，必须与面积比、陆地召回率、陆地精确率一起看。

### 3.6 统计口径

- 模型比较先算图内配对差，再对六幅地图等权汇总；
- 区间用 20,000 次双层自助法，依次重采样任务与单元内运行；
- 方差分配按有效运行数加权，依次加入任务、模型、条件、模型×条件项；
- 重复一致性用 ICC(1)，以模型—任务单元为组。

## 4. 数据构成

### 4.1 基线校准

`OR` 条件下，四款 GPT 模型各跑六幅地图，每幅五次。共 120 份完整交付，组成 24 个模型—任务单元。这组数据用于检验指标关系、受控扰动响应、参照敏感性和协议内重复一致性。

### 4.2 服务条件

主分析合并 7 个条件，共 578 份完整交付。其中 120 份用于基线校准，458 份用于跨条件比较。

| 条件 | 记录数 | 完整 | 部分 | 未完成 | 完整率/% | 说明 |
|---|---:|---:|---:|---:|---:|---|
| OR | 294 | 120 | 4 | 170 | 40.8 | 基线，4 模型 × 6 图 |
| A | 120 | 109 | 4 | 7 | 90.8 | GPT 6.1 Sol 变体组 |
| B | 31 | 30 | 0 | 1 | 96.8 | GPT 6.1 Sol |
| C | 130 | 107 | 21 | 2 | 82.3 | 4 个 GPT 型号 |
| D | 90 | 79 | 10 | 1 | 87.8 | 3 个 GPT 型号 |
| E | 125 | 47 | 1 | 77 | 37.6 | 2 个 GPT 型号 |
| F | 113 | 86 | 23 | 4 | 76.1 | 3 个 GPT 型号 |

完整率以该条件记录的全部运行数为分母。部分交付与未完成运行不进入质量指标。七个条件合计 903 次尝试、578 份完整交付、63 份部分通过、262 次失败。

数据在 `data/redacted/runs/`，字段说明见 `data/redacted/README.md`。该目录收录全部批次的 1,073 次尝试，是上表主分析条件的超集，因此按渠道统计的尝试数会略高于本表。

### 4.3 扩展测量

扩展测量固定 GPT 6.1 Sol，取两条接入路径 S1 和 S2。每条路径分高推理组（H）、低推理组（L）和高配置重复组（H-sham）。全部走单轮 API 协议。

有效输出 88 份。六图测量 72 份，两图测量 16 份（濑户内海与赫卡特）。六图形成 24 对低—高配置与 24 对同配置重复。

### 4.4 受控扰动

在参照矩阵上施加三类已知改动：边界平移、岛屿删除、窄水填塞。共 492 个场景，用于测量各指标对同一误差的响应方向和幅度。数据在 `paper_framework/analysis/controlled_corruption*.csv`。

## 5. 目录结构

| 路径 | 内容 |
|---|---|
| `run_map_experiments.py` | 主实验脚本：读取冻结提示词，逐个供应商/模型/任务/组别发起隔离会话，回收脚本与图像，按 1 错格 = 1 分打分并归档 |
| `run_experiments.py` / `run_experiments.ps1` | 入口包装，参数归一化后调用主脚本 |
| `experiment_config/` | 任务定义 `tasks.json`、极区任务 `arctic_tasks.json`、供应商别名 `providers.json`、冻结提示词 `prompts/*.txt` |
| `quant/` | 单图量化：参照矩阵 `refs/`、模型提交图 `submissions/`、指标代码与输出 `outputs/` |
| `paper_framework/analysis/` | 论文正文统计所用的汇总表（CSV/JSON） |
| `paper_framework/theory_*.csv` | 理论分析的运行级与聚合级指标 |
| `paper_framework/figs/` | 论文主图（PNG 与 2 个 SVG） |
| `paper_framework/supplement_20261105/` | 补充材料的核验脚本与核验结果 |
| `data/redacted/` | 全部实验尝试的脱敏归档（1,073 次运行） |
| `data/supplement_20261105/runs/` | 补充材料逐次运行的代表性产物 |
| `tests/` | 单元测试 |
| `webapp/` | 网页版评测器（同题目、同参照矩阵、同评分口径） |

## 6. 环境与运行

### 6.1 安装

```powershell
python -m pip install -r requirements.txt
```

`requirements.txt` 覆盖量化脚本（numpy / Pillow）与测试（openpyxl）所需依赖。

### 6.2 跑测试与量化

```powershell
# 单元测试
python -m unittest discover -s tests -v

# 单图量化诊断
python quant/quantify.py
python quant/make_report.py
```

输入只有 `quant/refs/*.csv`（参照矩阵）与 `quant/submissions/*.png`（模型提交图），两者都在仓库内并随附 SHA-256（`quant/outputs/hashes.json`）。

### 6.3 采集模型输出

```powershell
# 查看任务清单与供应商
python run_experiments.py --list-tasks
python run_experiments.py --list-providers

# 只打印计划，不调用模型
python run_experiments.py --dry-run --providers current --tasks all --groups 1

# 正式采集：六图，每组五次
python run_experiments.py --providers "Alias OR" --tasks all --groups 5
```

`experiment_config/providers.json` 里的别名是占位示例，需要按本地注册表自行填写。归档写入 `experiments/`。

### 6.4 主要参数

| 参数 | 说明 |
|---|---|
| `--providers` | 供应商名称或别名，逗号分隔；`all` 为全部，`current` 为当前项 |
| `--models` | 模型 slug；`all` 读取供应商 catalog 的全部模型 |
| `--tasks` | `global,csc,dzh,seto,arctic01,hecate` 的组合，或 `all` |
| `--groups` | 每个模型 × 每个任务的独立组数 |
| `--backend` | `api-tools`（默认）、`api`、`codex` |
| `--api-transport` | `auto`、`proxy`、`direct` |
| `--reasoning-effort` | `low`、`medium`、`high` |
| `--resume` | 跳过已归档且状态为 ok 的同一组 |
| `--score-only` | 只对已有归档重新评分 |
| `--report-only` | 只根据现有 meta 重建汇总文件 |

完整参数用 `python run_experiments.py --help` 查看。

## 7. 网页版评测器

`webapp/` 是主脚本的网页对应版本，题目、参照矩阵与评分口径一致，供需要图形界面时使用。

- 直接双击 `webapp/index.html` 即可运行，所有请求与评分都在浏览器 JS 内完成，不需要后端；
- 供应商接口不允许浏览器跨域时，可用附带的中继：`pip install flask && python webapp/server.py`；
- 页面可填接口地址、API Key 与模型 ID（不外传、不写入仓库），可选 Chat Completions / Responses API、直连 / 代理、流式 / 非流式；
- 六张地图可多选、可设每组重复次数；结果给出得分、准确率、HSS、边界 F1@1、虚报/漏报格数，并可导出矩阵 CSV、误差图与原始回复 JSON；
- 另含手动复核入口：粘贴矩阵文本或上传 PNG（含 8× 最近邻放大）立即按同一口径评分。

自测：

```powershell
node webapp/tools/build_assets.js    # 由 prompt/refs 重新生成内嵌资源
node webapp/tests/score.test.js      # 评分核心单元测试
python webapp/tests/server.test.py   # 可选中继的单元测试
```

## 8. 完整脱敏实验数据

`data/redacted/` 收录本项目全部 1,073 次实验尝试的脱敏归档。

| 路径 | 内容 |
|---|---|
| `data/redacted/runs/<run_id>/` | 每次尝试一个目录：提交图、8 倍图、矩阵 CSV、误差图、生成脚本、`meta.json` |
| `data/redacted/tables/runs.csv` | 1,073 行的运行级总表 |
| `data/redacted/tables/by_channel_task.csv` | 按渠道与地图的均值 |
| `data/redacted/tables/status_counts.csv` | 按渠道与状态的尝试数 |
| `data/redacted/summaries/` | 各批次汇总 |
| `data/redacted/analysis/batch-15/` | 已知偏移探针：协议、报告、审计、评分器压力测试 |
| `data/redacted/analysis/batch-16/` | 预算对比实验：协议、报告、表 |
| `data/redacted/analysis/batch-17/` | 2026-11-05 批次独立复核 |
| `data/redacted/manifest.json` | 机器可读索引 |
| `data/redacted/channels.json` | 渠道索引（运行数、模型、任务、批次） |
| `data/redacted/checksums.sha256` | 全部文件的 SHA-256 |

字段、目录与复现说明见 `data/redacted/README.md`。原始请求体、供应商响应、工具记录与 `_runs/` 执行目录有意排除，它们可能含端点、凭据、客户端 IP 与路由信息。

## 9. 补充材料核验

```powershell
python paper_framework/supplement_20261105/verify_supplement.py --source data/supplement_20261105
```

完整核验需要补充材料归档的原始目录，本仓库提供的是代表性子集。`verification.json` 记录了两者的对应关系，`REPORT.md` 给出核验结果。

`data/supplement_20261105/runs/<S>/<arm>/<task>/<run_id>/` 每个目录含四类文件：

- `candidate.png` / `candidate_8x.png` —— 提交图及其 8 倍放大版；
- `candidate.csv` —— 该提交的 0/1 栅格；
- `<run_id>.py` —— 模型为生成该图所写的脚本。

其中 `<S>` 取 S1 或 S2，`<arm>` 取 H、L 或 H-sham，`<task>` 取六个任务代号。

## 10. 数据来源与脱敏

- **参照矩阵**由 Natural Earth 1:10m 陆海数据栅格化得到（0/1）。
- **渠道命名已脱敏**：正文条件代号 `OR` 与 `A`–`G` 原样保留，补充材料的两个批次以 `S1`、`S2` 表示。真实渠道名、端点地址、密钥与本地绝对路径都不出现在发布集中。`experiment_config/providers.json` 与本脚本内的别名表为占位示例。
- **运行产物**保留原始 run ID、时间戳与指标，便于按 ID 复核。
- 模型显示名做了归一（例如 `GPT 6.1 Sol OR` 与 `GPT 6.1 Sol AH` 都记成 `GPT 6.1 Sol`），`channel` 列保留渠道对比信息。

## 11. 引用与可用性

论文使用的测试脚本、参照矩阵、模型输出归档与分析表都在本仓库。公开地址：

<Kouritest](https://github.com/KouriChat/Kouritest)>

引用格式以论文参考文献为准。
