# Kouritest — 海陆地图重建实验的测试脚本与公开数据

本仓库是论文配套的最小发布集：**只包含实验脚本本身，以及论文需要公开的数据**。
不包含运行目录、渲染中间产物、稿件草稿或任何本地绝对路径。

## 目录结构

| 路径 | 内容 |
|---|---|
| `run_map_experiments.py` | 主实验脚本（唯一的实现文件）：读取冻结提示词、逐个供应商/模型/任务/组别发起隔离会话、回收脚本与图像、按 1 错格 = 1 分打分、按历史命名归档 |
| `run_experiments.py` / `run_experiments.ps1` | 入口包装（参数归一化后调用主脚本） |
| `experiment_config/` | 任务定义 `tasks.json`、极区任务 `arctic_tasks.json`、供应商别名 `providers.json`、冻结提示词 `prompts/*.txt` |
| `tests/` | 单元测试，`python -m unittest discover -s tests -v` |
| `quant/` | 单图量化诊断：参照矩阵 `refs/`、模型提交图 `submissions/`、指标代码与输出 `outputs/` |
| `paper_framework/analysis/` | 论文正文统计所用的汇总表（CSV/JSON） |
| `paper_framework/theory_*.csv` | 理论分析用的运行级与聚合级指标 |
| `paper_framework/figs/` | 论文主图（PNG 与 2 个 SVG） |
| `paper_framework/refs.bib` | 参考文献 |
| `paper_framework/supplement_20261105/` | 补充材料的核验脚本与核验结果 |
| `data/supplement_20261105/runs/` | 补充材料中逐次运行的代表性产物（见下） |

## 环境安装

```powershell
python -m pip install -r requirements.txt
```

`requirements.txt` 覆盖量化脚本（numpy / Pillow）与测试（openpyxl）所需依赖。

## 跑测试

```powershell
python -m unittest discover -s tests -v
```

## 跑量化诊断

```powershell
python quant/quantify.py      # 写出 quant/outputs/metrics.{csv,json} 等
python quant/make_report.py   # 写出 quant/outputs/report.md
```

输入只有 `quant/refs/*.csv`（参照矩阵）与 `quant/submissions/*.png`（模型提交图），
两者都在仓库内并随附 SHA-256（`quant/outputs/hashes.json`）。

## 查看任务清单

```powershell
python run_experiments.py --list-tasks
```

## 数据来源与匿名化

- **参照矩阵**由 Natural Earth 1:10m 陆海数据栅格化得到（0/1），六张地图的规格：
  `global` 90×180（2°）、`csc` 111×141（0.2°）、`dzh` 71×231（0.2°）、
  `seto` 64×253（2/63°）、`arctic01` 61×241（0.1°）、`hecate` 127×127（2/126°）。
- **供应商命名已匿名化**：正文中的条件代号 `A`–`F` 与 `OR` 原样保留，
  补充材料的两个批次以 `S1` / `S2` 表示。真实渠道名、端点地址、
  密钥与本地绝对路径均不出现在发布集中；`experiment_config/providers.json`
  与本脚本内的别名表为占位示例，需要按本地注册表自行填写。
- **运行产物**保留原始 run ID、时间戳与指标，以便按 ID 复核。

## 补充材料运行产物的目录约定

`data/supplement_20261105/runs/<S>/<arm>/<task>/<run_id>/` 每个目录含四类文件：

- `candidate.png` / `candidate_8x.png` —— 模型提交图及其 8 倍放大版
- `candidate.csv` —— 该提交的 0/1 栅格
- `<run_id>.py` —— 模型为生成该图所写的脚本

其中 `<S>` ∈ {`S1`, `S2`}，`<arm>` ∈ {`H`, `L`, `H-sham`}，
`<task>` ∈ {`global`, `csc`, `dzh`, `seto`, `arctic01`, `hecate`}。

## 复现补充材料核验

```powershell
python paper_framework/supplement_20261105/verify_supplement.py --source data/supplement_20261105
```

完整核验需要补充材料归档的原始目录；本仓库提供的是其中的代表性子集，
`verification.json` 记录了两者的对应关系。
