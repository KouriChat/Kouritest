# -*- coding: utf-8 -*-
"""从 outputs/metrics.json 生成 outputs/report.md。"""
import json
from pathlib import Path

Q = Path(__file__).resolve().parent
d = json.loads((Q / "outputs" / "metrics.json").read_text(encoding="utf-8"))
rows = d["rows"]
refs = d["reference_meta"]
MAPORDER = ["global", "csc", "dzh", "seto", "arctic01", "hecate"]
MAPNAME = {
    "global": "全球",
    "csc": "南海",
    "dzh": "地中海",
    "seto": "濑户内海",
    "arctic01": "北极航道0.1°",
    "hecate": "赫卡特海峡",
}

lines = []
A = lines.append

A("# 海陆栅格重建：单图空间诊断报告")
A("")
A("生成方式：`python quantify.py`（参见 README.md）。全部指标是参照矩阵与提交图的确定性函数；")
A("唯一随机源是固定种子的分块自助法。报告只列出独立诊断指标，不生成跨指标综合分。")
A("")

A("## 1. 参照矩阵")
A("")
A("| 地图 | 尺寸 | 陆地点 | 陆地占比 | 多数类基线 | 边界点占比 | 连通分量 | 格距(km) |")
A("|---|---|---|---|---|---|---|---|")
for k in MAPORDER:
    m = refs[k]
    A(
        f"| {MAPNAME[k]} | {m['shape'][0]}×{m['shape'][1]} | {m['land']} | "
        f"{m['land_frac']*100:.1f}% | {m['majority_baseline']*100:.1f}% | "
        f"{m['boundary_frac']*100:.1f}% | {m['n_components']} | {m['km_per_cell']:.2f} |"
    )
A("")

A("## 2. 主结果表")
A("")
A("| 模型 | 地图 | acc% | acc_w% | HSS | 边界F1@1 | 窄水道保留 | 岛屿IoU中位 | 岛屿IoU面积加权 | IoU≥0.5比例 | acc 95%CI(分块自助) |")
A("|---|---|---|---|---|---|---|---|---|---|---|")
for k in MAPORDER:
    for r in sorted([x for x in rows if x["map"] == k], key=lambda x: -x["accuracy"]):
        ci = f"[{r['accuracy_ci_lo']*100:.2f}, {r['accuracy_ci_hi']*100:.2f}]"
        nw = r["narrow_water_keep"]
        nws = "n/a" if nw != nw else f"{nw*100:.0f}%"
        A(
            f"| {r['model']} | {MAPNAME[k]} | {r['accuracy']*100:.2f} | "
            f"{r['accuracy_area_weighted']*100:.2f} | {r['hss']:.3f} | "
            f"{r['boundary_f1_d1']:.3f} | {nws} | {r['island_iou_median']:.2f} | "
            f"{r['island_iou_area_weighted']:.2f} | {r['island_iou_ge50']*100:.0f}% | {ci} |"
        )
A("")

A("## 3. 空间诊断量")
A("")
A("| 模型 | 地图 | 质心偏移(km) | 面积比 | 岛屿召回 | 岛屿IoU中位 | 岛屿IoU面积加权 | IoU≥0.5比例 | 窄水道保留 | 边界中位距离(km) |")
A("|---|---|---|---|---|---|---|---|---|---|")
for k in MAPORDER:
    for r in sorted([x for x in rows if x["map"] == k], key=lambda x: -x["accuracy"]):
        nw = r["narrow_water_keep"]
        nws = "n/a" if nw != nw else f"{nw*100:.0f}%"
        A(
            f"| {r['model']} | {MAPNAME[k]} | {r['centroid_offset_km']:.0f} | "
            f"{r['area_ratio']:.3f} | {r['component_recall']*100:.0f}% | "
            f"{r['island_iou_median']:.2f} | {r['island_iou_area_weighted']:.2f} | "
            f"{r['island_iou_ge50']*100:.0f}% | {nws} | {r['boundary_mean_km']:.0f} |"
        )
A("")

A("## 4. 重复提交（同一文件）")
A("")
dups = [r for r in rows if r["duplicate_of"]]
if dups:
    for r in dups:
        A(f"- `{r['file']}` 与 `{r['duplicate_of']}` 的像素完全一致（SHA-256 相同）。")
else:
    A("（无）")
A("")

sens = d.get("sensitivity", [])
if sens:
    A("## 5. 参照敏感性")
    A("")
    A("| 地图 | 参照变体 | 模型 | 主参照 acc% | 变体 acc% | Δ(pp) | 改变的格数 |")
    A("|---|---|---|---|---|---|---|")
    for s in sens:
        A(
            f"| {MAPNAME.get(s['map'], s['map'])} | {s['variant']} | {s['model']} | "
            f"{s['accuracy_primary']*100:.2f} | {s['accuracy_variant']*100:.2f} | "
            f"{s['delta_pp']:+.2f} | {s['n_changed_cells']} |"
        )
    A("")

(Q / "outputs" / "report.md").write_text(chr(10).join(lines), encoding="utf-8")
print("wrote report.md", len(lines), "lines")
