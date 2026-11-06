#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
quantify.py -- 单图空间诊断指标

输入
  refs/<map>.csv                参照矩阵 R（Natural Earth 栅格化，0/1）
  submissions/<name>.png        模型提交图 P

输出
  outputs/metrics.csv           每 (模型, 地图) 一行，全部诊断指标
  outputs/metrics.json          同上 + 完整 CONFIG（可复现所需的一切参数）
  outputs/hashes.json           全部输入文件的 SHA-256
  outputs/report.md             人读报告
  outputs/decode_log.json       每份提交的解码记录（模式/位深/色阶/阈值）

说明
  本脚本不生成跨指标综合分。准确率、HSS、边界 F1、近陆窄水体保留率
  与岛屿 IoU 均作为独立诊断量报告。

复现性约定
  1. 指标全部是 (R, P) 的确定性函数；唯一随机源是固定种子的分块自助法。
  2. PNG 一律先 convert('L') 再按 >127.5 二值化。
     这一步是必须的：1-bit 灰度 PNG 在 PIL 里是 mode '1'，数组是 bool，
     直接比较阈值会静默得到全 0 矩阵。本脚本记录每份文件的真实模式与色阶。
  3. 参数全部预先写死在 CONFIG；不得在看到结果后调整。
  4. 逐格 i.i.d. 不成立，置信区间用分块自助法（block bootstrap）。
"""
from __future__ import annotations

import csv
import hashlib
import json
import math
import sys
from collections import deque
from pathlib import Path

import numpy as np
from PIL import Image

ROOT = Path(__file__).resolve().parent
REF_DIR = ROOT / "refs"
SUB_DIR = ROOT / "submissions"
OUT_DIR = ROOT / "outputs"

CONFIG = {
    # ---- 冻结的诊断参数
    "binary_threshold": 127.5,
    "boundary_tolerances": [0, 1, 2],
    "component_min_cells": 2,
    "component_iou_match": 0.30,
    "narrow_water_radius": 2,
    "bootstrap_B": 1000,
    "bootstrap_block": 5,
    "bootstrap_seed": 20261016,
}

MAPS = {
    "global": {
        "ref": "template_land_water_90x180.csv",
        "lat_top": 89.0, "lat_bot": -89.0, "lon_left": -179.0, "lon_right": 179.0,
        "label": "global 2deg",
        "variants": {
            "caspian_as_water": "template_variant_caspian_water_90x180.csv",
            "shelf_as_land": "template_variant_shelf_as_land_90x180.csv",
        },
    },
    "csc": {
        "ref": "template_region_south_china_sea_111x141.csv",
        "lat_top": 23.0, "lat_bot": 1.0, "lon_left": 99.0, "lon_right": 127.0,
        "label": "south_china_sea 0.2deg",
    },
    "dzh": {
        "ref": "template_region_mediterranean_71x231.csv",
        "lat_top": 45.0, "lat_bot": 31.0, "lon_left": -9.0, "lon_right": 37.0,
        "label": "mediterranean 0.2deg",
    },
    "seto": {
        "ref": "template_region_seto_inland_sea_64x253.csv",
        "lat_top": 35.0, "lat_bot": 33.0, "lon_left": 129.0, "lon_right": 137.0,
        "label": "seto_inland_sea 2/63deg",
        "uncertainty": "template_uncertainty_seto_inland_sea_64x253.png",
    },
    "arctic01": {
        "ref": "template_region_arctic_nwp_61x241.csv",
        "lat_top": 79.0, "lat_bot": 73.0, "lon_left": -105.0, "lon_right": -81.0,
        "label": "arctic_northwest_passage_core 0.1deg",
    },
    "hecate": {
        "ref": "template_region_hecate_strait_127x127.csv",
        "lat_top": 55.0, "lat_bot": 53.0, "lon_left": -131.0, "lon_right": -129.0,
        "label": "hecate_strait 2/126deg",
    },
}


def parse_submission(path: Path):
    stem = path.stem
    if stem == "handwritten-baseline_seto":
        return "handwritten-baseline", "seto", "baseline"
    # Strip channel and group decorations before matching the task suffix.
    # Examples: m_arc01@or, m_arc01@or#2, m_csc@or#3.
    channel = "main"
    name = stem
    if "@" in name:
        name, channel = name.split("@", 1)
        channel = channel.split("#", 1)[0]
    name = name.split("#", 1)[0]
    suffix_map = {
        "_csc": "csc", "_dzh": "dzh", "_seto": "seto",
        "_arc01": "arctic01", "_hec": "hecate",
    }
    m = "global"
    for suf, mp in suffix_map.items():
        if name.endswith(suf):
            name, m = name[: -len(suf)], mp
            break
    return name, m, channel


# ---------------------------------------------------------------- 读取与解码
def read_csv_matrix(path: Path) -> np.ndarray:
    rows = []
    for line in path.read_text(encoding="utf-8").splitlines():
        line = line.strip()
        if not line:
            continue
        vals = [v for v in line.replace(";", ",").split(",") if v.strip() != ""]
        rows.append([int(float(v)) for v in vals])
    return np.array(rows, dtype=np.uint8)


def read_png_matrix(path: Path):
    """返回 (矩阵, 元数据)。必须先 convert('L')：1-bit PNG 的数组是 bool。"""
    im = Image.open(path)
    im.load()
    meta = {"mode": im.mode, "size": im.size, "info_bits": im.info.get("bits")}
    g = im.convert("L")
    arr = np.array(g)
    levels = sorted(set(np.unique(arr).tolist()))
    m = (arr.astype(np.int16) > CONFIG["binary_threshold"]).astype(np.uint8)
    meta["levels"] = levels
    meta["land_fraction"] = float(m.mean())
    return m, meta


def sha256(path: Path) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(1 << 16), b""):
            h.update(chunk)
    return h.hexdigest()


# ---------------------------------------------------------------- 基础指标
def confusion(P, R):
    tp = int(np.sum((P == 1) & (R == 1)))
    tn = int(np.sum((P == 0) & (R == 0)))
    fp = int(np.sum((P == 1) & (R == 0)))
    fn = int(np.sum((P == 0) & (R == 1)))
    return tp, tn, fp, fn


def accuracy(P, R):
    return float(np.mean(P == R))


def balanced_accuracy(P, R):
    tp, tn, fp, fn = confusion(P, R)
    tpr = tp / (tp + fn) if (tp + fn) else 0.0
    tnr = tn / (tn + fp) if (tn + fp) else 0.0
    return 0.5 * (tpr + tnr)


def hss(P, R):
    tp, tn, fp, fn = confusion(P, R)
    N = tp + tn + fp + fn
    exp = ((tp + fn) * (tp + fp) + (tn + fp) * (tn + fn)) / (N * N)
    return (accuracy(P, R) - exp) / (1.0 - exp) if exp < 1.0 else 0.0


def majority_baseline(R):
    p = float(np.mean(R == 1))
    return max(p, 1.0 - p)


# ---------------------------------------------------------------- 粗网格
def downsample_majority(M, k):
    H, W = M.shape
    Hc, Wc = H // k, W // k
    if Hc == 0 or Wc == 0:
        return M.copy()
    blk = M[: Hc * k, : Wc * k].reshape(Hc, k, Wc, k).astype(float)
    return (blk.mean(axis=(1, 3)) >= 0.5).astype(np.uint8)


# ---------------------------------------------------------------- 平移 / 缩放
def label_components(M, min_cells=1):
    H, W = M.shape
    lab = np.zeros((H, W), dtype=np.int32)
    comps, cur = [], 0
    for i0 in range(H):
        for j0 in range(W):
            if M[i0, j0] == 1 and lab[i0, j0] == 0:
                cur += 1
                q = deque([(i0, j0)])
                lab[i0, j0] = cur
                cells = []
                while q:
                    i, j = q.popleft()
                    cells.append((i, j))
                    for di, dj in ((1, 0), (-1, 0), (0, 1), (0, -1)):
                        a, b = i + di, j + dj
                        if 0 <= a < H and 0 <= b < W and M[a, b] == 1 and lab[a, b] == 0:
                            lab[a, b] = cur
                            q.append((a, b))
                if len(cells) >= min_cells:
                    comps.append((len(cells),
                                  np.array([c[0] for c in cells]),
                                  np.array([c[1] for c in cells])))
                else:
                    lab[lab == cur] = 0
                    cur -= 1
    return lab, comps


def component_stats(P, R, min_cells, iou_match):
    labP, compsP = label_components(P, min_cells)
    labR, compsR = label_components(R, min_cells)
    sizesP = {}
    for v in np.unique(labP):
        if v > 0:
            sizesP[int(v)] = int(np.sum(labP == v))
    matched, ious = 0, []
    for size, ci, cj in compsR:
        pl = labP[ci, cj]
        pl = pl[pl > 0]
        best = 0.0
        if pl.size:
            vals, counts = np.unique(pl, return_counts=True)
            for v, inter in zip(vals, counts):
                union = size + sizesP.get(int(v), 0) - int(inter)
                iou = float(inter) / union if union else 0.0
                if iou > best:
                    best = iou
        ious.append(best)
        if best >= iou_match:
            matched += 1
    return {
        "n_ref_components": len(compsR),
        "n_pred_components": len(compsP),
        "component_recall": matched / len(compsR) if compsR else 0.0,
        "island_iou_median": float(np.median(ious)) if ious else 0.0,
        "island_iou_area_weighted": (
            float(
                sum(iou * int(c[0]) for iou, c in zip(ious, compsR))
                / sum(int(c[0]) for c in compsR)
            )
            if compsR else 0.0
        ),
        "island_iou_ge50": float(np.mean([1.0 if v >= 0.5 else 0.0 for v in ious])) if ious else 0.0,
    }


# ---------------------------------------------------------------- 边界几何
def boundary_set(M):
    b = np.zeros_like(M, dtype=bool)
    d = M[1:, :] != M[:-1, :]
    b[1:, :] |= d
    b[:-1, :] |= d
    d = M[:, 1:] != M[:, :-1]
    b[:, 1:] |= d
    b[:, :-1] |= d
    return b


def dilate(b, r):
    out = b.copy()
    for _ in range(r):
        nxt = out.copy()
        nxt[1:, :] |= out[:-1, :]
        nxt[:-1, :] |= out[1:, :]
        nxt[:, 1:] |= out[:, :-1]
        nxt[:, :-1] |= out[:, 1:]
        out = nxt
    return out


def boundary_f1(P, R, d):
    bp, br = boundary_set(P), boundary_set(R)
    if bp.sum() == 0 or br.sum() == 0:
        return 0.0, 0.0, 0.0
    prec = float(np.sum(bp & dilate(br, d))) / float(bp.sum())
    rec = float(np.sum(br & dilate(bp, d))) / float(br.sum())
    f1 = 2 * prec * rec / (prec + rec) if (prec + rec) else 0.0
    return prec, rec, f1


def chamfer_distances(P, R, km_per_cell):
    def dist_to(target):
        INF = 10 ** 6
        D = np.full(target.shape, INF, dtype=np.int32)
        q = deque()
        H, W = target.shape
        for i, j in np.argwhere(target):
            D[i, j] = 0
            q.append((i, j))
        while q:
            i, j = q.popleft()
            for di in (-1, 0, 1):
                for dj in (-1, 0, 1):
                    if di == 0 and dj == 0:
                        continue
                    a, b = i + di, j + dj
                    if 0 <= a < H and 0 <= b < W:
                        step = 1 if (di == 0 or dj == 0) else 2
                        if D[a, b] > D[i, j] + step:
                            D[a, b] = D[i, j] + step
                            q.append((a, b))
        return D.astype(float) / 2.0

    bp, br = boundary_set(P), boundary_set(R)
    if bp.sum() == 0 or br.sum() == 0:
        return dict(mean_cells=float("nan"), p95_cells=float("nan"),
                    mean_km=float("nan"), p95_km=float("nan"))
    both = np.concatenate([dist_to(br)[bp], dist_to(bp)[br]])
    return dict(mean_cells=float(np.mean(both)), p95_cells=float(np.percentile(both, 95)),
                mean_km=float(np.mean(both) * km_per_cell),
                p95_km=float(np.percentile(both, 95) * km_per_cell))


def narrow_water_preservation(P, R, radius):
    H, W = R.shape
    D = np.full((H, W), 10 ** 6, dtype=np.int32)
    q = deque()
    for i, j in np.argwhere(R == 1):
        D[i, j] = 0
        q.append((i, j))
    while q:
        i, j = q.popleft()
        for di, dj in ((1, 0), (-1, 0), (0, 1), (0, -1)):
            a, b = i + di, j + dj
            if 0 <= a < H and 0 <= b < W and D[a, b] > D[i, j] + 1:
                D[a, b] = D[i, j] + 1
                q.append((a, b))
    water = (R == 0).astype(np.uint8)
    lab, comps = label_components(water, 1)
    narrow = np.zeros((H, W), dtype=bool)
    for size, ci, cj in comps:
        if D[ci, cj].max() <= radius:
            narrow[ci, cj] = True
    tot = int(narrow.sum())
    if tot == 0:
        return float("nan"), 0
    return float(np.sum(narrow & (P == 0))) / tot, tot


# ---------------------------------------------------------------- 自助法
def block_bootstrap_ci(P, R, metric_fn, B, block, seed):
    rng = np.random.default_rng(seed)
    H, W = R.shape
    nb_i, nb_j = math.ceil(H / block), math.ceil(W / block)
    vals = []
    for _ in range(B):
        si = rng.integers(0, nb_i, size=nb_i)
        sj = rng.integers(0, nb_j, size=nb_j)
        rows = np.concatenate([np.arange(s * block, min((s + 1) * block, H)) for s in si])
        cols = np.concatenate([np.arange(s * block, min((s + 1) * block, W)) for s in sj])
        rows, cols = rows[rows < H], cols[cols < W]
        if rows.size == 0 or cols.size == 0:
            continue
        vals.append(metric_fn(P[np.ix_(rows, cols)], R[np.ix_(rows, cols)]))
    vals = np.array(vals, dtype=float)
    vals = vals[~np.isnan(vals)]
    if vals.size == 0:
        return float("nan"), float("nan")
    return float(np.percentile(vals, 2.5)), float(np.percentile(vals, 97.5))


# ---------------------------------------------------------------- 主流程
def main():
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    refs, ref_meta = {}, {}
    for key, cfg in MAPS.items():
        R = read_csv_matrix(REF_DIR / cfg["ref"])
        refs[key] = R
        H, W = R.shape
        dlat = abs(cfg["lat_top"] - cfg["lat_bot"]) / max(H - 1, 1)
        dlon = abs(cfg["lon_right"] - cfg["lon_left"]) / max(W - 1, 1)
        lat_mid = 0.5 * (cfg["lat_top"] + cfg["lat_bot"])
        km_per_cell = 111.32 * math.sqrt(dlat ** 2 + (dlon * math.cos(math.radians(lat_mid))) ** 2)
        ref_meta[key] = dict(
            shape=[H, W],
            land=int(R.sum()),
            land_frac=float(R.mean()),
            majority_baseline=majority_baseline(R),
            boundary_frac=float(boundary_set(R).mean()),
            n_components=len(label_components(R, 2)[1]),
            km_per_cell=km_per_cell,
        )
        print(
            f"[ref] {key}: {H}x{W} land={int(R.sum())} "
            f"maj={ref_meta[key]['majority_baseline']:.4f}",
            file=sys.stderr,
        )

    hashes = {str(p.relative_to(ROOT)): sha256(p) for p in sorted(REF_DIR.glob("*"))}
    hashes.update({str(p.relative_to(ROOT)): sha256(p) for p in sorted(SUB_DIR.glob("*"))})

    rows, decode_log = [], {}
    all_subs = sorted(SUB_DIR.glob("*.png"))
    hash_groups = {}
    for f in all_subs:
        hash_groups.setdefault(sha256(f), []).append(f.name)
    dup_of = {}
    for names in hash_groups.values():
        if len(names) > 1:
            for name in names:
                dup_of[name] = ",".join(x for x in names if x != name)

    for p in all_subs:
        name, map_key, channel = parse_submission(p)
        if map_key not in MAPS:
            continue
        R = refs[map_key]
        P, meta = read_png_matrix(p)
        decode_log[p.name] = meta
        h = sha256(p)
        dup = dup_of.get(p.name, "")
        if P.shape != R.shape:
            print(f"[skip] {p.name} shape {P.shape} != {R.shape}", file=sys.stderr)
            continue

        H, W = R.shape
        cfg = MAPS[map_key]
        acc = accuracy(P, R)
        tp, tn, fp, fn = confusion(P, R)
        bacc = balanced_accuracy(P, R)
        hssv = hss(P, R)
        maj = majority_baseline(R)

        lats = np.linspace(cfg["lat_top"], cfg["lat_bot"], H)
        wmat = np.repeat(np.clip(np.cos(np.deg2rad(lats)), 1e-6, None)[:, None], W, axis=1)
        acc_area = float(np.sum((P == R) * wmat) / np.sum(wmat))

        dlat = abs(cfg["lat_top"] - cfg["lat_bot"]) / max(H - 1, 1)
        dlon = abs(cfg["lon_right"] - cfg["lon_left"]) / max(W - 1, 1)
        lat_mid = 0.5 * (cfg["lat_top"] + cfg["lat_bot"])
        km_per_cell = 111.32 * math.sqrt(dlat ** 2 + (dlon * math.cos(math.radians(lat_mid))) ** 2)

        cstats = component_stats(P, R, CONFIG["component_min_cells"], CONFIG["component_iou_match"])
        bf1 = {d: boundary_f1(P, R, d)[2] for d in CONFIG["boundary_tolerances"]}
        cham = chamfer_distances(P, R, km_per_cell)
        nw_keep, nw_tot = narrow_water_preservation(P, R, CONFIG["narrow_water_radius"])

        area_ratio = float(np.sum(P == 1)) / float(np.sum(R == 1)) if np.sum(R == 1) else float("nan")

        def centroid(M):
            ii, jj = np.mgrid[0:M.shape[0], 0:M.shape[1]]
            m = (M == 1) * wmat
            total = m.sum()
            return None if total <= 0 else (float((ii * m).sum() / total), float((jj * m).sum() / total))

        cP, cR = centroid(P), centroid(R)
        if cP and cR:
            drow = (cP[0] - cR[0]) * dlat
            dcol = (cP[1] - cR[1]) * dlon
            cent_km = 111.32 * math.sqrt(drow ** 2 + (dcol * math.cos(math.radians(lat_mid))) ** 2)
        else:
            cent_km = float("nan")

        rows.append(dict(
            file=p.name, model=name, map=map_key, channel=channel, sha256=h, duplicate_of=dup,
            n_cells=int(H * W), km_per_cell=km_per_cell,
            accuracy=acc, accuracy_area_weighted=acc_area, balanced_accuracy=bacc, hss=hssv,
            majority_baseline=maj, skill_score=(acc - maj) / (1 - maj) if maj < 1 else 0.0,
            extra_land=int(fp), missing_land=int(fn),
            land_frac_pred=float(np.mean(P == 1)), land_frac_ref=float(np.mean(R == 1)),
            area_ratio=area_ratio, centroid_offset_km=cent_km,
            n_ref_components=cstats["n_ref_components"], n_pred_components=cstats["n_pred_components"],
            component_recall=cstats["component_recall"],
            island_iou_median=cstats["island_iou_median"],
            island_iou_area_weighted=cstats["island_iou_area_weighted"],
            island_iou_ge50=cstats["island_iou_ge50"],
            boundary_f1_d0=bf1[0], boundary_f1_d1=bf1[1], boundary_f1_d2=bf1[2],
            boundary_mean_km=cham["mean_km"], boundary_p95_km=cham["p95_km"],
            narrow_water_keep=nw_keep, narrow_water_cells=nw_tot,
        ))

    B, blk, seed = CONFIG["bootstrap_B"], CONFIG["bootstrap_block"], CONFIG["bootstrap_seed"]
    for row in rows:
        P, _ = read_png_matrix(SUB_DIR / row["file"])
        R = refs[row["map"]]
        row["accuracy_ci_lo"], row["accuracy_ci_hi"] = block_bootstrap_ci(
            P, R, accuracy, B, blk, seed
        )
        lo, hi = block_bootstrap_ci(
            P, R, lambda a, b: boundary_f1(a, b, 1)[2], max(200, B // 4), blk, seed + 1
        )
        row["boundary_f1_ci_lo"], row["boundary_f1_ci_hi"] = lo, hi

    sens = []
    for key, cfg in MAPS.items():
        for vname, vfile in cfg.get("variants", {}).items():
            Rp = REF_DIR / vfile
            if not Rp.exists():
                continue
            R2 = read_csv_matrix(Rp)
            for row in rows:
                if row["map"] != key:
                    continue
                P, _ = read_png_matrix(SUB_DIR / row["file"])
                sens.append(dict(
                    map=key,
                    variant=vname,
                    model=row["model"],
                    file=row["file"],
                    accuracy_variant=accuracy(P, R2),
                    accuracy_primary=row["accuracy"],
                    delta_pp=(accuracy(P, R2) - row["accuracy"]) * 100.0,
                    n_changed_cells=int(np.sum(R2 != refs[key])),
                ))
    if sens:
        with open(OUT_DIR / "reference_sensitivity.csv", "w", newline="", encoding="utf-8") as f:
            w = csv.DictWriter(f, fieldnames=list(sens[0].keys()))
            w.writeheader()
            w.writerows(sens)

    if rows:
        with open(OUT_DIR / "metrics.csv", "w", newline="", encoding="utf-8") as f:
            w = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
            w.writeheader()
            w.writerows(rows)

    (OUT_DIR / "metrics.json").write_text(
        json.dumps(
            {
                "config": CONFIG,
                "maps": MAPS,
                "reference_meta": ref_meta,
                "rows": rows,
                "sensitivity": sens,
            },
            ensure_ascii=False,
            indent=1,
        ),
        encoding="utf-8",
    )
    (OUT_DIR / "hashes.json").write_text(
        json.dumps(hashes, ensure_ascii=False, indent=1), encoding="utf-8"
    )
    (OUT_DIR / "decode_log.json").write_text(
        json.dumps(decode_log, ensure_ascii=False, indent=1), encoding="utf-8"
    )

    print("\nmodel,map,acc,HSS,boundaryF1@1,narrowWater,islandIoU")
    for r in sorted(rows, key=lambda x: (x["map"], -x["accuracy"])):
        nw = r["narrow_water_keep"]
        nws = "nan" if nw != nw else f"{nw:.3f}"
        print(
            f"{r['model']},{r['map']},{r['accuracy']*100:.2f},{r['hss']:.3f},"
            f"{r['boundary_f1_d1']:.3f},{nws},{r['island_iou_median']:.3f}"
        )
    print(f"\n[ok] {len(rows)} submissions -> {OUT_DIR}")


if __name__ == "__main__":
    main()
