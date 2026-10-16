#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""数据集「质量分级 + 张数测算」工具（本机跑，不连云卡）。

## 为什么需要它

训练脚本（`train_dreambooth_lora_qwenimage21.py`）的默认行为：
  · 固定缩放到 **1024x1024 方形**（`--resolution 1024`，训练台**没传** bucket 参数）
  · 缩放用 `InterpolationMode.BILINEAR`（第 966 行）—— **不是 LANCZOS**
  · 默认 `--center_crop` 为 False ⇒ 走 **RandomCrop**（随机裁）

⇒ 一张 541 短边的图，会被**放大 1.9 倍**再**随机裁掉约 30% 画面**。
   线稿被双线性插值放大后线条发虚，LoRA 学到的就是"一身模糊"。
   **这才是「面容模糊 / 笔触糊」的头号嫌疑，比作用面更靠前。**

## 它做什么

对目录里每张图算两个数：
  · `upscale` = 要达到 1024 需要的放大倍数（>1 就是假的细节）
  · `crop`    = 覆盖式缩放后必须裁掉的面积占比

然后分四级：
  A 精品   短边≥900 且 放大≤1.20     —— 几乎无损，首选
  B 可用   短边≥700 且 放大≤1.50     —— 勉强能上
  C 勉强   短边≥512 且 放大≤2.00     —— 只会在"张数不够"时兜底
  D 该弃   其余                        —— 放大>2 或短边<512，建议**直接不要**

## 用法

    python tools/pick_dataset.py <目录>              # 只看分级
    python tools/pick_dataset.py <目录> --min B      # 按等级筛出清单（A+B）
    python tools/pick_dataset.py <目录> --min B --copy 输出目录
        ⇒ 把 A+B 级的「图片+同名txt」复制到新目录（**不删原文件**）

⚠️ 本工具**只读原目录**，`--copy` 也只往新目录写；不会动、不会删任何原图。
"""
from __future__ import annotations

import argparse
import collections
import glob
import os
import shutil
import statistics
import sys

try:
    from PIL import Image
except Exception:
    print("需要 Pillow：用隔离 venv 的 python 跑本脚本")
    raise SystemExit(2)

IMG_EXT = (".png", ".jpg", ".jpeg", ".webp", ".bmp")
TXT_EXT = (".txt",)
RES = 1024                     # 与 --resolution 一致
BILINEAR = True                # 训练脚本用的是双线性插值
DIVISIBILITY = 32              # SIZE_MULTIPLE_OF（训练脚本第 104 行）
EXTREME_RATIO = 3.0            # 长宽比 > 3:1 就算极端条（桶列表覆盖不到，必被裁）

TIERS = {
    "A": ("精品", "短边≥900 且 放大≤1.20"),
    "B": ("可用", "短边≥700 且 放大≤1.50"),
    "C": ("勉强", "短边≥512 且 放大≤2.00"),
    "D": ("该弃", "放大>2.00 或 短边<512"),
}
ORDER = ["A", "B", "C", "D"]


# ★ 逐字复刻 diffusers/training_utils.py（2026-10-14 从云卡源码抄的，勿凭印象改）
#   比例表是 (宽, 高) 形式的 11 个常用比例
_DEFAULT_BUCKET_ASPECT_RATIOS = [
    (1, 1), (4, 5), (5, 4), (2, 3), (3, 2), (3, 4), (4, 3),
    (9, 16), (16, 9), (1, 2), (2, 1),
]


def bucket_ladder(resolution: int, divisibility: int = DIVISIBILITY) -> list[tuple[int, int]]:
    """复刻 diffusers.generate_aspect_ratio_buckets。

    算法：令 h*w ≈ resolution²，对每个比例 (rw, rh) 解 h 使 w/h = rw/rh，再按
    divisibility 取整（用 round，不是 int 截断 —— 这点当初凭实测反推时搞错过）。
    实测 resolution=1024 / divisibility=32 产出 11 个桶：
        1152x928, 928x1152, 1184x896, 896x1184, 1440x736, 736x1440,
        1376x768, 768x1376, 1024x1024, 1248x832, 832x1248
    """
    target_pixels = resolution * resolution
    buckets, seen = [], set()
    for rw, rh in _DEFAULT_BUCKET_ASPECT_RATIOS:
        aspect = rw / rh
        h = (target_pixels / aspect) ** 0.5
        w = h * aspect
        h = max(divisibility, round(h / divisibility) * divisibility)
        w = max(divisibility, round(w / divisibility) * divisibility)
        if (h, w) not in seen:
            seen.add((h, w))
            buckets.append((h, w))
    return buckets


def nearest_bucket(h: int, w: int, ladder: list[tuple[int, int]]) -> tuple[int, int]:
    """复刻 diffusers.find_nearest_bucket：判据是 |h*bw - w*bh|（交叉相乘，比较比例）。

    ⚠️ 不是「长宽比差 + 面积差」那种加权 —— 最初凭实测反推时写错了，这里已改正。
    """
    best, best_metric = ladder[0], float("inf")
    for bh, bw in ladder:
        metric = abs(h * bw - w * bh)
        if metric <= best_metric:
            best_metric = metric
            best = (bh, bw)
    return best


def measure(path: str, use_buckets: bool) -> dict:
    w, h = Image.open(path).size
    if use_buckets:
        # 脚本逻辑（第 952 行）：resolution = min(RES, round(sqrt(h*w)))
        r = min(RES, round((w * h) ** 0.5))
        ladder = bucket_ladder(r)
        th, tw = nearest_bucket(h, w, ladder)
    else:
        th = tw = RES
    scale = max(th / h, tw / w)            # 覆盖式缩放（cover）
    nh, nw = round(h * scale), round(w * scale)
    crop = 1 - (th * tw) / (nh * nw)       # 被 RandomCrop 裁掉的面积占比
    short = min(w, h)
    ratio = max(w, h) / max(1, short)
    return {"w": w, "h": h, "short": short, "upscale": scale, "crop": crop,
            "th": th, "tw": tw, "ratio": ratio, "mp": w * h / 1e6}


def tier_of(m: dict) -> str:
    s, u = m["short"], m["upscale"]
    if s >= 900 and u <= 1.20:
        return "A"
    if s >= 700 and u <= 1.50:
        return "B"
    if s >= 512 and u <= 2.00:
        return "C"
    return "D"


def main() -> int:
    ap = argparse.ArgumentParser(add_help=True)
    ap.add_argument("dir", help="数据集目录（图片 + 同名 txt）")
    ap.add_argument("--min", default="D", choices=ORDER,
                    help="筛出 <= 该等级 的图（A=只精品，B=A+B，默认 D=全列）")
    ap.add_argument("--buckets", action="store_true",
                    help="按「训练时开了分桶」计算（小图分小桶不放大；大图按比例落桶不裁切）")
    ap.add_argument("--copy", default=None,
                    help="把筛出的图 + 同名 txt 复制到这个新目录")
    ap.add_argument("--top", type=int, default=0,
                    help="筛完后只保留最好的前 N 张（0=不限）")
    a = ap.parse_args()

    d = a.dir
    if not os.path.isdir(d):
        print("[x] 目录不存在：%s" % d)
        return 2

    names = [f for f in os.listdir(d) if os.path.isfile(os.path.join(d, f))]
    imgs = sorted(f for f in names if os.path.splitext(f)[1].lower() in IMG_EXT)

    rows = []
    for f in imgs:
        try:
            m = measure(os.path.join(d, f), a.buckets)
        except Exception as e:
            print("  [x] 打不开 %s: %s" % (f, e))
            continue
        m["f"] = f
        m["t"] = tier_of(m)
        m["extreme"] = m["ratio"] > EXTREME_RATIO
        m["has_txt"] = os.path.isfile(os.path.join(d, os.path.splitext(f)[0] + ".txt"))
        rows.append(m)

    if not rows:
        print("[x] 一张图都没有（只认 %s）" % "/".join(IMG_EXT))
        return 1

    rows.sort(key=lambda r: (ORDER.index(r["t"]), r["upscale"]))

    mode = "开分桶（小图分小桶，大图按比例落桶）" if a.buckets else "固定 %d² 方形 + RandomCrop" % RES
    # ---- 分级明细 ----
    print("=" * 78)
    print("数据集质量分级  ——  目标训练分辨率 %d" % RES)
    print("当前模式：%s" % mode)
    print("插值方式：BILINEAR（torchvision 0.24.1 的 antialias 默认 True ⇒ 缩小无摩尔纹）")
    print("=" * 78)
    cnt = collections.Counter(r["t"] for r in rows)
    for t in ORDER:
        label, rule = TIERS[t]
        sub = [r for r in rows if r["t"] == t]
        print()
        print("【%s】%s —— %d 张  （%s）" % (t, label, len(sub), rule))
        if not sub:
            print("      （无）")
            continue
        for r in sub:
            flags = ""
            if not r["has_txt"]:
                flags += "  ⚠无txt"
            if r["extreme"]:
                flags += "  ⚠极端长宽比(必被裁)"
            print("      %-10s %-11s %5.2fMP 短边%5d → 训练%4dx%-4d  缩放x%.2f  裁掉%5.1f%%%s"
                  % (r["f"], "%dx%d" % (r["w"], r["h"]), r["mp"], r["short"],
                     r["tw"], r["th"], r["upscale"], r["crop"] * 100, flags))

    # ---- 累计测算 ----
    print()
    print("=" * 74)
    print("保留方案测算")
    print("=" * 74)
    cum = 0
    for t in ORDER:
        cum += cnt[t]
        print("  只保留 A~%s  ⇒  %2d 张" % (t, cum))

    ups = [r["upscale"] for r in rows]
    cr = [r["crop"] for r in rows]
    mps = [r["mp"] for r in rows]
    print()
    print("  全体：%.2f MP 中位（%.2f ~ %.2f）｜缩放中位 x%.2f（最大 x%.2f）｜裁掉中位 %.1f%%（最大 %.1f%%）"
          % (statistics.median(mps), min(mps), max(mps),
             statistics.median(ups), max(ups),
             statistics.median(cr) * 100, max(cr) * 100))
    bad = sum(1 for u in ups if u > 1.5)
    print("  放大 >1.5 倍的：%d / %d 张" % (bad, len(rows)))
    ex = [r for r in rows if r["extreme"]]
    if ex:
        print("  ⚠ 极端长宽比(>%.0f:1) 的：%d 张 ⇒ 即便开分桶也会被裁 ~50%%：%s"
              % (EXTREME_RATIO, len(ex), "、".join(r["f"] for r in ex[:6])))
    shrink = [r for r in rows if r["upscale"] <= 1.0]
    if shrink:
        print("  ✅ 缩放 ≤1.0（无需放大，质量安全）：%d 张" % len(shrink))

    # ---- 给建议 ----
    maxidx = max(ORDER.index(a.min), 0)
    picked = [r for r in rows if ORDER.index(r["t"]) <= maxidx]
    if a.top > 0:
        picked = picked[:a.top]

    print()
    print("=" * 74)
    print("★ 建议")
    print("=" * 74)
    nA, nB = cnt["A"], cnt["B"]
    if nA + nB >= 12:
        print("  A+B 已有 %d 张 ⇒ **足够开训**（风格 LoRA 10~25 张就够，多了只是重复）" % (nA + nB))
    elif nA + nB >= 8:
        print("  A+B 只有 %d 张 ⇒ 偏少但可训；若能再补几张同风格的更好" % (nA + nB))
    else:
        print("  A+B 只有 %d 张 ⇒ **建议先补图再训**（低于 8 张容易只学到构图、学不到风格）" % (nA + nB))
    if cnt["C"]:
        print("  C 级 %d 张：只有「A+B 凑不满 15 张」时才拿来兜底，否则别用" % cnt["C"])
    if cnt["D"]:
        print("  D 级 %d 张：**直接不要**（放大>2 倍或短边<512，学进去就是糊）" % cnt["D"])
    print()
    print("  补图的硬门槛（照这个去收集，最省事）：")
    print("     ★ 短边 ≥1024 = 下限（不放大）；2~4MP 是甜点区（缩放后还有余量给裁切）")
    print("       —— 大图**不是问题**：缩小走 antialias 双线性，实测 2MP 方 → x0.72、裁 0%")
    print("     ⛔ 但长宽比别超 %.0f:1 —— 极端条即便开分桶也要被裁 50%%" % EXTREME_RATIO)
    print("     ⛔ 短边 <512 直接不要（放大 >2 倍，线稿必糊）")
    if not a.buckets:
        print()
        print("  ⚠ 你现在看的是「不开分桶」的结果。不开时所有长宽比都要被裁到方形：")
        print("     2MP 横 3:2 → 裁 33%；4K 16:9 → 裁 44%  ⇒ 建议加 --buckets 再看一遍")

    if a.copy:
        os.makedirs(a.copy, exist_ok=True)
        n = 0
        for r in picked:
            src = os.path.join(d, r["f"])
            shutil.copy2(src, os.path.join(a.copy, r["f"]))
            tp = os.path.join(d, os.path.splitext(r["f"])[0] + ".txt")
            if os.path.isfile(tp):
                shutil.copy2(tp, os.path.join(
                    a.copy, os.path.splitext(r["f"])[0] + ".txt"))
            n += 1
        print()
        print("  已复制 %d 张（含同名 txt）→ %s" % (n, a.copy))
        print("  ⚠️ 原目录未做任何改动")
    elif a.min != "D":
        print()
        print("  （想看清单文件，加 --copy <新目录>；原目录不会被改动）")

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
