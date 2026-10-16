#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""把四档产物拼成一张 2×2 对比图（带标签），方便一眼看风格差异。

用法：python tools/verify/t_grid.py
从 ComfyUI 的 output/ 取图（probe_*.png），输出到项目 out/_LoRA四档对比.png。
"""
import os
import sys
from PIL import Image, ImageDraw, ImageFont

import pathlib as _pl
_here = _pl.Path(__file__).resolve()
for _r in (_here.parent, *_here.parents):          # 向上找仓库根（含 paths.py）
    if (_r / "paths.py").exists():
        sys.path.insert(0, str(_r))
        break
import paths  # noqa: E402  （路径唯一权威）

COMFY_OUT = str(paths.COMFY_OUTPUT)
DST = str(paths.REPO / "out")
ORDER = [
    ("probe_none_00001_.png", "不使用 LoRA (20步, 66s)"),
    ("probe_acc8_00001_.png", "加速 8 步 (36s)"),
    ("probe_acc5_00001_.png", "加速 5 步 (21s)"),
    ("probe_bj_00001_.png",   "风格 bj_style (20步, 48s)"),
]
S, BAR, cols = 512, 34, 2
rows = (len(ORDER) + cols - 1) // cols

os.makedirs(DST, exist_ok=True)
canvas = Image.new("RGB", (S * cols, (S + BAR) * rows), (250, 250, 252))
d = ImageDraw.Draw(canvas)
try:
    font = ImageFont.truetype(str(paths.WIN_FONTS / "msyh.ttc"), 20)
except Exception:
    font = ImageFont.load_default()

miss = []
for i, (fn, label) in enumerate(ORDER):
    p = os.path.join(COMFY_OUT, fn)
    if not os.path.exists(p):
        miss.append(fn)
        continue
    im = Image.open(p).convert("RGB").resize((S, S), Image.LANCZOS)
    cx, cy = (i % cols) * S, (i // cols) * (S + BAR)
    d.rectangle([cx, cy, cx + S, cy + BAR], fill=(38, 42, 52))
    d.text((cx + 12, cy + 6), label, fill=(255, 255, 255), font=font)
    canvas.paste(im, (cx, cy + BAR))

out = os.path.join(DST, "_LoRA四档对比.png")
canvas.save(out)
print("已生成:", out, canvas.size)
if miss:
    print("缺文件:", miss)
