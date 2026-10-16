# -*- coding: utf-8 -*-
"""云卡数据集体检：配对 / 触发起头 / 尺寸分布 / 是否有小图。

用法： /root/autodl-tmp/train-env/bin/python check_dataset.py /root/autodl-tmp/dataset
"""
from __future__ import annotations

import glob
import os
import statistics
import sys

import collections
from PIL import Image

d = sys.argv[1] if len(sys.argv) > 1 else "/root/autodl-tmp/dataset"
os.chdir(d)

imgs = sorted(glob.glob("*.png") + glob.glob("*.jpg") + glob.glob("*.jpeg") + glob.glob("*.webp"))
txts = sorted(glob.glob("*.txt"))

print("=== 文件 ===")
print("  图片 %d   txt %d" % (len(imgs), len(txts)))

subdirs = [x for x in os.listdir(".") if os.path.isdir(x)]
print("  子目录（应为空）:", subdirs)

stems_i = {os.path.splitext(x)[0] for x in imgs}
stems_t = {os.path.splitext(x)[0] for x in txts}
missing = sorted(stems_i - stems_t)
orphan = sorted(stems_t - stems_i)
print("  图缺 txt:", missing if missing else "无")
print("  txt 缺图:", orphan if orphan else "无")

bad = []
for t in txts:
    with open(t, encoding="utf-8-sig") as f:
        first = f.readline()
    if not first.startswith("bj_style"):
        bad.append(t)
print("  非 bj_style 开头:", bad if bad else "无")

empty = [t for t in txts if os.path.getsize(t) < 20]
print("  内容过短(<20B):", empty if empty else "无")

print("=== 尺寸 ===")
sizes = [Image.open(p).size for p in imgs]
mode_dist = collections.Counter(Image.open(p).mode for p in imgs)
print("  色彩模式分布:", dict(mode_dist))
ws = [s[0] for s in sizes]
hs = [s[1] for s in sizes]
print("  宽 min/median/max: %d / %g / %d" % (min(ws), statistics.median(ws), max(ws)))
print("  高 min/median/max: %d / %g / %d" % (min(hs), statistics.median(hs), max(hs)))
small = sorted(s for s in sizes if min(s) < 512)
print("  短边 <512 的共 %d 张:" % len(small))
for s in small:
    print("     ", s)

print("=== caption 长度 ===")
lens = []
for t in txts:
    with open(t, encoding="utf-8-sig") as f:
        lens.append(len(f.read().strip()))
print("  字符数 min/median/max: %d / %g / %d" % (min(lens), statistics.median(lens), max(lens)))

print("=== 总字节 ===")
total = sum(os.path.getsize(p) for p in imgs)
print("  图片合计 %.1f MB" % (total / 1024 / 1024))
