#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""实测各档 LoRA：真出图（768²，本机 3070 8G 跑得动），记录耗时与产物。

题面固定（含触发词 bj_style）、seed 固定，只换 LoRA 档位 ⇒ 可横向对比。
★ 服务是**异步**的：POST /api/generate 返回 pid ⇒ 轮询 /api/status?pid=… 直到出图。
"""
import json
import os
import sys
import time
import urllib.parse as PA
import urllib.request as U

import pathlib as _pl
_here = _pl.Path(__file__).resolve()
for _r in (_here.parent, *_here.parents):          # 向上找仓库根（含 paths.py）
    if (_r / "paths.py").exists():
        sys.path.insert(0, str(_r))
        break
import paths  # noqa: E402  （路径唯一权威）

B = "http://127.0.0.1:8770"
OUT = str(paths.REPO / "_lora_probe")
os.makedirs(OUT, exist_ok=True)

PROMPT = ("bj_style, 一位少女站在教室里看向镜头，浅金色及肩中长发，紫色瞳孔，"
          "身穿浅米色高领毛衣，背景是浅灰色砖墙，明亮柔和正面光，中景平视")
CASES = [
    ("none", "不使用", 20),
    ("acc8", "加速 8 步", None),      # 加速档自己接管步数
    ("acc5", "加速 5 步", None),
    ("bj",   "风格 bj_style", 20),
]


def post(path, data):
    req = U.Request(B + path, data=json.dumps(data).encode(), method="POST")
    req.add_header("Content-Type", "application/json")
    with U.urlopen(req, timeout=600) as r:
        return json.loads(r.read().decode("utf-8"))


def status(pid):
    with U.urlopen(B + "/api/status?" + PA.urlencode({"pid": pid}), timeout=120) as r:
        return json.loads(r.read().decode("utf-8"))


def fetch_image(fn, sub, dst):
    q = PA.urlencode({"filename": fn, "subfolder": sub or "", "type": "output"})
    with U.urlopen(B + "/api/image?" + q, timeout=300) as r:
        data = r.read()
    with open(dst, "wb") as f:
        f.write(data)
    return len(data)


print("=" * 80)
print("LoRA 各档实测（本机 3070 8G / 768² / 固定 seed=42）")
print("=" * 80)

rows = []
for key, label, steps in CASES:
    body = {
        "mode": "t2i", "prompt": PROMPT,
        "width": 768, "height": 768, "fixed": True,
        "cfg": 1.0, "sampler": "euler", "scheduler": "simple",
        "seed": 42, "lora": key, "prefix": "probe_" + key,
    }
    if steps is not None:
        body["steps"] = steps
    t0 = time.time()
    try:
        d = post("/api/generate", body)
    except Exception as e:
        print("[{}] ✗ 提交失败：{}".format(label, e))
        rows.append((label, None, None, None))
        continue
    if not d.get("ok"):
        print("[{}] ✗ 被拒：{}".format(label, d.get("error")))
        rows.append((label, None, None, None))
        continue
    pid = d["pid"]
    note = " | ".join(d.get("notes") or [])
    print("[{}] 已提交 pid={}  {}".format(label, pid[:8], note[:88]))

    # 轮询直到出图（本机 768² 大约 15~60s；给足 20 分钟）
    files, err = None, None
    while time.time() - t0 < 1200:
        time.sleep(3)
        try:
            st = status(pid)
        except Exception as e:
            err = str(e); continue
        if st.get("error"):
            err = st["error"]; break
        if st.get("done") or st.get("images"):
            files = st.get("images") or []
            break
        q = st.get("queue") or {}
        if q.get("status_str") == "error":
            err = json.dumps(q, ensure_ascii=False)[:200]; break
    dt = time.time() - t0

    if err:
        print("         ✗ 失败（{:.1f}s）：{}".format(dt, err))
        rows.append((label, dt, None, None)); continue
    if not files:
        print("         ✗ 超时未出图（{:.1f}s）".format(dt))
        rows.append((label, dt, None, None)); continue

    saved = []
    for im in files[:1]:
        fn = im.get("filename") if isinstance(im, dict) else str(im)
        sub = im.get("subfolder", "") if isinstance(im, dict) else ""
        dst = os.path.join(OUT, "{}_{}".format(key, fn))
        try:
            n = fetch_image(fn, sub, dst)
            saved.append((dst, n))
        except Exception as e:
            print("         取图失败：{}".format(e))
    print("         ✓ {:.1f}s  产物 {}  {} KB".format(
        dt, [os.path.basename(x[0]) for x in saved],
        [x[1] // 1024 for x in saved]))
    rows.append((label, dt, files, saved))

print()
print("=" * 80)
print("小结")
print("=" * 80)
for label, dt, files, saved in rows:
    print("  {:16s} {:>8s}  {}".format(
        label, ("%.1fs" % dt) if dt else "—",
        (saved[0][0] if saved else "（无产物）")))
