# -*- coding: utf-8 -*-
"""去掉 caption(.txt) 里的 UTF-8 BOM —— 为什么必须去：

kohya 读 caption 用的是 `encoding="utf-8"`（**不去 BOM**），BOM 会**粘在触发词上**：
2026-10-16 用户的训练记录 `ss_tag_frequency` 里就统计出了 `\ufeffjm_style`（**15/15 全带 BOM**）
⇒ 训练时学到的触发词是「BOM + jm_style」，而出图时你在提示词里写的是 `jm_style`
⇒ **触发词对不上，风格基本不生效**，只剩底模自己的味道（表现为"风格不吃/画风跑偏"）。

本工具只做一件事：把这些 .txt 重存成「UTF-8 无 BOM」（**内容一个字节都不改**，只去掉开头 3 字节）。

用法：
    python tools/strip_bom.py                      # 默认处理 train_anima/_staging
    python tools/strip_bom.py <某个目录>            # 处理该目录下的 *.txt
    python tools/strip_bom.py <目录> --dry-run      # 只列出来，不动文件
"""
from __future__ import annotations

import os
import sys

for _s in (sys.stdout, sys.stderr):
    try:
        _s.reconfigure(encoding="utf-8", errors="replace")
    except Exception:
        pass

HERE = os.path.dirname(os.path.abspath(__file__))
PROJ = os.path.dirname(HERE)
DEFAULT_DIR = os.path.join(PROJ, "train_anima", "_staging")
BOM = b"\xef\xbb\xbf"


def main() -> int:
    args = [a for a in sys.argv[1:] if not a.startswith("--")]
    dry = "--dry-run" in sys.argv
    d = args[0] if args else DEFAULT_DIR
    print("=" * 74)
    print("去掉 caption 的 UTF-8 BOM%s" % ("（演练，不写盘）" if dry else ""))
    print("=" * 74)
    print("  目录：%s" % d)
    if not os.path.isdir(d):
        print("[x] 目录不存在")
        return 1

    files = sorted(f for f in os.listdir(d) if f.lower().endswith(".txt"))
    bom = []
    for f in files:
        p = os.path.join(d, f)
        try:
            with open(p, "rb") as fh:
                raw = fh.read()
        except OSError as e:
            print("  ⚠️ 读不了 %s：%s" % (f, e))
            continue
        if raw.startswith(BOM):
            bom.append((f, raw))
    print("  共 %d 个 .txt，其中 **%d 个带 BOM**" % (len(files), len(bom)))
    if not bom:
        print("\n  ✓ 没有需要处理的（全部已无 BOM）")
        return 0

    for f, raw in bom:
        body = raw[3:]
        # 只做"去掉前 3 字节"，但要确认剩下的确实是合法 UTF-8，免得把别的编码弄坏
        try:
            text = body.decode("utf-8")
        except UnicodeDecodeError as e:
            print("  ⛔ %s 去掉 BOM 后不是合法 UTF-8（%s）⇒ 跳过，别动它" % (f, e))
            continue
        head = text.strip().splitlines()[0][:42] if text.strip() else "(空)"
        if dry:
            print("    将去 BOM：%-14s 首行：%s" % (f, head))
        else:
            with open(os.path.join(d, f), "wb") as fh:
                fh.write(body)
            print("    已去 BOM：%-14s 首行：%s" % (f, head))
    if not dry:
        print("\n  ✓ 完成。**重跑一次体检**确认「带 BOM」提示消失，再开训。")
    return 0


if __name__ == "__main__":
    sys.exit(main())
