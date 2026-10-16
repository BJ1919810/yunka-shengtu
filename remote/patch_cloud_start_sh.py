#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
把云卡镜像自带的 /root/start.sh 里的「直接 nohup 起 ComfyUI」那一段，
换成调用我们统一的 /root/autodl-tmp/start_comfy.sh。

为什么：原写法**没有幂等检查** —— 已在跑时再执行会起第二个进程、端口冲突失败。
新写法由 start_comfy.sh 收口：已在跑直接返回（绝不重复起）、等端口 200 才报成功、
日志统一到 /root/autodl-tmp/comfy.log、并清理僵死的旧进程。

安全措施：
  · 先备份成 /root/start.sh.bak-<日期>
  · 用「锚点定位 + 整段替换」，不依赖精确缩进
  · 找不到锚点就**直接退出、一个字不改**
用法：python3 _patch_start_sh.py
"""
import io
import os
import shutil
import sys

P = "/root/start.sh"
BAK = "/root/start.sh.bak-20261010"

START_ANCHOR = "# ==================== 启动 ComfyUI（6006） ===================="
END_ANCHOR = 'echo "✅ ComfyUI 已启动：http://127.0.0.1:6006"'
OLD_LOGPATH = 'LOG_PATH = "/root/logs/comfyui.log"'
NEW_LOGPATH = 'LOG_PATH = "/root/autodl-tmp/comfy.log"'

NEW_BLOCK = """# ==================== 启动 ComfyUI（6006） ====================
# 2026-10-10 改：不再直接 nohup，改为调用统一启动脚本 /root/autodl-tmp/start_comfy.sh
#   ① 幂等 —— 已在跑就立刻返回，**绝不会起第二个**（原写法没有这个检查）
#   ② 会等端口返回 200 才算成功；失败时打印日志尾部，便于排查
#   ③ 日志统一到 /root/autodl-tmp/comfy.log（下方 6008 日志页已同步指向）
#   ④ 顺带清理僵死的旧进程（升级 ComfyUI 代码后常见）
# ⚠️ 判断用 `-f` **不要用 `-x`** —— 上传的文件默认 644，没有可执行位，
#    用 `-x` 会误报"不存在"（2026-10-10 实测踩过）。`bash 脚本` 本来也不需要可执行位。
if [ -f /root/autodl-tmp/start_comfy.sh ]; then
    bash /root/autodl-tmp/start_comfy.sh
else
    echo "[ERROR] /root/autodl-tmp/start_comfy.sh 不存在，ComfyUI 未启动" >&2
fi
"""


def main():
    if not os.path.isfile(P):
        print("!! %s 不存在" % P)
        return 1
    s = io.open(P, encoding="utf-8").read()
    orig = s

    # --- 1) 换掉启动段（锚点定位，容错缩进差异）---
    if START_ANCHOR not in s or END_ANCHOR not in s:
        print("!! 找不到锚点：start=%s end=%s —— **文件未改动**"
              % (START_ANCHOR in s, END_ANCHOR in s))
        return 2
    i = s.index(START_ANCHOR)
    j = s.index(END_ANCHOR)
    j = s.index("\n", j) + 1          # 吃到该行末尾（含换行）
    s = s[:i] + NEW_BLOCK + s[j:]

    # --- 2) 6008 日志页指到新日志 ---
    n_logpath = s.count(OLD_LOGPATH)
    s = s.replace(OLD_LOGPATH, NEW_LOGPATH)

    if s == orig:
        print("!! 没有任何改动（可能已经改过了）")
        return 3

    shutil.copy2(P, BAK)
    with io.open(P, "w", encoding="utf-8", newline="\n") as f:
        f.write(s)

    print("OK 已改；备份 = %s" % BAK)
    print("   start_comfy.sh 调用点 : %d 处" % s.count("start_comfy.sh"))
    print("   LOG_PATH 换成新日志    : %s（命中 %d 处）"
          % (NEW_LOGPATH in s, n_logpath))
    print("   旧 nohup main.py 残留  : %s" % ("nohup /root/miniconda3/bin/python main.py" in s))
    print()
    print("--- 改后的启动段 ---")
    a = s.index(START_ANCHOR)
    b = s.index("# ==================== 6008")
    print(s[a:b].rstrip())
    return 0


if __name__ == "__main__":
    sys.exit(main())
