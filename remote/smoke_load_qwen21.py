# -*- coding: utf-8 -*-
"""真·加载冒烟测试：把 DiT / 文本编码器从 ComfyUI 单文件真正读进内存。

和 `validate_qwen21_loader.py` 的区别：那个只比 key/形状（走 meta 设备、几秒钟、2G 内存也能跑），
这个会**真的把权重读出来铺到模型上**。

用法：
    /root/autodl-tmp/train-env/bin/python smoke_load_qwen21.py         # 两个都测
    /root/autodl-tmp/train-env/bin/python smoke_load_qwen21.py dit     # 只测 DiT
    /root/autodl-tmp/train-env/bin/python smoke_load_qwen21.py te      # 只测文本编码器

⚠️⚠️ **必须在有卡模式跑**。无卡模式容器 `memory.max` 只有 2 GiB（2026-10-12 实测），
   而 `free` 显示的是宿主机的 503G —— 会被误导。本脚本会先自检，内存不够直接拒绝跑。
"""
from __future__ import annotations

import gc
import os
import resource
import sys
import time

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import qwen21_loader  # noqa: E402


def mem_limit_gb() -> float | None:
    for p in ("/sys/fs/cgroup/memory.max", "/sys/fs/cgroup/memory/memory.limit_in_bytes"):
        try:
            with open(p) as f:
                v = f.read().strip()
            if v != "max":
                return int(v) / 1024 ** 3
        except OSError:
            continue
    return None


def peak_gb() -> float:
    return resource.getrusage(resource.RUSAGE_SELF).ru_maxrss / 1024.0 / 1024.0


lim = mem_limit_gb()
print("容器内存上限(memory.max)：%s" % ("%.1f GB" % lim if lim else "无限制"))
if lim is not None and lim < 8:
    sys.exit(
        "✗ 当前容器只有 %.1f GB 内存，跑不动（会 OOM Killed）。\n"
        "  无卡模式内存极小，请切**有卡模式**再跑；或设 Q21_LOAD_DEVICE=cuda 直接铺到显存。\n"
        "  （只是校验 key/形状的话，用 validate_qwen21_loader.py，2G 也够。）" % lim
    )


def report(name: str, fn) -> None:
    t0 = time.time()
    print("== %s 开始（当前峰值 %.1f GB）" % (name, peak_gb()), flush=True)
    model = fn()
    n = sum(p.numel() for p in model.parameters())
    print("   ✓ 加载成功：%d 个参数张量 / %.2f B 参数 / %s / 耗时 %.0f s / 峰值内存 %.1f GB"
          % (len(list(model.parameters())), n, next(model.parameters()).dtype,
             time.time() - t0, peak_gb()), flush=True)
    del model
    gc.collect()


what = (sys.argv[1] if len(sys.argv) > 1 else "all").lower()

if what in ("all", "dit"):
    report("DiT ← Comfy 单文件（含 gate_up 拆分）", qwen21_loader.load_transformer)
if what in ("all", "te", "text"):
    report("文本编码器 ← Comfy 单文件（含 language_model 前缀改名）",
           qwen21_loader.load_text_encoder)

print("全部通过 ✓")
