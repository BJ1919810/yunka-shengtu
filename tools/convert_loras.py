#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""把训练产出的 12 档 checkpoint 批量处理成 ComfyUI 可用的 LoRA。

流程（对每一档）：
    1. 从解压目录取 `pytorch_lora_weights.safetensors`（PEFT 原始格式）
    2. 复制成 `<名字>_<步数>.safetensors`（原始档，strength 要填 0.5）
    3. 跑 fix_lora_alpha.py 生成 `<名字>_<步数>_comfy.safetensors`（★ 推荐用这个，strength=1.0）
    4. 打印体积与 md5 前 8 位

⚠️ 为什么必须做第 3 步：PEFT 的 `lora_alpha` 只写在 safetensors 的 **metadata** 里，
   ComfyUI 的加载器不读 metadata ⇒ 静默按 alpha=1.0 算 ⇒ 倍率错（本次是 16/16=1.0，
   所以本次恰好没错；但流程保持一致，以后改 alpha 就不会踩）。详见 `.skills/75`。

用法：
    python tools/convert_loras.py <解压出来的目录> [--name bj_style] [--out 输出目录]
"""
from __future__ import annotations

import argparse
import glob
import os
import re
import shutil
import subprocess
import sys

import pathlib as _pl
_here = _pl.Path(__file__).resolve()
for _r in (_here.parent, *_here.parents):          # 向上找仓库根（含 paths.py）
    if (_r / "paths.py").exists():
        sys.path.insert(0, str(_r))
        break
import paths  # noqa: E402  （路径唯一权威）

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(HERE)
FIX = os.path.join(ROOT, "train_web", "fix_lora_alpha.py")
PY = sys.executable


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("src", help="存放 checkpoint-* 的目录（解压后的）")
    ap.add_argument("--name", default="bj_style", help="LoRA 名字前缀")
    ap.add_argument("--out", default=str(paths.QWEN_LORA_DIR),
                    help="输出目录（默认 Qwen-Image-2.1 的 loras 目录，见 paths.py）")
    a = ap.parse_args()

    ckpts = sorted(glob.glob(os.path.join(a.src, "checkpoint-*")),
                   key=lambda p: int(re.search(r"(\d+)$", p).group(1)))
    if not ckpts:
        # 也许直接传的就是含 pytorch_lora_weights.safetensors 的目录
        ckpts = [a.src] if os.path.isfile(
            os.path.join(a.src, "pytorch_lora_weights.safetensors")) else []
    if not ckpts:
        print("[x] 没找到 checkpoint-* 目录：%s" % a.src)
        return 2

    os.makedirs(a.out, exist_ok=True)
    print("=" * 74)
    print("批量转换 %d 档 → %s" % (len(ckpts), a.out))
    print("=" * 74)

    ok = 0
    for c in ckpts:
        m = re.search(r"(\d+)$", os.path.basename(c.rstrip("/\\")))
        step = m.group(1) if m else "final"
        src = os.path.join(c, "pytorch_lora_weights.safetensors")
        if not os.path.isfile(src):
            print("  ✗ ckpt-%s 缺 pytorch_lora_weights.safetensors，跳过" % step)
            continue

        raw = os.path.join(a.out, "%s_%s.safetensors" % (a.name, step))
        comfy = os.path.join(a.out, "%s_%s_comfy.safetensors" % (a.name, step))

        shutil.copy2(src, raw)
        r = subprocess.run([PY, FIX, raw, comfy], capture_output=True, text=True)
        if r.returncode != 0:
            print("  ✗ ckpt-%s 补 alpha 失败：%s" % (step, (r.stderr or r.stdout)[-200:]))
            continue
        ok += 1
        print("  ✓ ckpt-%-4s → %-34s %4.0f MB"
              % (step, os.path.basename(comfy), os.path.getsize(comfy) / 1048576))
        if r.stdout.strip():
            print("      " + r.stdout.strip().split("\n")[-1])

    print()
    print("完成 %d / %d 档" % (ok, len(ckpts)))
    print("★ ComfyUI 用 `_comfy` 版，strength 填 1.0")
    return 0 if ok else 1


if __name__ == "__main__":
    raise SystemExit(main())
