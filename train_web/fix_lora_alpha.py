#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""给 diffusers/PEFT 训出的 LoRA 补 `.alpha` 张量（ComfyUI 专用）。

## 为什么需要这个

peft 保存 LoRA 时，`lora_alpha`（本次是 16）**只写进 safetensors 的
`__metadata__` JSON**（`lora_adapter_metadata`），**不落 `.alpha` 张量**。

而 ComfyUI 的 `comfy/lora.py::load_lora()` 是这么取 alpha 的：

    alpha_name = "{x}.alpha".format(x)
    alpha = lora[alpha_name].item() if alpha_name in lora.keys() else None

⇒ 拿不到就 `alpha=None`，`calculate_weight()` / `LoRAAdapter.h()` 里
回落成 **`alpha = 1.0`**（见 `comfy/weight_adapter/lora.py` 248~251 / 314~317 行）。

本次训练 `rank=32, lora_alpha=16` ⇒ 正确倍率应为 `16/32 = 0.5`，
直接挂上去会被**放大 2 倍**——而且**静默无报错**，只是出图「风格齁浓、发黑发糊」。

（同一个坑 2026-10-08 在 pruna 加速 LoRA 上踩过，见 `.skills/11-显存闸与采样LoRA.md` 硬约束 1。）

## 这个脚本做什么

读 256 个 `*.lora_{A,B}.weight` ⇒ 推出 128 个目标模块 ⇒ 每个目标补一个
`.alpha` 标量张量（值 = metadata 里的 `transformer.lora_alpha`）⇒ 另存新文件。
**不碰任何原有张量**，只加键。补完后 strength 填 1.0 就是正确比例。

### ⛔ 写入的值是 `lora_alpha` 本身（本例 16），**不是** `alpha/rank`（0.5）

ComfyUI 自己会再除一次 rank（`weight_adapter/lora.py` 248~251 行）：

    alpha = v[2] / mat2.shape[0]     # v[2] = 我们写的 .alpha，mat2 = lora_A = [32, 4096]

⇒ 写 16 得到 `16/32 = 0.5`（正确）；若误写 0.5 会得到 `0.5/32 = 0.0156`，
**比正确值小 32 倍**（2026-10-14 写第一版时真踩了，靠 rank 口径推回来才发现）。

## 用法

    python fix_lora_alpha.py <输入.safetensors> [输出.safetensors]

省略输出则写成 `<输入名>_alpha.safetensors`。
"""

import json
import os
import struct
import sys

import torch
from safetensors.torch import load_file, save_file

LORA_SUFFIXES = (".lora_A.weight", ".lora_B.weight")


def read_metadata(path: str) -> dict:
    """只读 safetensors 头部的 __metadata__，不加载张量。"""
    with open(path, "rb") as f:
        n = struct.unpack("<Q", f.read(8))[0]
        header = json.loads(f.read(n))
    return header.get("__metadata__", {})


def peek_alpha(metadata: dict) -> tuple[int, int] | None:
    """从 lora_adapter_metadata 里挖 (lora_alpha, r)。挖不到返回 None。"""
    raw = metadata.get("lora_adapter_metadata")
    if not raw:
        return None
    try:
        info = json.loads(raw)
    except Exception:
        return None
    alphas, ranks = set(), set()
    for k, v in info.items():
        if k.endswith(".lora_alpha") and isinstance(v, int):
            alphas.add(v)
        if k.endswith(".r") and isinstance(v, int):
            ranks.add(v)
    if len(alphas) != 1 or len(ranks) != 1:
        return None
    return alphas.pop(), ranks.pop()


def fix(src: str, dst: str, alpha: float) -> None:
    sd = load_file(src)
    targets = sorted({k[: -len(s)] for s in LORA_SUFFIXES
                      for k in sd if k.endswith(s)})
    if not targets:
        raise SystemExit("✗ 没找到任何 lora_A/B.weight 键，这不是 LoRA 文件？")

    added = 0
    for t in targets:
        an = "{}.alpha".format(t)
        if an in sd:
            continue
        sd[an] = torch.tensor(float(alpha), dtype=torch.float32)
        added += 1

    md = read_metadata(src)
    md["_alpha_injected_by"] = "云卡生图/train_web/fix_lora_alpha.py"
    save_file(sd, dst, metadata=md)
    print("✓ {} → {}  ({} 个目标，新增 {} 个 .alpha，总键 {}，{} MB)".format(
        os.path.basename(src), os.path.basename(dst),
        len(targets), added, len(sd), os.path.getsize(dst) // 1048576))


def main() -> None:
    if len(sys.argv) < 2:
        raise SystemExit(__doc__)
    src = sys.argv[1]
    a = 1.0
    peek = peek_alpha(read_metadata(src))
    if peek:
        alpha_i, rank = peek
        a = float(alpha_i)          # ⛔ 写 alpha 原值，ComfyUI 会自己除 rank
        print("metadata: lora_alpha={} / r={}  ⇒ 写入 .alpha={}（ComfyUI 算 {}/{}={}）".format(
            alpha_i, rank, a, alpha_i, rank, a / rank))
    else:
        print("⚠️ metadata 里没找到 lora_alpha/r，按 alpha=1.0 写入")

    if len(sys.argv) > 2:
        dst = sys.argv[2]
    else:
        base, ext = os.path.splitext(src)
        dst = base + "_alpha" + ext
    fix(src, dst, a)


if __name__ == "__main__":
    main()
