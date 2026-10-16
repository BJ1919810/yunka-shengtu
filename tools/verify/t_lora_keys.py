# -*- coding: utf-8 -*-
"""校验 LoRA 的模块名**能在 Anima 底模权重里找到**（防 ComfyUI 静默跳过）。

为什么必须有这个检查：ComfyUI 的 LoRA 加载器对**不认识的键只打一行 warning 然后跳过**
（`comfy/lora.py:93  logging.warning("lora key not loaded: {}")`，且 `comfy/lora.py` 的
`model_lora_keys_unet` 里**没有 anima 分支** ⇒ 只能靠"名字直接对上"）。
名字对不上的那部分权重**静默不生效**，症状是"风格不吃、只剩底模自己的味道"。

★ 基准必须取**底模权重自己的键名**（safetensors 只读头部，不载 3.9GB 权重），
  ⛔ 不能靠读代码猜类 —— 2026-10-16 我读了 `comfy/ldm/anima/model.py` 的 `TransformerBlock`
  （`o_proj`/`mlp.0`），差点误判；那是 **LLM 适配器**用的类，
  真正的 DiT 是 `comfy/ldm/cosmos/predict2.py` 的 `MiniTrainDIT`
  （`Attention.output_proj` + `GPT2FeedForward.layer1/layer2`），与底模权重一致。

用法：
    python tools/verify/t_lora_keys.py [LoRA文件] [底模文件]
默认取 `train_anima/_out/*/` 里最新的 `*_comfy.safetensors` 与 `anima-aesthetic-v1.1.safetensors`。
"""
from __future__ import annotations

import glob
import json
import os
import re
import struct
import sys

for _s in (sys.stdout, sys.stderr):
    try:
        _s.reconfigure(encoding="utf-8", errors="replace")
    except Exception:
        pass

import pathlib as _pl
_here = _pl.Path(__file__).resolve()
for _r in (_here.parent, *_here.parents):          # 向上找仓库根（含 paths.py）
    if (_r / "paths.py").exists():
        sys.path.insert(0, str(_r))
        break
import paths  # noqa: E402  （路径唯一权威）

HERE = os.path.dirname(os.path.abspath(__file__))
PROJ = os.path.dirname(os.path.dirname(HERE))
OUT_ROOT = os.path.join(PROJ, "train_anima", "_out")
DIT_DIR = str(paths.ANIMA_DIFFUSION_DIR)

FAILS: list[str] = []


def check(cond, msg):
    print("  %s %s" % ("✓" if cond else "✗", msg))
    if not cond:
        FAILS.append(msg)


def header_keys(path: str) -> list[str]:
    """只读 safetensors 头部，拿到全部键名（不加载张量）。"""
    with open(path, "rb") as f:
        n = struct.unpack("<Q", f.read(8))[0]
        hdr = json.loads(f.read(n).decode("utf-8"))
    return [k for k in hdr.keys() if k != "__metadata__"]


def newest_comfy() -> str | None:
    files = [p for p in glob.glob(os.path.join(OUT_ROOT, "*", "*_comfy.safetensors"))]
    return max(files, key=os.path.getmtime) if files else None


def main() -> int:
    lora = sys.argv[1] if len(sys.argv) > 1 else newest_comfy()
    dit = sys.argv[2] if len(sys.argv) > 2 else os.path.join(DIT_DIR, "anima-aesthetic-v1.1.safetensors")
    print("=" * 76)
    print("LoRA 模块名 ↔ Anima 底模权重（ComfyUI 静默跳过的判据）")
    print("=" * 76)
    if not lora or not os.path.isfile(lora):
        print("[x] 找不到 LoRA：%r" % lora)
        return 1
    if not os.path.isfile(dit):
        print("[x] 找不到底模：%r" % dit)
        return 1
    print("  LoRA : %s" % lora)
    print("  底模 : %s" % dit)

    # 底模：键前缀可能是 `model.diffusion_model.`（官方 v1.1 BF16 两档）
    #       或 `net.`（int8 量化版 `Anima-2.9B-preview-v1_int8_convrot`：2208 键、I8/U8 + 每权重一个
    #       `F32 *.weight_scale`）。⛔ 2026-10-16 实测：旧版只认前者 ⇒ 对 int8 报"解析出 0 个模块名"（工具缺口，已补）。
    mods = set()
    for k in header_keys(dit):
        for pfx in ("model.diffusion_model.", "net.", "diffusion_model."):
            if k.startswith(pfx) and k.endswith(".weight"):
                mods.add("diffusion_model." + k[len(pfx):-len(".weight")])
                break
    check(len(mods) > 200, "底模里解析出 %d 个模块名（作基准）" % len(mods))

    # LoRA：diffusion_model.blocks.N.<模块>.{alpha,lora_A.weight,lora_B.weight}
    lk = header_keys(lora)
    targets = {}
    for k in lk:
        m = re.match(r"^(diffusion_model\..+?)\.(alpha|lora_A\.weight|lora_B\.weight)$", k)
        if m:
            targets[m.group(1)] = targets.get(m.group(1), 0) + 1
    check(bool(targets), "LoRA 里解析出 %d 个训练目标" % len(targets))

    matched = [t for t in targets if t in mods]
    unmatched = sorted(t for t in targets if t not in mods)
    check(not unmatched, "全部 %d 个目标都能在底模里找到（对不上 = 会被静默跳过）" % len(targets))
    if unmatched:
        print("      ⛔ 对不上的（这些权重 ComfyUI 不会加载）：")
        for u in unmatched[:20]:
            print("         - %s" % u)

    # 作用面：确认含 MLP（管画风/纹理）
    mlp = [t for t in targets if ".mlp." in t]
    check(bool(mlp), "含 MLP 目标 %d 个（画风/纹理就靠它）" % len(mlp))
    per_block: dict[str, int] = {}
    for t in targets:
        per_block[t.split(".")[2]] = per_block.get(t.split(".")[2], 0) + 1
    print("      每层模块数分布: %s" % {k: v for k, v in sorted(per_block.items())[:3]})

    print()
    print("=" * 76)
    print("键名对齐 ✅（%d 个目标全部命中）" % len(targets) if not FAILS else "有问题 ❌（%d）" % len(FAILS))
    for f in FAILS:
        print("   ✗ " + f)
    print("=" * 76)
    return 0 if not FAILS else 1


if __name__ == "__main__":
    sys.exit(main())
