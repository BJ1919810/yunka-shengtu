# -*- coding: utf-8 -*-
"""零权重加载的映射校验：只用 safetensors 头部 + meta 设备，不读任何权重字节。

思路：目标 key/形状从「在 meta 设备上按 config 建出模型」拿到，
      来源 key/形状从「Comfy 单文件的头部」拿到（含 img_mlp 拆分），
      两边比对 —— 几秒钟出结果，不占内存、不读 30G。

用法：/root/autodl-tmp/train-env/bin/python validate_qwen21_loader.py
"""
from __future__ import annotations

import os
import sys

import torch
from safetensors import safe_open

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from qwen21_loader import DIT_SRC, MODEL_DIR, TE_SRC  # noqa: E402

FAILED = []


def comfy_dit_meta() -> dict:
    out = {}
    with safe_open(DIT_SRC, framework="pt", device="cpu") as f:
        for k in f.keys():
            if k.endswith(".img_mlp.gate_up.weight"):
                sh = tuple(f.get_slice(k).get_shape())
                half = (sh[0] // 2,) + sh[1:]
                base = k[: -len(".gate_up.weight")]
                out[base + ".gate_layer.weight"] = half
                out[base + ".proj.weight"] = half
            else:
                out[k] = tuple(f.get_slice(k).get_shape())
    return out


def comfy_te_meta() -> dict:
    with safe_open(TE_SRC, framework="pt", device="cpu") as f:
        return {k: tuple(f.get_slice(k).get_shape()) for k in f.keys()}


def compare(name: str, target: dict, got: dict) -> None:
    missing = sorted(set(target) - set(got))
    extra = sorted(set(got) - set(target))
    bad = sorted(k for k in set(target) & set(got) if target[k] != got[k])
    print("== %s" % name)
    print("   模型 %d 个张量 ／ 来源 %d 个 → 缺失 %d ｜ 多余 %d ｜ 形状不符 %d"
          % (len(target), len(got), len(missing), len(extra), len(bad)))
    for k in missing[:5]:
        print("     缺失   %-58s %s" % (k, target[k]))
    for k in extra[:5]:
        print("     多余   %-58s %s" % (k, got[k]))
    for k in bad[:5]:
        print("     形状   %-58s 模型%s != 来源%s" % (k, target[k], got[k]))
    if missing or extra or bad:
        FAILED.append(name)


print("=" * 70)
from diffusers import QwenImage21Transformer2DModel  # noqa: E402

with torch.device("meta"):
    dit = QwenImage21Transformer2DModel.from_config(
        QwenImage21Transformer2DModel.load_config(os.path.join(MODEL_DIR, "transformer"))
    )
compare("DiT (QwenImage21Transformer2DModel)",
        {k: tuple(v.shape) for k, v in dit.state_dict().items()}, comfy_dit_meta())

print("=" * 70)
from transformers import Qwen3VLConfig, Qwen3VLForConditionalGeneration  # noqa: E402

with torch.device("meta"):
    te = Qwen3VLForConditionalGeneration._from_config(
        Qwen3VLConfig.from_pretrained(os.path.join(MODEL_DIR, "text_encoder"))
    )
target_te = {k: tuple(v.shape) for k, v in te.state_dict().items()}
renamed = {}
for k, v in comfy_te_meta().items():
    nk = k
    if k not in target_te and k.startswith("model."):
        alt = "model.language_model." + k[len("model."):]
        if alt in target_te:
            nk = alt
    renamed[nk] = v
compare("文本编码器 (Qwen3VLForConditionalGeneration)", target_te, renamed)

print("=" * 70)
print("结论：%s" % ("全部对齐 ✓" if not FAILED else "不对齐 ✗ → %s" % FAILED))
sys.exit(1 if FAILED else 0)
