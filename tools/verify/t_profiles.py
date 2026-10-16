#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""三档位（cloud / local / anima）回归验证。

背景：2026-10-15 把 comfy_web 从「本机/云卡」二选一改成三选一，
      新增 anima 档（本机跑 Anima 二次元底模）。
      Anima 与 Qwen-Image-2.1 的**节点图完全不同**，所以必须防两件事：
        ① Qwen 链路被改坏（回归）
        ② Anima 链路悄悄用了 Qwen 的节点/参数（串档）

⚠️ 别用「工作流 md5」做回归断言 —— 默认 seed=-1 时 normalize 会**随机化 seed**，
   同一份输入两次跑出来哈希就不同（2026-10-15 我先栽在这上面，误报 7 处不一致）。
   ⇒ 断言必须**看节点结构**，不看整份哈希。

用法：  cd tools/verify && python t_profiles.py
"""
import json
import os
import sys

import pathlib as _pl
_here = _pl.Path(__file__).resolve()
for _r in (_here.parent, *_here.parents):          # 向上找仓库根（含 paths.py）
    if (_r / "paths.py").exists():
        sys.path.insert(0, str(_r))
        break
import paths  # noqa: E402  （路径唯一权威）

WEB = str(paths.COMFY_WEB)
sys.path.insert(0, WEB)
import server as S   # noqa: E402

FAILS = []


def check(cond, msg):
    print("  %s %s" % ("✓" if cond else "✗", msg))
    if not cond:
        FAILS.append(msg)


def wf(**kw):
    """固定 seed，避免随机化干扰。

    ⚠️ 别在这里硬塞 steps —— 那会盖掉档位默认值，
       于是「anima 档 steps 应该是 30」这条断言永远失败（第一版就栽了）。
    """
    p = S.normalize({"prompt": "x", "seed": 42, **kw})
    return S.build_workflow(p)


def types(w):
    return {n["class_type"] for n in w.values()}


def main() -> int:
    print("=" * 74)
    print("一、cloud 档 —— Qwen-Image-2.1 链路回归（必须与加档位前一致）")
    print("=" * 74)
    S.apply_profile("cloud")
    w = wf(lora="none")
    t = types(w)
    for need in ("UNETLoader", "CLIPLoader", "VAELoader", "QwenImage21Cache",
                 "TextEncodeQwenImage21", "KSampler", "VAEDecode", "SaveImage"):
        check(need in t, "含 %s" % need)
    check(w["38"]["inputs"]["type"] == "qwen_image", "CLIPLoader type = qwen_image")
    check(w["5"]["inputs"]["negative_prompt"] == "", "Qwen 负向恒为空串（cfg=1 不参与计算）")
    check("CLIPTextEncode" not in t, "不含 Anima 用的标准 CLIPTextEncode（没串档）")

    w = wf(lora="bj2_450")
    check(w["9"]["class_type"] == "LoraLoaderModelOnly", "风格档挂 LoraLoaderModelOnly")
    check(w["3"]["class_type"] == "KSampler", "风格档走 KSampler（不是蒸馏链路）")
    check(w["66"]["inputs"]["model"] == ["9", 0], "LoRA 在 Cache **之前**")

    w = wf(lora="acc8")
    check(w["3"]["class_type"] == "SamplerCustom", "加速档走 SamplerCustom")
    check("ManualSigmas" in types(w), "加速档含 ManualSigmas")

    print()
    print("=" * 74)
    print("二、anima 档 —— Anima 链路（照官方 blueprint）")
    print("=" * 74)
    S.apply_profile("anima")
    w = wf()
    t = types(w)
    check(w["37"]["inputs"]["unet_name"] == "anima-aesthetic-v1.1.safetensors", "UNETLoader = anima-aesthetic-v1.1")
    check(w["38"]["inputs"]["clip_name"] == "qwen_3_06b_base.safetensors", "CLIPLoader = qwen_3_06b_base")
    check(w["38"]["inputs"]["type"] == "stable_diffusion", "CLIPLoader type = stable_diffusion（官方就这么填）")
    check(w["39"]["inputs"]["vae_name"] == "qwen_image_vae.safetensors", "VAELoader = qwen_image_vae")
    check(w["5"]["class_type"] == "EmptyLatentImage", "latent 来自 EmptyLatentImage")
    check("CLIPTextEncode" in t, "用标准 CLIPTextEncode")
    check("TextEncodeQwenImage21" not in t, "不含 Qwen 专用编码节点")
    check("QwenImage21Cache" not in t, "不含 QwenImage21Cache（Anima 不需要）")
    check("LoraLoaderModelOnly" not in t, "默认不挂 LoRA")
    check(w["3"]["inputs"]["sampler_name"] == "er_sde", "采样器 = er_sde")
    check(w["3"]["inputs"]["scheduler"] == "simple", "调度器 = simple")
    check(w["3"]["inputs"]["cfg"] == 4.0, "CFG = 4.0")
    check(w["3"]["inputs"]["steps"] == 30, "步数 = 30")
    check("worst quality" in w["7"]["inputs"]["text"], "负向提示词 = 官方那串（cfg>1 ⇒ 真的生效）")
    check(w["3"]["inputs"]["negative"] == ["7", 0], "KSampler 接了负向分支")

    print()
    print("=" * 74)
    print("二之二、anima 档 —— 裸 i2i（2026-10-15 加）")
    print("=" * 74)
    wi = wf(mode="i2i", image="1.png", refs=["1.png"])
    ti = types(wi)
    check("LoadImage" in ti, "含 LoadImage")
    check("ImageScale" in ti, "含 ImageScale（缩放源图到目标画幅）")
    check("VAEEncode" in ti, "含 VAEEncode")
    check(wi["3"]["inputs"]["latent_image"] == ["72", 0], "KSampler 的 latent 来自 VAEEncode")
    check(wi["3"]["inputs"]["denoise"] == 0.7, "denoise = 0.7（档位默认）")
    check("EmptyLatentImage" not in ti, "i2i 不该再有 EmptyLatentImage")
    # t2i 仍走 EmptyLatentImage + denoise 1.0
    wt = wf()
    check("EmptyLatentImage" in types(wt), "t2i 仍用 EmptyLatentImage")
    check(wt["3"]["inputs"]["denoise"] == 1.0, "t2i 的 denoise 恒 1.0")
    # denoise 可调
    wd = wf(mode="i2i", image="1.png", refs=["1.png"], denoise=0.35)
    check(wd["3"]["inputs"]["denoise"] == 0.35, "denoise 可被请求覆盖（0.35）")
    # 越界要拒
    try:
        S.normalize({"prompt": "x", "seed": 42, "denoise": 1.8})
        check(False, "denoise=1.8 应被拒")
    except ValueError:
        check(True, "denoise 超出 0~1 被拒")

    print()
    print("=" * 74)
    print("二之三、anima 档 —— ControlNet-LLLite 控制图（2026-10-15 加）")
    print("=" * 74)
    wc = wf(mode="i2i", image="1.png", refs=["1.png"], control="canny")
    tc = types(wc)
    check("ModelPatchLoader" in tc, "canny：含 ModelPatchLoader")
    check("AnimaLLLiteApply" in tc, "canny：含 AnimaLLLiteApply")
    check("Canny" in tc, "canny：含核心 Canny 节点（无需装插件）")
    check(wc["82"]["inputs"]["image"] == ["81", 0], "LLLite 吃 Canny 的输出")
    check(wc["3"]["inputs"]["model"] == ["82", 0], "KSampler.model 接 LLLite 输出")
    check(wc["80"]["inputs"]["name"] == "anima-lllite-any-test-like-v2.safetensors",
          "默认控制模型 = 多合一版")
    wd = wf(mode="i2i", image="1.png", refs=["1.png"], control="direct")
    check("Canny" not in types(wd), "direct：不经过 Canny（源图直接当控制图）")
    check(wd["82"]["inputs"]["image"] == ["r0", 0], "direct：LLLite 吃源图")
    wn = wf(mode="i2i", image="1.png", refs=["1.png"])
    check("AnimaLLLiteApply" not in types(wn), "control=none 时不加任何控制节点")
    # t2i 用不了控制图（没源图）
    try:
        S.build_workflow(S.normalize({"prompt": "x", "seed": 42, "control": "canny"}))
        check(False, "t2i + control 应被拒")
    except ValueError:
        check(True, "t2i 下用控制图被明确拒绝（需要源图）")
    # 未知类型要拒
    try:
        S.build_workflow(S.normalize({"prompt": "x", "seed": 42, "mode": "i2i",
                                      "image": "1.png", "refs": ["1.png"],
                                      "control": "pose"}))
        check(False, "未支持的控制类型应被拒")
    except ValueError:
        check(True, "未支持的 control=pose 被明确拒绝")

    print()
    print("=" * 74)
    print("三、串档拦截")
    print("=" * 74)
    # multi 仍不支持（Anima 只吃单张源图）
    try:
        S.build_workflow(S.normalize({"prompt": "x", "seed": 42,
                                      "mode": "multi", "refs": ["1.png"]}))
        check(False, "anima 档下 multi 应被拒")
    except ValueError:
        check(True, "anima 档下 multi 被明确拒绝（单张源图模型）")
    # i2i 不传源图要拒
    try:
        S.build_workflow(S.normalize({"prompt": "x", "seed": 42, "mode": "i2i"}))
        check(False, "anima i2i 无源图应被拒")
    except ValueError:
        check(True, "anima i2i 缺源图被明确拒绝")
    try:
        S.normalize({"prompt": "x", "seed": 42, "lora": "bj2_450"})
        check(False, "anima 档挂 Qwen LoRA 应被拒")
    except ValueError:
        check(True, "anima 档挂 bj2_450 被明确拒绝")

    print()
    print("=" * 74)
    print("四、档位切换干净（切回 cloud 要完全恢复）")
    print("=" * 74)
    S.apply_profile("cloud")
    check(S.SAMPLER_CHOICES == ("euler",), "采样器白名单回到 ('euler',)")
    check(S.DEFAULTS["steps"] == 10 and S.DEFAULTS["cfg"] == 1.0, "steps/cfg 回到 10 / 1.0")
    check(S.DEFAULTS["unet"].startswith("qwen_image_2.1"), "unet 回到 Qwen")
    w = wf(mode="i2i", image="1.png")
    check(w["3"]["class_type"] == "KSampler", "cloud 的 i2i 仍然可用")
    check("QwenImage21Cache" in types(w), "cloud 仍带 Cache 节点")

    print()
    print("=" * 74)
    print("全部通过 ✅" if not FAILS else "有 %d 处不符 ❌" % len(FAILS))
    for f in FAILS:
        print("   - " + f)
    print("=" * 74)
    return 1 if FAILS else 0


if __name__ == "__main__":
    raise SystemExit(main())
