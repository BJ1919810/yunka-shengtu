#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""离线验证：LoRA 单选四种档位的工作流是否正确（不出图、不需要显卡）。

判据：
  1. none ⇒ 无 LoraLoaderModelOnly、走 KSampler、步数不被改
  2. acc8 / acc5 ⇒ 有 LoraLoader 且插在 Cache 之前、走 SamplerCustom、
                   步数被强制成档位值、sigma 表对得上
  3. bj ⇒ 有 LoraLoader（strength 1.0）、**仍走 KSampler**（不能套蒸馏链路）、
          步数**不被**改
  4. 旧字段兼容：lora=true + loraSteps=5 应迁移成 acc5
"""
import sys

import pathlib as _pl
_here = _pl.Path(__file__).resolve()
for _r in (_here.parent, *_here.parents):          # 向上找仓库根（含 paths.py）
    if (_r / "paths.py").exists():
        sys.path.insert(0, str(_r))
        break
import paths  # noqa: E402  （路径唯一权威）

sys.path.insert(0, str(paths.COMFY_WEB))
import server as S

P = "bj_style, 一位少女站在教室里"


def probe(name, extra, expect):
    o = S.normalize(dict({"prompt": P}, **extra))
    wf = S.build_workflow(o)
    has_lora = "9" in wf
    sampler = wf["3"]["class_type"]
    lora_name = wf["9"]["inputs"]["lora_name"] if has_lora else None
    strength = wf["9"]["inputs"]["strength_model"] if has_lora else None
    # Cache 节点的 model 输入是不是指向 LoRA（= LoRA 在 Cache 之前）
    cache_src = wf["66"]["inputs"].get("model")
    before_cache = (not has_lora) or (cache_src == ["9", 0])
    steps = o["steps"]
    sig = wf.get("475", {}).get("inputs", {}).get("sigmas")

    ok = True
    def chk(label, cond, got):
        nonlocal ok
        if not cond:
            ok = False
        print("   {} {:34s} {}".format("✓" if cond else "✗", label, got))

    print("[{}]  lora={!r}".format(name, o["lora"]))
    chk("挂 LoRA 节点", has_lora == expect["has_lora"], has_lora)
    chk("采样链路", sampler == expect["sampler"], sampler)
    chk("插在 Cache 之前", before_cache, cache_src)
    if expect.get("file"):
        chk("权重文件", lora_name == expect["file"], lora_name)
        chk("strength", strength == expect.get("strength"), strength)
    chk("步数", steps == expect["steps"], "{}（题面填 20）".format(steps))
    chk("sigma 表", (sig is not None) == expect["has_sigma"],
        (sig[:34] + "…") if sig else None)
    print()
    return ok


allok = True
print("=" * 74)
print("LoRA 单选 —— 四档工作流离线校验（题面都显式填 steps=20）")
print("=" * 74)
allok &= probe("none", {"lora": "none", "steps": 20},
               dict(has_lora=False, sampler="KSampler", steps=20, has_sigma=False, file=None))
allok &= probe("acc8", {"lora": "acc8", "steps": 20},
               dict(has_lora=True, sampler="SamplerCustom", steps=8, has_sigma=True,
                    file=S.LORA_FILES[8], strength=1.0))
allok &= probe("acc5", {"lora": "acc5", "steps": 20},
               dict(has_lora=True, sampler="SamplerCustom", steps=5, has_sigma=True,
                    file=S.LORA_FILES[5], strength=1.0))
allok &= probe("bj", {"lora": "bj", "steps": 20},
               dict(has_lora=True, sampler="KSampler", steps=20, has_sigma=False,
                    file=S.STYLE_LORA_FILE, strength=1.0))

print("=" * 74)
print("旧字段兼容（老页面 / 老版 App 直接 POST）")
print("=" * 74)
for legacy, want in (({"lora": True, "loraSteps": 5}, "acc5"),
                     ({"lora": True, "loraSteps": 8}, "acc8"),
                     ({"lora": False}, "none")):
    got = S.normalize(dict({"prompt": P}, **legacy))["lora"]
    good = got == want
    allok &= good
    print("   {} {} → {!r}（期望 {!r}）".format("✓" if good else "✗", legacy, got, want))

print()
print("坏档位应被拒：")
for bad in ("xyz", ""):
    try:
        S.normalize({"prompt": P, "lora": bad})
        print("   ✗ 未拒绝 {!r}".format(bad))
        allok = False
    except ValueError as e:
        print("   ✓ 已拒绝 {!r}：{}".format(bad, str(e)[:58]))

print()
print("=" * 74)
print("全部通过 ✅" if allok else "有失败 ❌")
print("=" * 74)
sys.exit(0 if allok else 1)
