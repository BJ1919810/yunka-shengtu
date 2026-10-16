#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""在 Node 里真跑 APK 的 Comfy.buildWorkflow()，校验四档接线（与 server.py 同判据）。"""
import json
import os
import subprocess
import sys

import pathlib as _pl
_here = _pl.Path(__file__).resolve()
for _r in (_here.parent, *_here.parents):          # 向上找仓库根（含 paths.py）
    if (_r / "paths.py").exists():
        sys.path.insert(0, str(_r))
        break
import paths  # noqa: E402  （路径唯一权威）

APK = str(paths.APK_ASSETS / "api.js")
NODE = str(paths.NODE_EXE)

JS = r"""
const fs = require('fs'), path = require('path'), Module = require('module');
const P = process.argv[2];
let src = fs.readFileSync(P, 'utf8');
src += '\nmodule.exports = (typeof Comfy !== "undefined") ? Comfy : undefined;';
const m = new Module(P, null);
m.filename = P; m.paths = Module._nodeModulePaths(path.dirname(P));
m._compile(src, P);
const C = m.exports;
const PROMPT = 'bj_style, 一位少女站在教室里';
function run(lora, steps) {
  const raw = { prompt: PROMPT, mode: 't2i', lora: lora, steps: steps,
                width: 1024, height: 1024, cfg: 1.0, sampler: 'euler',
                scheduler: 'simple', seed: 42, followRef: false,
                unet: 'qwen_image_2.1_int8_convrot.safetensors',
                clip: 'qwen3vl_8b_w4a8.safetensors',
                vae: 'qwen_image_2.1_vae_bf16.safetensors', resolution: 1024 };
  const o = C.normalize(raw);
  const wf = C.buildWorkflow(o);
  const has = !!wf['9'];
  return { lora: o.lora, steps: o.steps, has_lora: has,
           sampler: wf['6'].class_type,
           lora_name: has ? wf['9'].inputs.lora_name : null,
           strength: has ? wf['9'].inputs.strength_model : null,
           before_cache: has ? (wf['66'].inputs.model[0] === '9') : null,
           has_sigma: !!wf['475'], sig_loraSteps: o.loraSteps };
}
const out = [];
for (const c of [['none',20],['acc8',20],['acc5',20],['bj',20]]) out.push(run(c[0], c[1]));
// 旧字段兼容
const legacy = [];
for (const c of [[true,5],[true,8],[false,null]]) {
  const raw = { prompt: PROMPT, mode: 't2i', lora: c[0], loraSteps: c[1],
                width: 1024, height: 1024, steps: 20, cfg: 1.0, sampler: 'euler',
                scheduler: 'simple', seed: 42, followRef: false,
                unet: 'qwen_image_2.1_int8_convrot.safetensors',
                clip: 'qwen3vl_8b_w4a8.safetensors',
                vae: 'qwen_image_2.1_vae_bf16.safetensors', resolution: 1024 };
  legacy.push(C.normalize(raw).lora);
}
console.log(JSON.stringify({ cases: out, legacy: legacy }));
"""

tmp = "_apk_wf.js"
with open(tmp, "w", encoding="utf-8") as f:
    f.write(JS)

r = subprocess.run([NODE, tmp, APK], capture_output=True, text=True, timeout=180)
os.remove(tmp)
if r.returncode != 0:
    print("Node 执行失败：")
    print(r.stdout[-600:]); print(r.stderr[-1500:])
    sys.exit(1)

d = json.loads(r.stdout.strip().splitlines()[-1])

EXPECT = {
    "none": dict(has_lora=False, steps=20, sampler="KSampler", has_sigma=False),
    "acc8": dict(has_lora=True, steps=8, sampler="SamplerCustom", has_sigma=True,
                 lora_name="p_qwen_image_2.1_8step_v0.1_comfyui.safetensors", strength=1.0),
    "acc5": dict(has_lora=True, steps=5, sampler="SamplerCustom", has_sigma=True,
                 lora_name="p_qwen_image_2.1_5step_v0.1_comfyui.safetensors", strength=1.0),
    "bj":   dict(has_lora=True, steps=20, sampler="KSampler", has_sigma=False,
                 lora_name="bj_style_500_comfy.safetensors", strength=1.0),
}
print("=" * 76)
print("APK 端 Comfy.buildWorkflow() 四档接线校验（Node 真跑）")
print("=" * 76)
ok = True
for c in d["cases"]:
    e = EXPECT[c["lora"]]
    bad = []
    for k, v in e.items():
        if c.get(k) != v:
            bad.append("{}: 期望{!r} 实际{!r}".format(k, v, c.get(k)))
    if c["has_lora"] and c.get("before_cache") is not True:
        bad.append("LoRA 没插在 Cache 之前！")
    if bad:
        ok = False
        print("✗ {:<6} {}".format(c["lora"], "; ".join(bad)))
    else:
        print("✓ {:<6} steps={:<3} {} {} {}".format(
            c["lora"], c["steps"], c["sampler"],
            (c["lora_name"] or "-"), "sigma✓" if c["has_sigma"] else ""))

print()
print("旧字段兼容：{} → {}".format([[True, 5], [True, 8], [False, None]],
                              d["legacy"]))
want = ["acc5", "acc8", "none"]
if d["legacy"] != want:
    ok = False
    print("✗ 期望 {}".format(want))
else:
    print("✓ 一致")
print()
print("=" * 76)
print("APK 端通过 ✅" if ok else "APK 端有问题 ❌")
print("=" * 76)
sys.exit(0 if ok else 1)
