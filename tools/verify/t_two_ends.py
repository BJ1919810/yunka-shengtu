#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""两端一致性校验：APK 的 Comfy.LORA_PRESETS 必须与 server.py 的 LORA_PRESETS 逐字段相同。

两端无共享代码（App 自己 buildWorkflow 直连 ComfyUI，不走 server.py），
只能靠这张表手工同步 —— 这个脚本就是防止"改了一边忘了另一边"。

★ 在 Node 里真跑一遍 api.js（它不依赖 DOM），读出 Comfy.LORA_PRESETS，
  再和 server.py 的 LORA_PRESETS 对比。
"""
import json
import subprocess
import sys

import pathlib as _pl
_here = _pl.Path(__file__).resolve()
for _r in (_here.parent, *_here.parents):          # 向上找仓库根（含 paths.py）
    if (_r / "paths.py").exists():
        sys.path.insert(0, str(_r))
        break
import paths  # noqa: E402  （路径唯一权威）

WEB = str(paths.COMFY_WEB)
APK = str(paths.APK_ASSETS)
NODE = str(paths.NODE_EXE)

sys.path.insert(0, WEB)
import server as S   # noqa: E402

# 用 Node 加载 api.js（IIFE 挂到 root.Comfy；模拟一个最小 root）
JS = r"""
var fs = require('fs');
var src = fs.readFileSync(process.argv[1], 'utf8');
var root = {};
// api.js 是 (function(root){...})(root) —— 直接 eval 时给个 root
eval(src.replace('(function (root)', '(function (root)'));
// 上面 replace 只是形式；真正做法是包一层调用
"""
# 更稳的做法：把 api.js 包成 CommonJS 模块再 require
WRAP = """
const fs = require('fs');
const path = process.argv[2];
let src = fs.readFileSync(path, 'utf8');
src += '\\nmodule.exports = (typeof Comfy !== "undefined") ? Comfy : undefined;';
const Module = require('module');
const m = new Module(path, null);
m.filename = path;
m.paths = Module._nodeModulePaths(require('path').dirname(path));
m._compile(src, path);
const C = m.exports;
if (!C) { console.error('NO_COMFY'); process.exit(2); }
console.log(JSON.stringify(C.LORA_PRESETS));
"""

tmp = "_apk_lora_presets.js"
with open(tmp, "w", encoding="utf-8") as f:
    f.write(WRAP)

print("=" * 78)
print("两端 LoRA_PRESETS 一致性校验")
print("=" * 78)

try:
    out = subprocess.run([NODE, tmp, APK + "/api.js"],
                         capture_output=True, text=True, timeout=120)
    if out.returncode != 0:
        print("Node 读取 api.js 失败：")
        print(out.stdout[-800:])
        print(out.stderr[-1200:])
        sys.exit(1)
    apk = json.loads(out.stdout.strip().splitlines()[-1])
except Exception as e:
    print("读取失败：", e)
    sys.exit(1)

ok = True
keys_a = list(apk.keys())
keys_s = list(S.LORA_PRESETS.keys())
print()
print("档位集合：")
if keys_a == keys_s:
    print("   ✓ 一致：{}".format(keys_a))
else:
    ok = False
    print("   ✗ 不一致：APK={} server={}".format(keys_a, keys_s))

print()
print("逐字段对比：")
FIELDS = ("file", "steps", "sigmas", "strength")
for k in keys_s:
    a = apk.get(k)
    s = S.LORA_PRESETS[k]
    if a is None:
        ok = False
        print("   ✗ {:<6} APK 缺这一档".format(k))
        continue
    for f in FIELDS:
        av, sv = a.get(f), s.get(f)
        if av != sv:
            ok = False
            print("   ✗ {:<6} {:<9} APK={!r}  server={!r}".format(k, f, av, sv))
    else:
        print("   ✓ {:<6} 四字段全一致  file={}".format(k, a.get("file")))

import os
os.remove(tmp)
print()
print("=" * 78)
print("两端一致 ✅" if ok else "有差异 ❌（改一处必须改另一处）")
print("=" * 78)
sys.exit(0 if ok else 1)
