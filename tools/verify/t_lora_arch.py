# -*- coding: utf-8 -*-
"""comfy_web 前端 —— 「LoRA 按档位过滤」真浏览器验证（**不需要起任何服务**）。

背景（2026-10-16 用户问「lora 下拉表里还是空的」）：
    anima 档原来把 LoRA 白名单写成 `("none",)`（因为 Anima 挂不了 Qwen 系 LoRA），
    前端 applyProfile() 会**隐藏**不在白名单里的选项 ⇒ 看起来"下拉是空的"。
    现在给 anima 档加了自训的 `jm_style`（Anima 架构），同时必须保证：
      ① anima 档能看到 jm_style
      ② cloud/local 档**看不到** jm_style（拿 Anima 的 LoRA 去挂 Qwen 会直接崩）
      ③ 在 anima 档选了 jm_style 后切回 cloud ⇒ **必须改选**，不能把这个值带过去

为什么走 file://：本环境无头 Edge 连不上 127.0.0.1（系统代理拦），file:// 不走网络，
    而 `applyProfile()` 是页面里的全局函数 ⇒ 直接把合成的档位对象喂给它就能验状态机。

用法：python tools/verify/t_lora_arch.py
"""
from __future__ import annotations

import io
import json
import os
import shutil
import subprocess
import sys
import tempfile

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
COMVY = str(paths.COMFY_WEB)
PW_PY = str(paths.COMFY_PY)

PROBE = r'''
import asyncio, json, sys
from playwright.async_api import async_playwright

URL = sys.argv[1]

JS = r"""
() => {
  const $ = id => document.getElementById(id);
  // 合成两个档位对象：只带 applyProfile 真正读的字段
  const mk = (name, loras, graph) => ({
    name: name, label: name, graph: graph, unet: 'x', clip: 'x', vae: 'x',
    samplers: ['euler'], scheduler: 'simple', sampler: 'euler',
    schedulers: ['simple'], steps: 30, cfg: 4.0, negative: 'n',
    modes: ['t2i', 'i2i'], loras: loras, controls: null, control: null,
    controlModel: null, controlStrength: 1.0, lockParams: false,
    modelPresets: null, modelPresetDefault: null,
  });
  const vis = () => [...$('loraSel').options].filter(o => !o.hidden && !o.disabled).map(o => o.value);
  const out = {};

  // ① anima 档
  applyProfile(mk('anima', ['none', 'jm_style'], 'plain'), { model_patches: [] });
  out.anima_visible = vis();
  out.anima_selected = $('loraSel').value;

  // ③ 在 anima 档选中 jm_style，再切回 cloud —— 值必须被改掉
  $('loraSel').value = 'jm_style';
  applyProfile(mk('cloud', null, 'qwen'), { model_patches: [] });
  out.after_switch_selected = $('loraSel').value;
  out.cloud_visible = vis();
  out.cloud_has_jm = vis().indexOf('jm_style') >= 0;

  // ② local 档（也是 None 白名单）
  applyProfile(mk('local', null, 'qwen'), { model_patches: [] });
  out.local_has_jm = vis().indexOf('jm_style') >= 0;

  return out;
}
"""

async def main():
    async with async_playwright() as p:
        b = await p.chromium.launch(channel="msedge")
        pg = await b.new_page()
        errs = []
        pg.on("pageerror", lambda e: errs.append("pageerror:" + str(e)))
        await pg.goto(URL, wait_until="load")
        await pg.wait_for_timeout(600)
        r = await pg.evaluate(JS)
        print(json.dumps({"errors": errs, "state": r}, ensure_ascii=False))
        await b.close()

asyncio.run(main())
'''


def main() -> int:
    html = io.open(os.path.join(COMVY, "index.html"), encoding="utf-8").read()
    tmpdir = tempfile.mkdtemp(prefix="cw_lora_arch_")
    page = os.path.join(tmpdir, "index.html")
    io.open(page, "w", encoding="utf-8").write(html)
    probe = os.path.join(tmpdir, "probe.py")
    io.open(probe, "w", encoding="utf-8").write(PROBE)
    url = "file:///" + page.replace("\\", "/")

    print("=" * 74)
    print("comfy_web · LoRA 按档位过滤（file:// 真 Edge，无服务）")
    print("=" * 74)
    try:
        r = subprocess.run([PW_PY, probe, url], capture_output=True, text=True,
                           encoding="utf-8", errors="replace", timeout=180)
        lines = [l for l in (r.stdout or "").strip().splitlines() if l.strip()]
        if not lines:
            print("[x] 浏览器无输出：", (r.stderr or "")[-600:])
            return 1
        d = json.loads(lines[-1])
    finally:
        shutil.rmtree(tmpdir, ignore_errors=True)

    st, errs = d["state"], d["errors"]
    fails = []
    if errs:
        for e in errs:
            print("  ✗ " + e[:180])
        fails.append("页面有 JS 报错")

    print("  ① anima 档可见: %s" % st["anima_visible"])
    ok1 = st["anima_visible"] == ["none", "jm_style"]
    print("     %s 应该是 ['none', 'jm_style']" % ("✓" if ok1 else "✗"))
    if not ok1:
        fails.append("anima 档没正确显示 jm_style")

    print("  ② cloud 档可见: %s（共 %d 项）" % (st["cloud_visible"][:4] + ["…"], len(st["cloud_visible"])))
    ok2 = (not st["cloud_has_jm"]) and len(st["cloud_visible"]) >= 3
    print("     %s 不能出现 jm_style（Anima 的 LoRA 挂 Qwen 会崩）" % ("✓" if ok2 else "✗"))
    if not ok2:
        fails.append("cloud 档混进了 jm_style")
    ok3 = not st["local_has_jm"]
    print("  ③ local 档不含 jm_style: %s" % ("✓" if ok3 else "✗"))
    if not ok3:
        fails.append("local 档混进了 jm_style")

    print("  ④ anima 选 jm_style 后切回 cloud，当前值 = %r" % st["after_switch_selected"])
    ok4 = st["after_switch_selected"] != "jm_style"
    print("     %s 必须被改选（否则会把 Anima LoRA 带到 Qwen 档）" % ("✓" if ok4 else "✗"))
    if not ok4:
        fails.append("切换档位后把 jm_style 带过去了")

    print()
    print("=" * 74)
    print("全部通过 ✅" if not fails else "有问题 ❌（%d）" % len(fails))
    for f in fails:
        print("   ✗ " + f)
    print("=" * 74)
    return 0 if not fails else 1


if __name__ == "__main__":
    sys.exit(main())
