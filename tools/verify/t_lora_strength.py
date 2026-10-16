# -*- coding: utf-8 -*-
"""comfy_web「LoRA 强度」入口验证（进程内 + 真 Edge）。

背景（2026-10-16 用户反馈）：「strength 怎么调，没有入口」——以前强度写死在
`LORA_PRESETS[...]["strength"]`，前端只能看不能改。现在前端有 `#loraStrength` 输入框，
payload 里带 `lora_strength`，服务端 `normalize()` 夹到 0~2（不传就用档位自带值），
`build_workflow()` 把它写进节点 9 的 `strength_model`。

要守住的两件事：
  ① 不传 `lora_strength` 时**必须保留档位自带值** —— `bj` 档（bj_style 原始版权重）默认是
     **0.5**，若被前端猜成 1.0，风格会静默浓一倍（这是本条改动的最大风险）。
  ② 非法值/越界值不能把请求带崩：`"abc"`/空串 ⇒ 落回默认；`5` ⇒ 夹到 2.0；`-1` ⇒ 0.0。

用法：python tools/verify/t_lora_strength.py
"""
from __future__ import annotations

import importlib.util
import json
import os
import subprocess
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
CW = str(paths.COMFY_WEB)
PW_PY = str(paths.COMFY_PY)
FAILS: list[str] = []


def check(cond, msg):
    print("  %s %s" % ("✓" if cond else "✗", msg))
    if not cond:
        FAILS.append(msg)


def load_server():
    spec = importlib.util.spec_from_file_location("cwsrv", os.path.join(CW, "server.py"))
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


BROWSER = r'''
import asyncio, json, sys
from playwright.async_api import async_playwright

URL = sys.argv[1]

async def main():
    async with async_playwright() as p:
        b = await p.chromium.launch(channel="msedge")
        pg = await b.new_page()
        errs = []
        pg.on("pageerror", lambda e: errs.append("pageerror:" + str(e)))
        await pg.goto(URL, wait_until="load")
        await pg.wait_for_timeout(600)
        # ① 服务端给了默认强度 ⇒ 预填 + 行可见 + 发得出去
        r1 = await pg.evaluate("""() => {
            LORA_STRENGTH_DEFAULT.jm_style = 1.0;
            document.getElementById('loraSel').value = 'jm_style';
            syncLora();
            return {
              rowShown: document.getElementById('loraStrengthRow').style.display !== 'none',
              input: document.getElementById('loraStrength').value,
              hint: document.getElementById('loraStrengthHint').textContent,
              sent: collect().lora_strength,
            };
        }""")
        # ② 用户手改 ⇒ 按用户的值发
        r2 = await pg.evaluate("""() => {
            document.getElementById('loraStrength').value = '1.4';
            return { sent: collect().lora_strength };
        }""")
        # ③ 选「不使用」⇒ 整行隐藏
        r3 = await pg.evaluate("""() => {
            document.getElementById('loraSel').value = 'none';
            syncLora();
            return { rowShown: document.getElementById('loraStrengthRow').style.display !== 'none' };
        }""")
        # ④ 拿不到默认值（老服务端）⇒ **不许发这个字段**（否则会把 0.5 的档位抬成 1.0）
        r4 = await pg.evaluate("""() => {
            Object.keys(LORA_STRENGTH_DEFAULT).forEach(k => delete LORA_STRENGTH_DEFAULT[k]);
            document.getElementById('loraSel').value = 'bj';
            syncLora();
            const c = collect();
            return { hasField: Object.prototype.hasOwnProperty.call(c, 'lora_strength'),
                     hint: document.getElementById('loraStrengthHint').textContent };
        }""")
        print(json.dumps({"errors": errs, "r1": r1, "r2": r2, "r3": r3, "r4": r4}, ensure_ascii=False))
        await b.close()

asyncio.run(main())
'''


def main() -> int:
    print("=" * 76)
    print("comfy_web · LoRA 强度入口（进程内 + 真 Edge）")
    print("=" * 76)

    # ---------- A. 服务端：normalize + build_workflow ----------
    print("\nA. 服务端（normalize → build_workflow 的节点 9）")
    S = load_server()
    check("jm_style" in S.LORA_PRESETS, "服务端认得 jm_style 档")
    # ⚠️ 别把"某档应该等于 0.5"写死 —— 预设指向的都是 `_comfy` 版（alpha 已内置）⇒ 全是 1.0。
    #    这里只记录事实；真正的判据是下面那条「不传 ⇒ 必须等于该档自带值」的通用回归。
    print("     各档自带 strength：%s" % {k: v.get("strength") for k, v in S.LORA_PRESETS.items()})

    def run(profile, lora, strength):
        S.PROFILE = profile
        raw = dict(S.DEFAULTS)
        raw.update({"lora": lora, "prompt": "jm_style, 1girl", "seed": 12345})
        if strength is not ...:
            raw["lora_strength"] = strength
        p = S.normalize(raw)
        wf = S.build_workflow(p)
        node = wf.get("9")
        return (p.get("lora_strength"), (node or {}).get("inputs", {}).get("strength_model"),
                (node or {}).get("inputs", {}).get("lora_name"))

    ls, sm, fn = run("anima", "jm_style", ...)
    check(sm == 1.0 and fn == "jm_style_comfy.safetensors",
          "不传 ⇒ 用档位自带值：node9 strength_model=%r file=%r" % (sm, fn))
    ls, sm, fn = run("anima", "jm_style", 1.4)
    check(sm == 1.4, "传 1.4 ⇒ node9 strength_model=%r" % sm)
    ls, sm, fn = run("anima", "jm_style", 5)
    check(sm == 2.0, "传 5 ⇒ 夹到 2.0（实测 %r）" % sm)
    ls, sm, fn = run("anima", "jm_style", -1)
    check(sm == 0.0, "传 -1 ⇒ 夹到 0.0（实测 %r）" % sm)
    for bad in ("abc", ""):
        ls, sm, fn = run("anima", "jm_style", bad)
        check(sm == 1.0, "非法值 %r ⇒ 落回档位默认 1.0（实测 %r）" % (bad, sm))
    # ★ 最关键的通用回归：**每个档位**不传强度时，节点 9 都必须等于该档位自带值。
    #   同时这条会在两条链路各跑一遍（arch=='anima' ⇒ plain 分支；否则 Qwen 分支）——
    #   2026-10-16 就是它抓出「plain 分支从来没挂过 LoRA」这个真 bug。
    bad = []
    n_file = 0
    for _name, _pr in S.LORA_PRESETS.items():
        if not _pr.get("file"):
            continue
        n_file += 1
        _prof = "anima" if _pr.get("arch") == "anima" else "cloud"
        try:
            _, _sm, _fn = run(_prof, _name, ...)
        except Exception as _e:
            bad.append("%s(%s) 抛错 %s" % (_name, _prof, _e))
            continue
        if _sm is None or abs(float(_sm) - float(_pr["strength"])) > 1e-6:
            bad.append("%s(%s) 期望 %r 实得 %r（file=%r）" % (_name, _prof, _pr["strength"], _sm, _fn))
    check(n_file >= 17, "参与回归的档位 %d 个（18 档里 none 没有 file ⇒ 17）" % n_file)
    check(not bad, "★ 全部 %d 档：不传 lora_strength ⇒ 节点 9 = 该档自带值%s"
          % (n_file, "" if not bad else "；异常：%s" % bad[:4]))

    # ★ 光"有 node9"不算挂上 —— 采样器必须真的吃它，否则只是个孤儿节点（这正是本轮的 bug：
    #   plain 分支连 node9 都没有；修好后还要防"加了节点但没接线"）。
    S.PROFILE = "anima"
    _wf = S.build_workflow(S.normalize(dict(S.DEFAULTS, **{
        "lora": "jm_style", "prompt": "jm_style, 1girl", "seed": 1})))
    check(_wf.get("9", {}).get("inputs", {}).get("model") == ["37", 0],
          "plain 链路：node9 吃 UNETLoader(37)（实测 %r）"
          % (_wf.get("9", {}).get("inputs", {}).get("model"),))
    check(_wf.get("3", {}).get("inputs", {}).get("model") == ["9", 0],
          "★ plain 链路：KSampler(3) 真的吃 node9（实测 %r）—— 这才叫「挂上了」"
          % (_wf.get("3", {}).get("inputs", {}).get("model"),))
    check(_wf["9"]["inputs"]["lora_name"] == "jm_style_comfy.safetensors",
          "plain 链路：lora_name = %r" % _wf["9"]["inputs"]["lora_name"])
    # 不挂 LoRA 时 KSampler 必须仍直连 UNETLoader（别把老链路改坏）
    _wf2 = S.build_workflow(S.normalize(dict(S.DEFAULTS, **{
        "lora": "none", "prompt": "1girl", "seed": 1})))
    check("9" not in _wf2 and _wf2.get("3", {}).get("inputs", {}).get("model") == ["37", 0],
          "lora=none ⇒ 无 node9、KSampler 直连 37（实测 %r）"
          % (_wf2.get("3", {}).get("inputs", {}).get("model"),))
    # 选「不使用」时不该有节点 9
    ls, sm, fn = run("anima", "none", ...)
    check(sm is None, "lora=none ⇒ 不建节点 9（实测 strength_model=%r）" % sm)

    # ---------- B. 前端：真 Edge ----------
    print("\nB. 前端（file:// 真 Edge，注入默认强度映射后驱动 DOM）")
    bp = os.path.join(os.environ.get("TEMP", "."), "lora_strength_browser.py")
    with open(bp, "w", encoding="utf-8") as f:
        f.write(BROWSER)
    r = subprocess.run([PW_PY, bp, "file:///" + os.path.join(CW, "index.html").replace("\\", "/")],
                       capture_output=True, text=True, encoding="utf-8", errors="replace", timeout=300)
    lines = [l for l in (r.stdout or "").strip().splitlines() if l.strip()]
    if not lines:
        check(False, "浏览器没输出：%s" % (r.stderr or "")[-400:])
    else:
        d = json.loads(lines[-1])
        check(not d["errors"], "页面无 JS 报错（%s）" % d["errors"])
        check(d["r1"]["rowShown"] and d["r1"]["input"] in ("1.0", "1"),
              "有默认值时：行可见、输入框预填 %r" % d["r1"]["input"])
        check("档位默认" in d["r1"]["hint"], "提示写着档位默认：%r" % d["r1"]["hint"])
        check(d["r1"]["sent"] == 1.0, "payload 发出 lora_strength=%r" % d["r1"]["sent"])
        check(d["r2"]["sent"] == 1.4, "用户改成 1.4 ⇒ payload=%r" % d["r2"]["sent"])
        check(not d["r3"]["rowShown"], "选「不使用」⇒ 整行隐藏")
        check(not d["r4"]["hasField"],
              "★ 拿不到默认值时**不发** lora_strength（否则会把 bj 的 0.5 抬成 1.0）")
        check("档位默认" in d["r4"]["hint"] and "0.5" not in d["r4"]["hint"],
              "无默认值时提示不编造数值：%r" % d["r4"]["hint"])

    print("\n" + "=" * 76)
    print("全部通过 ✅" if not FAILS else "有问题 ❌（%d）" % len(FAILS))
    for f in FAILS:
        print("   ✗ " + f)
    print("=" * 76)
    return 0 if not FAILS else 1


if __name__ == "__main__":
    sys.exit(main())
