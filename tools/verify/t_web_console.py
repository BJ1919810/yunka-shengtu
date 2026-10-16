"""真 Edge 验证 comfy_web 前端：控制台报错 + 事件绑定 + 档位联动。

背景：2026-10-15 我把 PROF_DENOISE 等三个 let 声明放在 setMode 首次调用**之后**，
      触发 TDZ（暂时性死区）⇒ 页面在 setMode("t2i") 那行抛 ReferenceError 中断
      ⇒ 后面所有事件绑定（源图点击/Ctrl+V 粘贴/拖拽）全没执行
      ⇒ 用户看到「点图片没反应、粘贴失效」。
      这个脚本就是防回归：只要「事件没绑上」或「有控制台报错」就失败。

用法：python tools/verify/t_web_console.py [端口]
⛔ 必须先起 ComfyUI（否则 /api/options 拿不到模型列表，前端会停在"加载中"）。
"""
import json
import subprocess
import sys
import tempfile

import pathlib as _pl
_here = _pl.Path(__file__).resolve()
for _r in (_here.parent, *_here.parents):          # 向上找仓库根（含 paths.py）
    if (_r / "paths.py").exists():
        sys.path.insert(0, str(_r))
        break
import paths  # noqa: E402  （路径唯一权威）

PW_PY = str(paths.COMFY_PY)
PORT = sys.argv[1] if len(sys.argv) > 1 else "7877"

SCRIPT = r'''
import asyncio, json, sys
from playwright.async_api import async_playwright

PORT = sys.argv[1]

async def main():
    async with async_playwright() as p:
        b = await p.chromium.launch(channel="msedge")
        pg = await b.new_page()
        errs = []
        pg.on("console", lambda m: errs.append("console:" + m.text) if m.type == "error" else None)
        pg.on("pageerror", lambda e: errs.append("pageerror:" + str(e)))
        await pg.goto(f"http://127.0.0.1:{PORT}/", wait_until="networkidle")
        await pg.wait_for_timeout(2500)   # 等 loadOptions 跑完

        r = await pg.evaluate("""() => {
            const $ = id => document.getElementById(id);
            const modes = [...document.querySelectorAll('[data-mode]')]
                .map(b => [b.dataset.mode, b.disabled]);
            return {
                // ★ 关键：事件绑定在不在（没绑上就是 null）
                srcDropOnclick: typeof $('srcDrop').onclick,
                srcFileExists: !!$('srcFile'),
                // denoise 框
                denoiseExists: !!$('denoise'),
                denoiseWrapDisplay: $('denoiseWrap') ? $('denoiseWrap').style.display : null,
                denoiseVal: $('denoise') ? $('denoise').value : null,
                // 成对存在：源图/参考图 上传入口
                refFileExists: !!$('refFile'),
                // 档位
                modes: modes,
                profile: (window.__prof || null),
                env: (document.querySelector('header .sub') || {}).textContent,
                loraSelCount: $('loraSel') ? $('loraSel').options.length : -1,
                // 可见的 LoRA 档数（anima 档应该只剩「不使用」）
                loraVisible: $('loraSel')
                    ? [...$('loraSel').options].filter(o => !o.hidden && !o.disabled).map(o => o.value)
                    : [],
                loraValue: $('loraSel') ? $('loraSel').value : null,
                stepsVal: $('steps') ? $('steps').value : null,
                cfgVal: $('cfg') ? $('cfg').value : null,
                samplerVal: $('sampler') ? $('sampler').value : null,
                // ★ 控制图（LLLite）—— 只有 anima 档有，且只在 i2i 显示
                controlExists: !!$('controlSel'),
                controlOptions: $('controlSel') ? [...$('controlSel').options].map(o => o.value) : [],
                controlFieldDisplay: $('controlField') ? $('controlField').style.display : null,
                controlModels: $('controlModelSel') ? $('controlModelSel').options.length : -1,
                // ★ Anima 英文词表入口（只在 plain 图/anima 档显示）
                tagsLinkDisplay: $('tagsLink') ? $('tagsLink').style.display : null,
                // ★ 锁定范围：采样器/调度器恒 disabled；steps/cfg **应可编辑**
                stepsDisabled: $('steps') ? $('steps').disabled : null,
                cfgDisabled: $('cfg') ? $('cfg').disabled : null,
                samplerDisabled: $('sampler') ? $('sampler').disabled : null,
                schedulerDisabled: $('scheduler') ? $('scheduler').disabled : null,
                modelHint: ($('modelHint') || {}).textContent || '',
                unetLabels: $('unet') ? [...$('unet').options].map(o => o.textContent) : [],
                unetValues: $('unet') ? [...$('unet').options].map(o => o.value) : [],
                // ★ 只看「可见可选」的（anima 档应只剩它自己认得的模型）
                unetUsable: $('unet')
                    ? [...$('unet').options].filter(o => !o.hidden && !o.disabled)
                        .map(o => [o.value, o.textContent])
                    : [],
                // ★ 档位相关提示文案（防"写死 2.1"回归）
                hints: {
                    steps: ($('stepsHint') || {}).textContent || '',
                    cfg: ($('cfgHint') || {}).textContent || '',
                    sampler: ($('samplerHint') || {}).textContent || '',
                    alpha: ($('alphaHint') || {}).textContent || '',
                    lora: ($('loraHint') || {}).textContent || '',
                },
            };
        }""")
        print(json.dumps({"errors": errs, "state": r}, ensure_ascii=False))
        await b.close()

asyncio.run(main())
'''

with tempfile.NamedTemporaryFile("w", suffix=".py", delete=False, encoding="utf-8") as f:
    f.write(SCRIPT)
    tmp = f.name

r = subprocess.run([PW_PY, tmp, PORT], capture_output=True, text=True,
                   encoding="utf-8", errors="replace", timeout=300)
import os
os.unlink(tmp)

out = (r.stdout or "").strip().splitlines()
if not out:
    print("[x] 浏览器没输出：", (r.stderr or "")[-600:]); sys.exit(1)
d = json.loads(out[-1])
st, errs = d["state"], d["errors"]


def is_anima_pre(s):
    return (s.get("env") or "").startswith("Anima")

print("=" * 74)
print("一、控制台报错")
print("=" * 74)
if errs:
    for e in errs:
        print("  ✗ " + e[:200])
else:
    print("  ✓ 无报错")

print()
print("=" * 74)
print("二、事件绑定（★ 这是「点击/粘贴失效」的判据）")
print("=" * 74)
ok = True
for k, want in [("srcDropOnclick", "function"), ("srcFileExists", True),
                ("refFileExists", True), ("denoiseExists", True)]:
    v = st.get(k)
    good = (v == want)
    ok &= good
    print("  %s %-16s = %r" % ("✓" if good else "✗", k, v))

print()
print("=" * 74)
print("三、档位联动")
print("=" * 74)
print("  页头     : %s" % st.get("env"))
print("  模式按钮 : %s" % st.get("modes"))
print("  steps/cfg/sampler = %s / %s / %s"
      % (st.get("stepsVal"), st.get("cfgVal"), st.get("samplerVal")))
print("  denoise  : 值=%s  显示=%r" % (st.get("denoiseVal"), st.get("denoiseWrapDisplay")))
print("  LoRA     : 可见%s  当前值=%s" % (st.get("loraVisible"), st.get("loraValue")))
print("  控制图   : 选项%s  控制模型 %s 个  显示=%r"
      % (st.get("controlOptions"), st.get("controlModels"), st.get("controlFieldDisplay")))
print("  词表入口 : 显示=%r （anima 档应可见，Qwen 档应 hidden）" % st.get("tagsLinkDisplay"))
print("  锁定范围 : 采样器=%s 调度器=%s （应恒 True） ｜ 步数=%s CFG=%s"
      % (st.get("samplerDisabled"), st.get("schedulerDisabled"),
         st.get("stepsDisabled"), st.get("cfgDisabled")))
# ★ 采样器/调度器：**所有档位**都锁（它们就是只读显示）
if not (st.get("samplerDisabled") and st.get("schedulerDisabled")):
    print("  ✗ 采样器/调度器应该锁死")
    ok = False
# ★ CFG：所有档位都应可手改
if st.get("cfgDisabled"):
    print("  ✗ CFG 不该被锁")
    ok = False
# ★ 步数：**只有"加速 LoRA 档"才该锁**（acc8/acc5 走 ManualSigmas，步数由 sigma 表定，
#   手填那个框根本不参与计算 ⇒ 锁上是正确的、防误导）。
#   其余情况（anima 全档、cloud 的 none/bj/bj2_*）都必须可改。
lv = st.get("loraValue") or ""
accel = lv.startswith("acc")
if accel:
    print("         （当前 LoRA=%s 是加速档 ⇒ 步数锁死属预期，它走 ManualSigmas）" % lv)
    if not st.get("stepsDisabled"):
        print("  ✗ 加速档下步数应该锁死")
        ok = False
else:
    if st.get("stepsDisabled"):
        print("  ✗ 非加速档下步数不该被锁（用户明确要求只锁采样器+调度器）")
        ok = False
if is_anima_pre(st):
    # ★ LoRA 档位过滤（2026-10-16）：anima 档要能看到**自己架构**的 jm_style；
    #   而 Qwen 档（cloud/local）**绝不能**看到它 —— 拿 Anima 的 LoRA 去挂 Qwen 会直接崩。
    #   （无服务版同判据见 tools/verify/t_lora_arch.py）
    lv_list = st.get("loraVisible") or []
    if lv_list != ["none", "jm_style"]:
        print("  ✗ anima 档的 LoRA 可见项应为 ['none','jm_style']，实际 %s" % lv_list)
        ok = False
    else:
        print("  ✓ anima 档 LoRA 可见项 = ['none','jm_style']")
    print("  模型档位（可见可选）:")
    for v, lab in (st.get("unetUsable") or []):
        print("      %-52s %s" % (v, lab))
    print("  档位提示 : %s" % (st.get("modelHint") or "").strip()[:110])
    ok &= st.get("controlOptions") == ["none", "canny", "direct"]
    ok &= (st.get("controlModels") or 0) > 0
    ok &= st.get("tagsLinkDisplay") != "none"
    # ★ anima 档：可见模型必须只有它认得的（别让 Qwen 的混进来）
    bad = [v for v, _ in (st.get("unetUsable") or []) if not v.startswith("anima-")
           and not v.startswith("Anima-")]
    if bad:
        print("  ✗ anima 档混进了非 Anima 模型：%s" % bad)
        ok = False
else:
    ok &= st.get("tagsLinkDisplay") == "none"
    # ★ 非 anima 档不许出现 jm_style（它是 Anima 架构的权重）
    if "jm_style" in (st.get("loraVisible") or []):
        print("  ✗ 非 anima 档混进了 jm_style（架构不同，挂了会崩）")
        ok = False
    else:
        print("  ✓ 非 anima 档不含 jm_style")

print()
print("=" * 74)
print("四、提示文案（★ 不能出现写死的 2.1）")
print("=" * 74)
hints = st.get("hints") or {}
# Anima 档：文案里**不该**再出现 "2.1" 或 Qwen 的说法
is_anima = st.get("env", "").startswith("Anima")
for k in ("steps", "cfg", "sampler", "alpha"):
    t = (hints.get(k) or "").strip()
    bad = ("2.1" in t) or ("Qwen" in t)
    if is_anima:
        ok &= not bad
        print("  %s %-8s %s" % ("✗" if bad else "✓", k, t[:96]))
    else:
        print("       %-8s %s" % (k, t[:96]))
if is_anima:
    bad_any = [k for k in ("steps", "cfg", "sampler", "alpha")
               if "2.1" in (hints.get(k) or "") or "Qwen" in (hints.get(k) or "")]
    if bad_any:
        print("  ✗ 这些提示还写着 Qwen/2.1：%s" % bad_any)

print()
print("=" * 74)
print("全部通过 ✅" if (ok and not errs) else "有问题 ❌")
print("=" * 74)
sys.exit(0 if (ok and not errs) else 1)
