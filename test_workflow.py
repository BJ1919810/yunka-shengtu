# -*- coding: utf-8 -*-
"""cloudgen.py 的离线回归测试（不连云卡，零成本，秒级跑完）。

覆盖两个曾经踩坑踩到怀疑人生的点：
  1. TextEncodeQwenImage21.images 是 Autogrow ⇒ API 格式必须用**扁平点号键**
     "images.image_1"；写成嵌套 "images": {...} 会被引擎**静默丢弃**
     （参考图全丢 + 出图恒定 1024²，见 cloudgen.wf_multi 注释）。
  2. --refs 必须用 action="append" ⇒ 重复传参不会被覆盖。
  3. 指定画幅要靠 EmptyLatentImage 顶掉 TextEncodeQwenImage21 自带的那颗 latent
     （后者的尺寸被硬编码成第一张参考图，见 cloudgen.apply_size 注释）。

用法：
    python test_workflow.py
"""
from __future__ import annotations

import argparse
import ast
import json
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import cloudgen as cg  # noqa: E402

HERE = os.path.dirname(os.path.abspath(__file__))
PASS = FAIL = 0


def check(name: str, cond: bool, extra: str = "") -> None:
    global PASS, FAIL
    if cond:
        PASS += 1
        print("  ok   %s" % name)
    else:
        FAIL += 1
        print("  FAIL %s  %s" % (name, extra))


def mkargs(**kw) -> argparse.Namespace:
    base = dict(cmd="multi", prompt="p", negative="", resolution=0, steps=25, cfg=1.0,
                sampler="euler", scheduler="simple", seed=1, prefix="xy",
                unet=cg.DEFAULTS["unet"], clip=cg.DEFAULTS["clip"], vae=cg.DEFAULTS["vae"],
                cache_device="cpu", cache_dtype="default", width=1024, height=1024,
                size=None,
                shift=3.5, src=None, refs=None)
    base.update(kw)
    return argparse.Namespace(**base)


print("[1] wf_multi：Autogrow 必须是扁平点号键")
wf = cg.wf_multi(mkargs(), ["xy_m0_a.jpeg", "xy_m1_b.jpg"])
tex = wf["5"]["inputs"]
check("images.image_1 存在", tex.get("images.image_1") == ["r0", 0], repr(tex.get("images.image_1")))
check("images.image_2 存在", tex.get("images.image_2") == ["r1", 0], repr(tex.get("images.image_2")))
check("没有残留嵌套的 images 键", "images" not in tex)
check("LoadImage 节点已建", wf["r0"]["inputs"]["image"] == "xy_m0_a.jpeg")
check("sampler 用编码节点的 latent", wf["6"]["inputs"]["latent_image"] == ["5", 2])
check("cfg 保留传入值", wf["6"]["inputs"]["cfg"] == 1.0)

print("[2] wf_inpaint：同样扁平")
wfi = cg.wf_inpaint(mkargs(cmd="inpaint"), "src.png", "msk.png")
ti = wfi["5"]["inputs"]
check("images.image_1 = 原图", ti.get("images.image_1") == ["r0", 0])
check("images.image_2 = mask", ti.get("images.image_2") == ["r1", 0])
check("没有残留嵌套的 images 键", "images" not in ti)

print("[3] 自检：旧嵌套写法必须被拦下")
bad = {"5": {"class_type": "TextEncodeQwenImage21",
             "inputs": {"resolution": 0, "images": {"image_1": ["r0", 0]}}}}
try:
    cg._selfcheck_autogrow(bad)
    check("嵌套写法被拦下", False, "自检没报错！")
except SystemExit:
    check("嵌套写法被拦下", True)
check("正确写法能通过", cg._selfcheck_autogrow(wf) is None)

print("[4] split_refs：两种写法都要支持")
check("重复传参", cg.split_refs(["a.png", "b.png"]) == ["a.png", "b.png"])
check("逗号分隔", cg.split_refs(["a.png,b.png"]) == ["a.png", "b.png"])
check("混用 + 空格 + 空项", cg.split_refs(["a.png, b.png", "c.png", ""]) == ["a.png", "b.png", "c.png"])
check("单张", cg.split_refs(["a.png"]) == ["a.png"])

print("[5] argparse 行为：append vs nargs='+'（后者是当年的 bug）")
p1 = argparse.ArgumentParser()
p1.add_argument("--refs", action="append")
n1 = p1.parse_args(["--refs", "a.png", "--refs", "b.png"]).refs
check("append 保住两张", cg.split_refs(n1) == ["a.png", "b.png"], repr(n1))
p2 = argparse.ArgumentParser()
p2.add_argument("--refs", nargs="+")
n2 = p2.parse_args(["--refs", "a.png", "--refs", "b.png"]).refs
check("nargs='+' 确实会只剩最后一张（记录坑）", n2 == ["b.png"], repr(n2))

src = open(os.path.join(HERE, "cloudgen.py"), encoding="utf-8").read()
check('cloudgen.py 里 --refs 用的是 action="append"',
      'add_argument("--refs", action="append"' in src)

# 自检要对**每一种**工作流都跑一遍：只要哪个分支漏改回嵌套 dict，这里就会炸。
# （不用 AST 扫源码 —— SaveImage 的输入本身就叫 images，是列表不是 dict，会误判。）
allwf = {
    "t2i": cg.wf_t2i(mkargs(cmd="t2i", resolution=1024)),
    "multi": wf,
    "i2i": cg.wf_multi(mkargs(cmd="i2i"), ["one.png"]),
    "inpaint": wfi,
    "alpha": cg.wf_alpha("c.png", "m.png", "xy_alpha", 0.5, False),
}
for name, w in allwf.items():
    try:
        cg._selfcheck_autogrow(w)
        check("wf_%s 自检通过" % name, True)
    except SystemExit as e:
        check("wf_%s 自检通过" % name, False, str(e).splitlines()[0] + "：" + str(e).splitlines()[1].strip())

print("[6] 出图尺寸预期（resolution=0 ⇒ image_1 尺寸四舍五入到 32 的倍数）")
SIZES = {("518,728"): (512, 736), ("1080,1920"): (1088, 1920), ("1024,1024"): (1024, 1024)}


def r32(v: int) -> int:
    return round(v / 32) * 32


for k, want in SIZES.items():
    w, h = (int(x) for x in k.split(","))
    check("%s → %s" % (k, want), (r32(w), r32(h)) == want, "得到 %s" % ((r32(w), r32(h)),))

print("[7] 画幅：显式指定 ⇒ 用 EmptyLatentImage 顶掉节点自带的 latent")
check("1MP 1:1 → 1024x1024", cg.calc_size("1:1", 1.0) == (1024, 1024), repr(cg.calc_size("1:1", 1.0)))
w916, h916 = cg.calc_size("9:16", 4.0)
check("4MP 9:16 → 1536x2720（官方 2K 档是 1536x2752，差在取整）",
      (w916, h916) == (1536, 2720), repr((w916, h916)))
check("9:16 实际比例误差 < 1%%", abs((h916 / w916) / (16 / 9) - 1) < 0.01,
      "%.4f" % ((h916 / w916) / (16 / 9) - 1))
check("输出恒为 32 的倍数",
      all(v % 32 == 0 for k in cg.ASPECTS for v in cg.calc_size(k, 1.7)),
      repr({k: cg.calc_size(k, 1.7) for k in cg.ASPECTS}))
try:
    cg.calc_size("5:4", 1.0)
    check("非法比例被拦下", False, "没报错！")
except SystemExit:
    check("非法比例被拦下", True)

# 不指定 ⇒ 保持节点自带 latent（跟随 image_1）
wf_n = cg.wf_multi(mkargs(size=None), ["a.png"])
check("不指定画幅：仍用节点 latent 且不建节点 9",
      wf_n["6"]["inputs"]["latent_image"] == ["5", 2] and "9" not in wf_n)

# 指定 ⇒ 插入 EmptyLatentImage
wf_s = cg.wf_multi(mkargs(size=(1536, 2720)), ["a.png", "b.png"])
check("指定画幅：KSampler 改用 EmptyLatentImage", wf_s["6"]["inputs"]["latent_image"] == ["9", 0],
      repr(wf_s["6"]["inputs"]["latent_image"]))
check("EmptyLatentImage 宽高正确",
      wf_s["9"]["inputs"] == {"width": 1536, "height": 2720, "batch_size": 1},
      repr(wf_s["9"]["inputs"]))
check("指定画幅不影响参考图接线", wf_s["5"]["inputs"].get("images.image_1") == ["r0", 0])
check("指定画幅的工作流自检通过", cg._selfcheck_autogrow(wf_s) is None)

# 非 8 倍数要自动对齐
wf_a = cg.wf_multi(mkargs(size=(1000, 1003)), ["a.png"])
check("非 8 倍数自动对齐", wf_a["9"]["inputs"]["width"] % 8 == 0 and wf_a["9"]["inputs"]["height"] % 8 == 0,
      repr(wf_a["9"]["inputs"]))

# inpaint / t2i 同样支持
check("inpaint 支持指定画幅",
      cg.wf_inpaint(mkargs(cmd="inpaint", size=(1024, 1024)), "s.png", "m.png")["6"]["inputs"]["latent_image"] == ["9", 0])
check("t2i 支持指定画幅",
      cg.wf_t2i(mkargs(cmd="t2i", resolution=1024, size=(2048, 2048)))["6"]["inputs"]["latent_image"] == ["9", 0])

print("[8] 显存预算闸：超限自动降档（实测边界见 cloudgen.py 顶部常量注释）")
P1 = os.path.join(HERE, "input", "pose_white.jpg")
P2 = os.path.join(HERE, "input", "char_blue.jpg")
K4 = os.path.join(HERE, "_tmp_4k_test.jpg")
if not os.path.exists(K4):
    from PIL import Image
    Image.new("RGB", (3840, 2160), (120, 120, 120)).save(K4, quality=55)


def run_guard(a, argv):
    """guard_budget 靠 sys.argv 判断"用户有没有显式传 --resolution"，测试里要模拟。"""
    old = sys.argv
    sys.argv = ["cloudgen.py"] + argv
    try:
        cg.guard_budget(a)
    finally:
        sys.argv = old
    return a


def total_mp(a, n):
    ref = n * a.resolution ** 2 / 1048576.0
    out = (a.size[0] * a.size[1] / 1048576.0) if a.size else a.resolution ** 2 / 1048576.0
    return ref + out, ref, out


R = cg.REF_DEFAULT_RES
a = run_guard(mkargs(cmd="i2i", resolution=R, src=P1), ["i2i"])
check("编辑模式默认档 = %d（★ 不再自动置 0）" % R, a.resolution == R, str(a.resolution))

a = run_guard(mkargs(cmd="multi", refs=[P1, P2], resolution=R), ["multi"])
tot, ref, out = total_mp(a, 2)
check("2 张参考图 + 跟随尺寸：档位与画幅都不动", a.resolution == R and a.size is None, repr((a.resolution, a.size)))
check("2 张参考图 + 跟随尺寸：合计在预算内", tot <= cg.TOTAL_MAX_MP + 1e-6, "%.2fMP" % tot)

# ★ 已被实测跑通的组合不能被误伤（3.57MP 参考 + 4.18MP 输出）
a = run_guard(mkargs(cmd="multi", refs=[P1, P2], resolution=R, size=(1536, 2720)), ["multi"])
check("已跑通的「2 张 + 9:16 4MP」原样放行",
      a.size == (1536, 2720) and a.resolution == R, repr((a.resolution, a.size)))

a = run_guard(mkargs(cmd="multi", refs=[P1] * 10, resolution=R), ["multi"])
tot, ref, out = total_mp(a, 10)
check("10 张参考图：档位自动下调", a.resolution < R, str(a.resolution))
check("10 张参考图：合计仍在预算内", tot <= cg.TOTAL_MAX_MP + 1e-6, "%.2fMP" % tot)

a = run_guard(mkargs(cmd="multi", refs=[P1, P2], resolution=R, size=(4096, 4096)), ["multi"])
check("4096² 输出被夹下来", a.size[0] < 4096, repr(a.size))
tot, ref, out = total_mp(a, 2)
check("夹完后仍在预算内", tot <= cg.TOTAL_MAX_MP + 1e-6, "%.2fMP" % tot)
check("等比：1:1 仍是 1:1", abs(a.size[0] / a.size[1] - 1) < 0.01, repr(a.size))

a = run_guard(mkargs(cmd="multi", refs=[P1, P2], resolution=R, size=(4096, 2304)), ["multi"])
w, h = a.size
check("等比：16:9 仍是 16:9（误差<2%）", abs((w / h) / (16 / 9) - 1) < 0.02, "%d:%d = %.4f" % (w, h, w / h))
check("夹完仍是 32 的倍数", w % 32 == 0 and h % 32 == 0, "%d,%d" % (w, h))

a = run_guard(mkargs(cmd="t2i", resolution=1024, size=(4096, 4096)), ["t2i"])
check("t2i 4096² 被夹到 ≤ 上限",
      a.size[0] * a.size[1] / 1048576.0 <= cg.TOTAL_MAX_MP + 1e-6, repr(a.size))
a = run_guard(mkargs(cmd="t2i", resolution=4096, size=None), ["t2i"])
check("t2i --resolution 4096 被夹下来且仍是 32 的倍数",
      a.resolution < 4096 and a.resolution % 32 == 0, str(a.resolution))

# 显式 --resolution 0：小图保真放行；4K 必须拦（就是把云卡服务打死的那一次）
a = run_guard(mkargs(cmd="multi", refs=[P1, P2], resolution=0), ["multi", "--resolution", "0"])
check("显式 resolution=0 + 小图：保持 0（保真路径没被误伤）", a.resolution == 0, str(a.resolution))
a = run_guard(mkargs(cmd="multi", refs=[K4], resolution=0), ["multi", "--resolution", "0"])
check("显式 resolution=0 + 4K 图：必须降档", a.resolution == R, str(a.resolution))
tot, ref, out = total_mp(a, 1)
check("4K 图降档后合计在预算内", tot <= cg.TOTAL_MAX_MP + 1e-6, "%.2fMP" % tot)

for w0, h0 in ((4096, 4096), (4096, 2304), (3000, 3000), (5000, 2000)):
    w1, h1 = cg.clamp_size_to(2.0, (w0, h0))
    okc = (w1 * h1 / 1048576.0 <= 2.0 + 1e-6) and abs((w1 / h1) / (w0 / h0) - 1) < 0.02
    check("clamp_size_to %dx%d → %dx%d 面积达标且等比" % (w0, h0, w1, h1), okc,
          "面积 %.2f 比例误差 %.4f" % (w1 * h1 / 1048576.0, (w1 / h1) / (w0 / h0) - 1))

try:
    os.remove(K4)
except Exception:
    pass

print("\n%s  通过 %d 项 / 失败 %d 项" % ("🎉 全绿" if FAIL == 0 else "❌ 有失败", PASS, FAIL))
sys.exit(1 if FAIL else 0)
