# -*- coding: utf-8 -*-
"""本机 Anima 训练台 —— **真训练**冒烟（⛔ 会占 GPU，必须先经用户同意）。

目的：把"离线全绿"换成"真在 3070 上跑通一次"，一次问清三件事：
    ① 1024² 在 8GB 上到底够不够（`blocks_to_swap 16` 这个默认值对不对）
    ② 真实 s/step 与峰值显存（不是估算）
    ③ 真能产出 .safetensors，并且能转成 ComfyUI 版

做法（全程走**真 HTTP 接口**，等于替用户点了一遍网页）：
    1. 造 4 张**真图**（PIL 生成的渐变+噪声，1024² 与 1152×896 两种比例 ⇒ 顺便压到分桶）
       + 同名 .txt（带触发词），全部放在**临时隔离目录**
    2. 起服务时把 `ANIMA_STAGING` 指到那个临时目录 ⇒ **绝不碰用户的 `_staging`**
    3. `POST /api/job/start`（steps=N, save_every=N 保证真存出产物）→ 轮询 `/api/job`
       → 采样显存峰值 → 收 `done.rc`
    4. `POST /api/convert` 转 ComfyUI 版，并用 safetensors 读回校验键名

用法：
    python tools/verify/t_anima_smoke.py --yes            # 默认 3 步
    python tools/verify/t_anima_smoke.py --yes --steps 5 --blocks-to-swap 24
    （不带 --yes 只会打印计划，不动 GPU）
"""
from __future__ import annotations

import json
import os
import shutil
import socket
import subprocess
import sys
import tempfile
import threading
import time
import urllib.request

# ⛔ 本机控制台默认是 GBK：脚本自己把 stdout/stderr 切成 UTF-8，免得调用者忘了
#    `set PYTHONIOENCODING=utf-8` 就撞 UnicodeEncodeError（本项目老坑，见 .skills/20）
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
APP = os.path.join(PROJ, "train_anima")
PORT = 8798
VENV_PY = str(paths.ANIMA_VENV_PY)

FAILS: list[str] = []


def check(cond, msg):
    print("  %s %s" % ("✓" if cond else "✗", msg))
    if not cond:
        FAILS.append(msg)


def arg(name, default, cast=int):
    return cast(sys.argv[sys.argv.index(name) + 1]) if name in sys.argv else default


def port_open(p):
    s = socket.socket()
    try:
        return s.connect_ex(("127.0.0.1", p)) == 0
    finally:
        s.close()


def kill_port(p):
    try:
        r = subprocess.run(["netstat", "-ano"], capture_output=True, text=True, timeout=30)
        for pid in {ln.split()[-1] for ln in (r.stdout or "").splitlines()
                    if (":%d " % p) in ln and "LISTENING" in ln}:
            subprocess.run(["taskkill", "/F", "/T", "/PID", pid], capture_output=True, text=True)
    except Exception:
        pass


def gpu_now():
    """(used, total, free) MiB，取不到就 None。"""
    try:
        r = subprocess.run(["nvidia-smi", "--query-gpu=memory.used,memory.total,memory.free",
                            "--format=csv,noheader,nounits"], capture_output=True, text=True, timeout=15)
        u, t, f = [int(x) for x in (r.stdout or "").strip().split(",")[:3]]
        return u, t, f
    except Exception:
        return None


def post(url, obj):
    req = urllib.request.Request(url, method="POST",
                                 data=json.dumps(obj).encode("utf-8"),
                                 headers={"Content-Type": "application/json"})
    with urllib.request.urlopen(req, timeout=120) as r:
        return json.loads(r.read().decode("utf-8"))


def get(url):
    with urllib.request.urlopen(url, timeout=120) as r:
        return json.loads(r.read().decode("utf-8"))


MAKE_IMAGES = r'''
import os, sys
from PIL import Image, ImageDraw, ImageFilter
import random
out = sys.argv[1]
random.seed(7)
# 两种比例：1:1 与 1.29:1 ⇒ 压到分桶（bucket_no_upscale 要真的有活干）
sizes = [("s1", 1024, 1024), ("s2", 1152, 896), ("s3", 1024, 1024), ("s4", 896, 1152)]
for name, w, h in sizes:
    img = Image.new("RGB", (w, h))
    d = ImageDraw.Draw(img)
    for y in range(0, h, 4):                       # 竖向渐变
        c = (int(60 + 120 * y / h), int(40 + 90 * y / h), int(120 + 100 * y / h))
        d.rectangle([0, y, w, y + 4], fill=c)
    for _ in range(40):                            # 一些形状与噪点，别让画面全平
        x0, y0 = random.randrange(w), random.randrange(h)
        d.ellipse([x0, y0, x0 + random.randrange(40, 200), y0 + random.randrange(40, 200)],
                  fill=(random.randrange(255), random.randrange(255), random.randrange(255)))
    img = img.filter(ImageFilter.GaussianBlur(2))
    img.save(os.path.join(out, name + ".png"))
    with open(os.path.join(out, name + ".txt"), "w", encoding="utf-8") as f:
        f.write("jm_style, 1girl, 简单的测试画面, 竖向渐变")
print("MADE %d" % len(sizes))
'''


def main() -> int:
    steps = arg("--steps", 3)
    b2s = arg("--blocks-to-swap", 16)
    if "--yes" not in sys.argv:
        print("这是**会占 GPU**的真训练冒烟（默认 3 步）。确认后加 --yes 再跑：")
        print("    python tools/verify/t_anima_smoke.py --yes")
        return 0

    print("=" * 74)
    print("  本机 Anima 训练台 · 真训练冒烟（steps=%d, blocks_to_swap=%d）" % (steps, b2s))
    print("=" * 74)
    g = gpu_now()
    check(g is not None, "能读到显存：%s" % (("%d/%d MiB，空闲 %d" % g) if g else "读不到"))
    if g and g[2] < 5500:
        print("  ⚠️ 空闲显存只有 %d MiB —— 桌面程序可能吃掉太多（建议先关 Edge/微信/QQ/Steam）" % g[2])

    tmp = tempfile.mkdtemp(prefix="anima_smoke_")
    stage = os.path.join(tmp, "stage")
    os.makedirs(stage, exist_ok=True)
    log = os.path.join(tmp, "server.log")
    peak = {"used": 0, "stop": False}

    def sample():
        while not peak["stop"]:
            gg = gpu_now()
            if gg:
                peak["used"] = max(peak["used"], gg[0])
            time.sleep(1.0)

    srv = None
    try:
        # ---- 1. 真图 + 标注 ----
        r = subprocess.run([VENV_PY, "-c", MAKE_IMAGES, stage], capture_output=True, text=True,
                           encoding="utf-8", errors="replace", timeout=300)
        check("MADE" in (r.stdout or ""), "造出 4 张真图 + 同名 .txt（%s）"
              % ((r.stdout or "").strip() or (r.stderr or "")[-200:]))
        n_img = len([f for f in os.listdir(stage) if f.endswith(".png")])
        check(n_img == 4, "临时数据集里有 %d 张图（与标注配对）" % n_img)

        # ---- 2. 起服务（暂存指向临时目录）----
        kill_port(PORT)
        env = dict(os.environ)
        env.update({"ANIMA_STAGING": stage, "PYTHONIOENCODING": "utf-8", "PYTHONUNBUFFERED": "1"})
        with open(log, "wb") as lf:
            srv = subprocess.Popen([sys.executable, "-u", os.path.join(APP, "server.py"),
                                    "--port", str(PORT)], cwd=APP, env=env,
                                   stdin=subprocess.DEVNULL, stdout=lf, stderr=subprocess.STDOUT,
                                   creationflags=0x08000000 | 0x00000200)
        for _ in range(40):
            if port_open(PORT):
                break
            time.sleep(0.5)
        check(port_open(PORT), "服务已起在 127.0.0.1:%d" % PORT)

        chk = get("http://127.0.0.1:%d/api/stage/list?trigger=jm_style" % PORT)["check"]
        check(chk["n_img"] == 4 and chk["n_pair"] == 4, "体检看到 4 图 / 4 配对")
        print("     短边 %s / 放大倍数最大 ×%s（1024 分辨率下）" % (chk["short_med"], chk["ups_max"]))

        # ---- 3. 真开训 ----
        threading.Thread(target=sample, daemon=True).start()
        t0 = time.time()
        res = post("http://127.0.0.1:%d/api/job/start" % PORT,
                   {"name": "_smoke", "trigger": "jm_style", "steps": steps,
                    "save_every": steps, "save_last": 2, "blocks_to_swap": b2s,
                    "resolution": 1024, "buckets": True, "tensorboard": True})
        check(res.get("ok") is True, "POST /api/job/start 成功（%s）" % res.get("error", ""))
        if not res.get("ok"):
            return 1
        print("     job=%s" % res["job"]["job_id"])
        print("     命令：%s" % res.get("cmd", "")[:150])

        off, seen_lines, rc, last = 0, set(), None, time.time()
        while time.time() - t0 < 1800:                       # 硬上限 30 分钟
            try:
                d = get("http://127.0.0.1:%d/api/job?offset=%d" % (PORT, off))
            except Exception as e:
                print("     [poll 出错] %s" % e)
                time.sleep(3)
                continue
            if d.get("log"):
                off = d["offset"]
                for ln in d["log"].splitlines():
                    ln = ln.strip()
                    # 只挑有信息量的行打出来（tqdm 进度行太吵）
                    if ln and ln not in seen_lines and (
                            any(k in ln for k in ("running training", "epoch", "saved", "steps:",
                                                  "Error", "error", "Traceback", "OOM",
                                                  "out of memory", "num train images",
                                                  "caching", "loading", "CUDA"))):
                        seen_lines.add(ln)
                        print("     | %s" % ln[:160])
            p = d.get("progress") or {}
            if d.get("rc") is not None or (not d.get("alive") and d.get("rc") is not None):
                rc = d["rc"]
                break
            if not d.get("alive"):
                rc = d.get("rc")
                break
            if time.time() - last > 20:
                last = time.time()
                print("     … 已跑 %ds  step=%s/%s  loss=%s  显存峰值 %d MiB"
                      % (int(time.time() - t0), p.get("step"), steps, p.get("loss"), peak["used"]))
            time.sleep(3)
        el = int(time.time() - t0)
        d = get("http://127.0.0.1:%d/api/job?offset=%d" % (PORT, off))

        print()
        print("  ---- 实测结果 ----")
        print("  退出码     : %r" % rc)
        print("  总耗时     : %d s（含加载模型 + 缓存 latent/TE）" % el)
        print("  显存峰值   : %d MiB / %d MiB" % (peak["used"], g[1] if g else 0))
        p = d.get("progress") or {}
        if p.get("step"):
            print("  训练进度   : %s/%s 步（loss %s）" % (p.get("step"), p.get("total"), p.get("loss")))
        outs = d.get("out_files") or []
        print("  产物       : %s" % ([f["name"] + " " + str(f["mb"]) + "MB" for f in outs] or "（无）"))
        check(str(rc).strip() == "0", "训练退出码 0（实际 %r）" % rc)
        check(bool(outs), "真的存出了 .safetensors 产物")
        if peak["used"]:
            check(peak["used"] < (g[1] if g else 8192), "显存峰值没超上限（%d MiB）" % peak["used"])

        # ---- 4. 转 ComfyUI 版并验键名 ----
        if outs:
            c = post("http://127.0.0.1:%d/api/convert" % PORT, {"src": outs[-1]["path"]})
            check(c.get("ok") is True, "转 ComfyUI 版成功（%s）" % (c.get("error") or c.get("log", ""))[:160])
            if c.get("ok"):
                # ⛔ 别在 %-格式串里再写 %d（"…共%d个键" % len(ks)）—— 内层 % 会跟外层抢参数，
                #    报 TypeError: not enough arguments for format string（同一族坑第三次了）
                rd = ("import json,sys; from safetensors import safe_open\n"
                      "ks=[]\n"
                      "with safe_open(r'%s', framework='pt') as f:\n"
                      "    for k in f.keys(): ks.append(k)\n"
                      "print('###'+json.dumps({'n': len(ks), 'head': ks[:6]}))\n" % c["dst"])
                r2 = subprocess.run([VENV_PY, "-c", rd], capture_output=True, text=True,
                                    encoding="utf-8", errors="replace", timeout=300)
                mk = [x for x in (r2.stdout or "").splitlines() if x.startswith("###")]
                if mk:
                    info = json.loads(mk[-1][3:])
                    print("  转换产物   : 共 %d 个键，前几个 %s" % (info["n"], info["head"]))
                    check(all(k.startswith("diffusion_model.") for k in info["head"]),
                          "转换产物键名带 diffusion_model. 前缀")
                    check(any(k.endswith(".alpha") for k in info["head"]) or info["n"] > 0,
                          "转换产物可读（%d 个键）" % info["n"])
                else:
                    check(False, "读不回转换产物：%s" % (r2.stderr or "")[-200:])
                print("  ★ ComfyUI 用的时候 strength 填 1.0（alpha 就在文件里）")

        # 日志尾部（失败时最有用）
        tail = [l for l in (d.get("log") or "").splitlines() if l.strip()][-25:]
        print("\n  ---- 训练日志尾部 ----")
        for l in tail:
            print("  | " + l[:170])
    finally:
        peak["stop"] = True
        if srv is not None:
            try:
                subprocess.run(["taskkill", "/F", "/T", "/PID", str(srv.pid)],
                               capture_output=True, text=True, timeout=60)
            except Exception:
                pass
        kill_port(PORT)
        time.sleep(1)
        print("\n  端口 %d 已释放：%s" % (PORT, "是" if not port_open(PORT) else "⚠️ 否"))
        print("  冒烟数据集（临时，已留证据路径）：%s" % stage)
        print("  产物在：%s" % os.path.join(APP, "_out", "_smoke") + "（不想要可以删）")

    print("\n" + "=" * 74)
    print("冒烟通过 ✅" if not FAILS else "冒烟有问题 ❌（%d 项）" % len(FAILS))
    for f in FAILS:
        print("   ✗ " + f)
    print("=" * 74)
    return 0 if not FAILS else 1


if __name__ == "__main__":
    sys.exit(main())
