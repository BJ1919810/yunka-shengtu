# -*- coding: utf-8 -*-
"""训练台「自动去 BOM」验证（真 HTTP + 真 Edge）。

背景：caption 的 .txt 带 UTF-8 BOM 时，kohya 不去 BOM ⇒ BOM 粘在触发词上
（训练记录 `ss_tag_frequency` 里就是 `\\ufeffjm_style`）⇒ 出图写干净触发词**对不上、风格不生效**。
用户问「就不能自动转吗」⇒ 现在是 **上传即自动去 BOM**（`/api/stage/put`、`/api/stage/zip`），
外加体检里的「一键去掉」按钮（`/api/stage/strip-bom`，兜底处理"上传前就躺在暂存里"的文件）。

本脚本验四件事（都有真副作用可断言，不是"看代码觉得对"）：
  ① 上传带 BOM 的 .txt ⇒ 响应 `bom_stripped=true` **且磁盘上的文件真的没有 BOM 了**
  ② 上传非 UTF-8（GBK）伪 BOM 文件 ⇒ `bom_stripped=false` **且字节原样不动**（不能弄坏别的编码）
  ③ 直接往暂存塞一个带 BOM 的 .txt（模拟历史文件）⇒ `POST /api/stage/strip-bom` 真把它去掉
  ④ 真 Edge 打开页面 ⇒ 体检发现 BOM 时出现「一键去掉」按钮，**点下去之后磁盘文件被改、提示消失**

用法：python tools/verify/t_anima_bom.py
"""
from __future__ import annotations

import json
import os
import shutil
import socket
import subprocess
import sys
import tempfile
import time
import urllib.request

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
PORT = 8799
PW_PY = str(paths.COMFY_PY)
BOM = b"\xef\xbb\xbf"

FAILS: list[str] = []


def check(cond, msg):
    print("  %s %s" % ("✓" if cond else "✗", msg))
    if not cond:
        FAILS.append(msg)


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


def make_fake_png(path: str, w: int, h: int) -> None:
    """造一个「头部合法」的假 PNG（只为让 _img_size 能读尺寸，不是真图）。"""
    import struct
    with open(path, "wb") as f:
        f.write(b"\x89PNG\r\n\x1a\n" + b"\x00\x00\x00\x0d" + b"IHDR" + struct.pack(">II", w, h))


def post(url, data: bytes):
    req = urllib.request.Request(url, method="POST", data=data)
    with urllib.request.urlopen(req, timeout=60) as r:
        return json.loads(r.read().decode("utf-8"))


BROWSER = r'''
import asyncio, json, sys
from playwright.async_api import async_playwright

PORT = sys.argv[1]

async def main():
    async with async_playwright() as p:
        b = await p.chromium.launch(channel="msedge")
        pg = await b.new_page()
        errs = []
        pg.on("pageerror", lambda e: errs.append("pageerror:" + str(e)))
        await pg.goto("http://127.0.0.1:%s/" % PORT, wait_until="load")
        await pg.wait_for_timeout(1500)          # 等 refreshStage + 体检跑完
        before = await pg.evaluate("""() => {
            const b = document.getElementById('debom');
            return {
              hasBtn: !!b,
              btnText: b ? b.textContent : null,
              msg: (document.getElementById('ds-msg') || {}).textContent || '',
            };
        }""")
        if before["hasBtn"]:
            await pg.click("#debom")
            await pg.wait_for_timeout(1500)
        after = await pg.evaluate("""() => ({
            hasBtn: !!document.getElementById('debom'),
            msg: (document.getElementById('ds-msg') || {}).textContent || '',
        })""")
        print(json.dumps({"errors": errs, "before": before, "after": after}, ensure_ascii=False))
        await b.close()

asyncio.run(main())
'''


def main() -> int:
    tmp = tempfile.mkdtemp(prefix="anima_bom_")
    stage = os.path.join(tmp, "stage")
    os.makedirs(stage, exist_ok=True)
    print("=" * 76)
    print("训练台 · 自动去 BOM（真 HTTP + 真 Edge）")
    print("=" * 76)
    print("  隔离暂存：%s" % stage)
    kill_port(PORT)
    env = dict(os.environ)
    env.update({"ANIMA_STAGING": stage, "PYTHONIOENCODING": "utf-8", "PYTHONUNBUFFERED": "1"})
    srv = None
    try:
        with open(os.path.join(tmp, "server.log"), "wb") as lf:
            srv = subprocess.Popen([sys.executable, "-u", os.path.join(APP, "server.py"),
                                    "--port", str(PORT)], cwd=APP, env=env,
                                   stdin=subprocess.DEVNULL, stdout=lf, stderr=subprocess.STDOUT,
                                   creationflags=0x08000000 | 0x00000200)
        for _ in range(40):
            if port_open(PORT):
                break
            time.sleep(0.5)
        check(port_open(PORT), "服务已起在 127.0.0.1:%d" % PORT)

        print("\n① 上传带 BOM 的 .txt（走真 HTTP /api/stage/put）")
        d = post("http://127.0.0.1:%d/api/stage/put?name=1.txt" % PORT,
                 BOM + "jm_style, 1girl, 日式漫画".encode("utf-8"))
        check(d.get("bom_stripped") is True, "响应 bom_stripped=true（%s）" % d)
        raw = open(os.path.join(stage, "1.txt"), "rb").read()
        check(not raw.startswith(BOM), "★ 磁盘上的文件**真的**没有 BOM 了（头 3 字节 = %s）" % raw[:3])
        check(raw.decode("utf-8").startswith("jm_style"), "内容一字节没坏：%r" % raw.decode("utf-8")[:24])

        print("\n② 非 UTF-8（GBK 伪装 BOM）⇒ 必须原样不动")
        gbk = BOM + "触发词".encode("gbk")
        d = post("http://127.0.0.1:%d/api/stage/put?name=gbk.txt" % PORT, gbk)
        check(d.get("bom_stripped") is False, "响应 bom_stripped=false（%s）" % d)
        check(open(os.path.join(stage, "gbk.txt"), "rb").read() == gbk, "字节原样不动（没把别的编码弄坏）")

        print("\n③ 模拟「上传前就躺在暂存里」的带 BOM 文件 ⇒ 一键接口")
        open(os.path.join(stage, "old.txt"), "wb").write(BOM + "jm_style, 老文件".encode("utf-8"))
        d = post("http://127.0.0.1:%d/api/stage/strip-bom" % PORT, b"{}")
        check(d.get("stripped") == 1 and d.get("files") == ["old.txt"], "POST /api/stage/strip-bom 去掉 1 条（%s）" % d)
        check(not open(os.path.join(stage, "old.txt"), "rb").read().startswith(BOM), "old.txt 的 BOM 真没了")

        print("\n④ 真 Edge：体检发现 BOM 时出现按钮，点下去真改文件")
        # ⚠️ 体检的 BOM 检查只扫**与图片配对**的 caption（孤儿 .txt 不参与训练，不查是对的）
        #    ⇒ 这里必须同时放图，否则 n_bom 恒为 0、按钮不会出现（第一版测试就栽在这）
        make_fake_png(os.path.join(stage, "legacy.png"), 1024, 1024)
        open(os.path.join(stage, "legacy.txt"), "wb").write(BOM + "jm_style, 历史文件".encode("utf-8"))
        bp = os.path.join(tmp, "browser.py")
        open(bp, "w", encoding="utf-8").write(BROWSER)
        r = subprocess.run([PW_PY, bp, str(PORT)], capture_output=True, text=True,
                           encoding="utf-8", errors="replace", timeout=300)
        lines = [l for l in (r.stdout or "").strip().splitlines() if l.strip()]
        if not lines:
            check(False, "浏览器没输出：%s" % (r.stderr or "")[-400:])
        else:
            b = json.loads(lines[-1])
            check(not b["errors"], "页面无 JS 报错（%s）" % b["errors"])
            has = bool(b["before"]["hasBtn"])          # ⛔ check() 不返回布尔值，别写 has = check(...)
            check(has, "按钮出现：%r" % b["before"]["btnText"])
            check("BOM" in (b["before"]["msg"] or ""), "点击前提示里有 BOM 字样")
            # ⛔ 下面三条必须**先确认按钮真的出现过**再判 —— 否则"没有按钮"会让它们全部假绿
            if has:
                check(not b["after"]["hasBtn"], "★ 点击后按钮消失（说明 BOM 已清、体检重新跑过）")
                check("BOM" not in (b["after"]["msg"] or ""),
                      "★ 点击后提示里不再有 BOM：%r" % (b["after"]["msg"] or "")[:60])
                left = open(os.path.join(stage, "legacy.txt"), "rb").read()
                check(not left.startswith(BOM), "★ legacy.txt 的 BOM 被那一下点击真去掉了")
            else:
                check(False, "（跳过点击后的三条判据 —— 按钮没出现）")
    finally:
        if srv is not None:
            try:
                subprocess.run(["taskkill", "/F", "/T", "/PID", str(srv.pid)],
                               capture_output=True, text=True, timeout=60)
            except Exception:
                pass
        kill_port(PORT)
        shutil.rmtree(tmp, ignore_errors=True)
        print("\n  端口 %d 已释放：%s" % (PORT, "是" if not port_open(PORT) else "⚠️ 否"))

    print("\n" + "=" * 76)
    print("全部通过 ✅" if not FAILS else "有问题 ❌（%d）" % len(FAILS))
    for f in FAILS:
        print("   ✗ " + f)
    print("=" * 76)
    return 0 if not FAILS else 1


if __name__ == "__main__":
    sys.exit(main())
