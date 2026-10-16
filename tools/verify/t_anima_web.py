# -*- coding: utf-8 -*-
"""本机 Anima 训练台 —— **真浏览器** 验证（复用系统 Edge，不下 Chromium）。

为什么必须跑这个（`.skills/20` 的血泪）：`index.html` 里一个 JS 错误（比如 `let` 的 TDZ、
注释写坏）会让**整块 `<script>` 静默不执行** —— 页面看着正常，但事件全没绑上
（症状就是"点了没反应"）。`node --check` 抓不到，只有真浏览器能抓 `pageerror`。

⛔ 「按钮存在≠按钮有用」：光断言 `onclick` 是 function 不算数，所以第五节起**真点下去**：
    · 真上传（`set_input_files` 塞 2 图 + 2 txt）→ 断言暂存列表/体检/计数都真的变了
    · 真调参 → 点「开始训练」→ **拦下** `/api/job/start` 断言 payload 里就是刚改的值（不真的开训！）
    · 喂一份假 `/api/job` → 断言进度条/步数/loss 真的渲染出来，tqdm 噪声行被滤掉
    · 点「停止训练」→ 断言 `/api/job/stop` 真的被调用

用法：python tools/verify/t_anima_web.py [端口]
    ⚠️ 服务由本脚本自己起在临时端口，并把 `ANIMA_STAGING` 指向临时目录
       ⇒ **绝不碰用户的 `train_anima/_staging`**；用完即杀（try/finally）。
"""
from __future__ import annotations

import json
import os
import shutil
import socket
import struct
import subprocess
import sys
import tempfile
import time

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
PW_PY = str(paths.COMFY_PY)
PORT = sys.argv[1] if len(sys.argv) > 1 else "8794"
FAILS: list[str] = []


def check(cond, msg):
    print("  %s %s" % ("✓" if cond else "✗", msg))
    if not cond:
        FAILS.append(msg)


def fake_png(path, w, h):
    """服务端只读 IHDR 拿宽高 ⇒ 造个头部正确的假 PNG 就够体检用。"""
    with open(path, "wb") as f:
        f.write(b"\x89PNG\r\n\x1a\n" + b"\x00\x00\x00\x0d" + b"IHDR"
                + struct.pack(">II", w, h) + b"\x08\x02")


SCRIPT = r'''
import asyncio, json, os, re, sys, traceback
from playwright.async_api import async_playwright

PORT = sys.argv[1]
FILES = json.loads(sys.argv[2])          # 要喂给 #picker 的文件
STAGE = sys.argv[3]

JOB = {
    "ok": True, "has_job": True, "job_id": "verify01", "name": "verify_ui",
    "out_dir": os.path.join(STAGE, "_out", "verify_ui"),
    "alive": True, "rc": None, "stopped": False, "offset": 123, "elapsed": 42,
    "log": "running training / 学習経過\n"
           "steps:  35%|###       | 7/20 [00:21<00:39,  1.17it/s, avr_loss=0.1342]\n"
           "model saved to somewhere\n",
    "progress": {"step": 7, "total": 20, "pct": 35.0, "loss": 0.1342, "epoch": 1, "epochs": 3},
    "gpu": "1200 MiB, 8192 MiB, 55 %",
    "out_files": [{"name": "verify_ui.safetensors", "path": "x.safetensors", "mb": 12.3}],
}
STATE = {"started": False}
SEEN = {"start_payload": None, "stop_called": False}


async def main():
    out = {"errors": [], "state": {}, "upload": {}, "jobs": {}, "posted": {}, "stopped": {}}
    async with async_playwright() as p:
        b = await p.chromium.launch(channel="msedge")     # 复用系统 Edge
        pg = await b.new_page()
        errs = out["errors"]
        pg.on("console", lambda m: errs.append("console:" + m.text) if m.type == "error" else None)
        pg.on("pageerror", lambda e: errs.append("pageerror:" + str(e)))
        pg.on("dialog", lambda d: asyncio.ensure_future(d.accept()))     # confirm() 一律确认

        # ---- 拦截：绝不真的开训 / 真的停止 ----
        async def on_start(route):
            try:
                SEEN["start_payload"] = json.loads(route.request.post_data or "{}")
            except Exception as e:
                SEEN["start_payload"] = {"__parse_error__": str(e)}
            await route.fulfill(status=200, content_type="application/json",
                                body=json.dumps({"ok": True, "job": {"job_id": "verify01"},
                                                 "cmd": "（被拦截，没有真开训）"}))
            STATE["started"] = True

        async def on_job(route):
            if STATE["started"]:
                await route.fulfill(status=200, content_type="application/json", body=json.dumps(JOB))
            else:
                await route.fulfill(status=200, content_type="application/json",
                                    body=json.dumps({"ok": True, "has_job": False}))

        async def on_stop(route):
            SEEN["stop_called"] = True
            await route.fulfill(status=200, content_type="application/json",
                                body=json.dumps({"ok": True, "detail": "（被拦截）"}))

        # ⛔ 路由匹配器一律用**正则**：早期用 lambda 谓词（`u.path == "/api/job"`），
        #    谓词里一抛异常，Playwright 会把**所有**请求都挂住（连主文档都到不了
        #    domcontentloaded，现象是 goto 超时 —— 极难看出真因）。正则没有这个问题。
        await pg.route(re.compile(r".*/api/job/start(\?.*)?$"), on_start)
        await pg.route(re.compile(r".*/api/job/stop(\?.*)?$"), on_stop)
        await pg.route(re.compile(r".*/api/job(\?.*)?$"), on_job)

        await pg.goto(f"http://127.0.0.1:{PORT}/", wait_until="domcontentloaded")
        # ⛔ 别用 wait_until="networkidle"：`/api/env` 会跑两次"冷启动 import torch"的探测
        #    （每次十几~几十秒）⇒ 30s 的默认超时到不了 networkidle，goto 直接超时。
        # ✅ 改成等**条件**（环境卡不再显示"正在体检"），超时给足。
        await pg.wait_for_function(
            "() => { const e = document.getElementById('env-bar');"
            " return e && e.textContent && !e.textContent.includes('正在体检'); }",
            timeout=180000)
        await pg.wait_for_timeout(1500)

        out["state"] = await pg.evaluate("""() => {
            const $ = id => document.getElementById(id);
            const v = id => $(id) ? $(id).value : null;
            const c = id => $(id) ? $(id).checked : null;
            return {
                dropOnclick: typeof $('drop').onclick, pickerOnchange: typeof $('picker').onchange,
                goOnclick: typeof $('go').onclick, stopOnclick: typeof $('stop').onclick,
                clearOnclick: typeof $('clear').onclick, loraOnclick: typeof $('lora').onclick,
                nControls: document.querySelectorAll('input,select,button').length,
                envBar: ($('env-bar') || {}).textContent || '',
                envRows: ($('env-kv') || {}).querySelectorAll('dt').length,
                dsMsg: ($('ds-msg') || {}).textContent || '',
                dsCount: ($('ds-count') || {}).textContent || '',
                name: v('name'), trigger: v('trigger'), dit: v('dit'),
                steps: v('steps'), resolution: v('resolution'), buckets: c('buckets'),
                dim: v('network_dim'), alpha: v('network_alpha'), optimizer: v('optimizer'),
                b2s: v('blocks_to_swap'), vae2d: c('vae_2d'), cacheTe: c('cache_te'),
                saveState: c('save_state'), tb: c('tensorboard'), resume: v('resume'),
                logText: ($('log') || {}).textContent || '',
                bodyLen: document.body.innerHTML.length,
            };
        }""")

        # ---------- 真上传：塞 2 图 + 2 txt ----------
        await pg.evaluate("document.getElementById('picker').removeAttribute('hidden')")
        await pg.set_input_files("#picker", FILES)
        await pg.wait_for_timeout(4000)
        out["upload"] = await pg.evaluate("""() => {
            const $ = id => document.getElementById(id);
            return {
                dsCount: ($('ds-count') || {}).textContent || '',
                dsMsg: ($('ds-msg') || {}).textContent || '',
                nRows: $('files').querySelectorAll('.frow').length,
                nImgTags: $('files').querySelectorAll('.tag.img').length,
                nTxtTags: $('files').querySelectorAll('.tag.txt').length,
                checkRows: $('ds-check').querySelectorAll('dt').length,
                checkTxt: ($('ds-check') || {}).textContent || '',
            };
        }""")

        # ---------- 真调参 → 点开始训练（拦截）----------
        await pg.evaluate("""() => {
            document.getElementById('name').value = 'ui_probe';
            document.getElementById('steps').value = '20';
            document.getElementById('resolution').value = '768';
            document.getElementById('network_dim').value = '8';
            document.getElementById('blocks_to_swap').value = '24';
            document.getElementById('buckets').checked = false;
            document.getElementById('trigger').value = 'ui_probe';
        }""")
        await pg.click("#go")
        try:
            await pg.wait_for_function(
                "() => (document.getElementById('job-line')||{}).textContent.includes('7/20')",
                timeout=30000)
        except Exception:
            pass
        await pg.wait_for_timeout(1500)
        out["posted"] = SEEN["start_payload"] or {}
        out["jobs"] = await pg.evaluate("""() => {
            const $ = id => document.getElementById(id);
            const bar = $('bar');
            return {
                jobLine: ($('job-line') || {}).textContent || '',
                runMsg: ($('run-msg') || {}).textContent || '',
                barClass: bar.className,
                barWidth: bar.firstElementChild.style.width,
                stopDisabled: $('stop').disabled,
                logText: ($('log') || {}).textContent || '',
            };
        }""")

        # ---------- 点停止（拦截）----------
        await pg.click("#stop")
        await pg.wait_for_timeout(2500)
        out["stopped"] = {"stop_called": SEEN["stop_called"],
                          "runMsg": await pg.evaluate("document.getElementById('run-msg').textContent")}
        await b.close()
    print(json.dumps(out, ensure_ascii=False))


try:
    asyncio.run(main())
except Exception:
    # 无论如何都要吐一行可解析的 JSON，免得外面只看到"浏览器没输出"
    print(json.dumps({"errors": ["FATAL:" + traceback.format_exc()[-900:]], "state": {},
                      "upload": {}, "jobs": {}, "posted": {}, "stopped": {}}, ensure_ascii=False))
    raise
'''

def port_open(port: int) -> bool:
    s = socket.socket()
    try:
        return s.connect_ex(("127.0.0.1", port)) == 0
    finally:
        s.close()


def kill_port(port: int) -> None:
    try:
        r = subprocess.run(["netstat", "-ano"], capture_output=True, text=True, timeout=30)
        pids = {ln.split()[-1] for ln in (r.stdout or "").splitlines()
                if (":%d " % port) in ln and "LISTENING" in ln}
        for pid in pids:
            subprocess.run(["taskkill", "/F", "/T", "/PID", pid], capture_output=True,
                           text=True, timeout=60)
    except Exception:
        pass


def main() -> int:
    print("=" * 74)
    print("  本机 Anima 训练台 · 真浏览器验证（系统 Edge，含真上传/真调参/真进度）")
    print("=" * 74)
    if not os.path.isfile(PW_PY):
        print("[x] 找不到带 playwright 的 python：%s" % PW_PY)
        return 1
    kill_port(int(PORT))
    time.sleep(0.5)
    if port_open(int(PORT)):
        print("[x] 端口 %s 被占用且清不掉" % PORT)
        return 1

    # ⛔ 把暂存指到临时目录 ⇒ 真上传也不会碰到用户的 _staging
    #    （注意：这个目录必须**一开始是空的** —— 否则"空暂存提醒"和"上传后计数"两条断言全没意义）
    stage = tempfile.mkdtemp(prefix="anima_web_stage_")
    up_dir = tempfile.mkdtemp(prefix="anima_web_upload_")
    files = []
    try:
        # 只在上传源目录里造文件；暂存目录保持空着
        fake_png(os.path.join(up_dir, "a.png"), 1200, 900)
        fake_png(os.path.join(up_dir, "b.png"), 600, 800)
        open(os.path.join(up_dir, "a.txt"), "w", encoding="utf-8").write("jm_style, 1girl")
        open(os.path.join(up_dir, "b.txt"), "w", encoding="utf-8").write("没有触发词，有竖排文字")
        files = [os.path.join(up_dir, n) for n in ("a.png", "b.png", "a.txt", "b.txt")]

        env = dict(os.environ)
        env["ANIMA_STAGING"] = stage
        env["PYTHONIOENCODING"] = "utf-8"
        env["PYTHONUNBUFFERED"] = "1"
        log = os.path.join(tempfile.gettempdir(), "anima_web_verify.log")
        srv = None
        try:
            with open(log, "wb") as lf:
                srv = subprocess.Popen([sys.executable, "-u", os.path.join(APP, "server.py"),
                                        "--port", PORT], cwd=APP, env=env,
                                       stdin=subprocess.DEVNULL, stdout=lf, stderr=subprocess.STDOUT,
                                       creationflags=0x08000000 | 0x00000200)
            for _ in range(40):
                if port_open(int(PORT)):
                    break
                time.sleep(0.5)
            check(port_open(int(PORT)), "服务已在 127.0.0.1:%s 起来（暂存指向临时目录）" % PORT)
            # 先把"服务端能不能响应"和"浏览器那侧的问题"分开 —— 否则 goto 超时会指向错误的方向
            import urllib.request as _url
            try:
                with _url.urlopen("http://127.0.0.1:%d/" % int(PORT), timeout=20) as r0:
                    n0 = len(r0.read())
                check(r0.status == 200 and n0 > 5000, "服务端本身能响应（首页 %d 字节）" % n0)
            except Exception as e:
                check(False, "服务端响应失败：%s" % e)

            with tempfile.NamedTemporaryFile("w", suffix=".py", delete=False, encoding="utf-8") as f:
                f.write(SCRIPT)
                tmp = f.name
            r = subprocess.run([PW_PY, tmp, PORT, json.dumps(files), stage], capture_output=True,
                               text=True, encoding="utf-8", errors="replace", timeout=420)
            os.unlink(tmp)
            lines = [l for l in (r.stdout or "").strip().splitlines() if l.startswith("{")]
            if not lines:
                print("[x] 浏览器没输出：", (r.stderr or "")[-900:])
                return 1
            d = json.loads(lines[-1])
            st, errs = d["state"], d["errors"]

            print("\n一、控制台")
            for e in errs:
                print("   ✗ " + e[:200])
            check(not errs, "页面没有 JS 报错（有错 ⇒ 事件可能整块没绑上）")

            print("\n二、事件绑定")
            for k in ("dropOnclick", "pickerOnchange", "goOnclick", "stopOnclick",
                      "clearOnclick", "loraOnclick"):
                check(st.get(k) == "function", "%-16s = %r" % (k, st.get(k)))

            print("\n三、页面内容与默认值")
            check(st.get("nControls", 0) >= 30, "控件数 %s" % st.get("nControls"))
            check(st.get("bodyLen", 0) > 5000, "页面渲染完整（html 长度 %s）" % st.get("bodyLen"))
            check("后端" in (st.get("envBar") or ""), "环境卡渲染了：%s" % (st.get("envBar") or "")[:60])
            check(st.get("envRows", 0) >= 5, "环境体检列出 %s 行" % st.get("envRows"))
            check("一张图都没有" in (st.get("dsMsg") or ""), "空暂存时有提醒")
            for k, want in [("name", "jm_style"), ("dit", "aesthetic"), ("steps", "600"),
                            ("resolution", "1024"), ("dim", "16"), ("alpha", "16"),
                            ("optimizer", "AdamW8bit"), ("b2s", "16")]:
                check(str(st.get(k)) == want, "默认 %s = %s（实际 %r）" % (k, want, st.get(k)))
            for k in ("buckets", "vae2d", "cacheTe", "saveState", "tb"):
                check(st.get(k) is True, "默认勾选 %s" % k)
            check(st.get("resume") == "", "续跑默认留空")

            print("\n四、★ 真上传（set_input_files 塞 2 图 + 2 txt，走真 fetch PUT）")
            upd = d["upload"]
            on_disk = sorted(os.listdir(stage)) if os.path.isdir(stage) else []
            check(len(on_disk) == 4, "服务端暂存目录里正好 4 个文件（实际 %d：%r）" % (len(on_disk), on_disk))
            check("2 张图" in (upd.get("dsCount") or ""),
                  "顶部计数真的变了：%s" % (upd.get("dsCount") or "")[:60])
            check(upd.get("nRows") == 4, "文件列表出现 4 行（实际 %s）" % upd.get("nRows"))
            check(upd.get("nImgTags") == 2 and upd.get("nTxtTags") == 2,
                  "标签数 图2/txt2（实际 %s/%s）" % (upd.get("nImgTags"), upd.get("nTxtTags")))
            check("已收下" in (upd.get("dsMsg") or "") or "体检" in (upd.get("dsMsg") or ""),
                  "上传后有反馈：%s" % (upd.get("dsMsg") or "")[:70])
            check(upd.get("checkRows", 0) >= 4, "体检表渲染了 %s 行" % upd.get("checkRows"))
            ck = upd.get("checkTxt") or ""
            check("短边" in ck, "体检里有尺寸行")
            check(("×1.71" in ck) or ("放大" in ck), "体检算出放大倍数（600 宽 ⇒ ×1.71）")
            # ⛔ caption 告警在 #ds-msg（体检提醒），不在 #ds-check（那张表只有尺寸类数字）
            dm = upd.get("dsMsg") or ""
            check("触发词" in dm or "排版词" in dm,
                  "体检抓到 caption 问题（缺触发词/排版词）：%s" % dm[:80])

            print("\n五、★ 真调参 → 点「开始训练」（拦下请求，不真开训）")
            po = d["posted"]
            check(po.get("name") == "ui_probe", "payload.name = %r" % po.get("name"))
            check(po.get("steps") == 20, "payload.steps = %r（必须是数字 20）" % po.get("steps"))
            check(po.get("resolution") == 768, "payload.resolution = %r" % po.get("resolution"))
            check(po.get("network_dim") == 8, "payload.network_dim = %r" % po.get("network_dim"))
            check(po.get("blocks_to_swap") == 24, "payload.blocks_to_swap = %r" % po.get("blocks_to_swap"))
            check(po.get("buckets") is False, "payload.buckets = %r（取消勾选真的传下去了）" % po.get("buckets"))
            check(po.get("trigger") == "ui_probe", "payload.trigger = %r" % po.get("trigger"))
            check(len(po) >= 25, "payload 字段数 %d（KEYS 都收进去了）" % len(po))

            print("\n六、★ 进度/log 真渲染（喂假 /api/job）")
            jb = d["jobs"]
            check("7/20" in (jb.get("jobLine") or ""), "状态行显示步数：%s" % (jb.get("jobLine") or "")[:80])
            check("0.1342" in (jb.get("jobLine") or ""), "状态行显示 loss 0.1342")
            check(jb.get("barWidth") == "35%", "进度条宽度 35%%（实际 %r）" % jb.get("barWidth"))
            check("indet" not in (jb.get("barClass") or ""), "进度条不是不确定态")
            check(jb.get("stopDisabled") is False, "「停止训练」按钮已可用")
            check("running training" in (jb.get("logText") or ""), "日志正文渲染出来了")
            check("avr_loss" not in (jb.get("logText") or ""),
                  "tqdm 噪声行被滤掉（日志里不该出现 avr_loss）")

            print("\n七、★ 点「停止训练」（拦下请求）")
            check(d["stopped"].get("stop_called") is True, "/api/job/stop 真的被调用了")
        finally:
            if srv is not None:
                try:
                    subprocess.run(["taskkill", "/F", "/T", "/PID", str(srv.pid)],
                                   capture_output=True, text=True, timeout=60)
                except Exception:
                    pass
            kill_port(int(PORT))
            time.sleep(1.2)
            check(not port_open(int(PORT)), "关掉后 %s 端口已释放（没留孤儿）" % PORT)
            # 真上传写进的是临时暂存；确认用户的 _staging 没被碰
            real_stage = os.path.join(APP, "_staging")
            left = sorted(os.listdir(real_stage)) if os.path.isdir(real_stage) else []
            check(not left, "用户的 train_anima/_staging 仍是空的（实际 %r）" % left)
    finally:
        shutil.rmtree(stage, ignore_errors=True)
        shutil.rmtree(up_dir, ignore_errors=True)

    print("\n" + "=" * 74)
    print("全部通过 ✅" if not FAILS else "有问题 ❌（%d 项）" % len(FAILS))
    for f in FAILS:
        print("   ✗ " + f)
    print("=" * 74)
    return 0 if not FAILS else 1


if __name__ == "__main__":
    sys.exit(main())
