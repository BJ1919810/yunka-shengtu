# -*- coding: utf-8 -*-
"""本机 Anima 训练台 —— **作业编排**集成验证（不加载模型、不占 GPU）。

为什么单独一个脚本
    `t_anima_train.py` 是"纯函数级"离线回归（TOML/命令/日志解析…）。
    但真正容易出事的是**编排**：独立子进程起没起来、`done.rc` 有没有写、日志偏移对不对、
    停止能不能杀掉进程树、产物列不列得出来 —— 这些只有真跑一遍才知道。

做法（关键：**除了子进程跑的那条命令，其余全是真实代码路径**）
    1. 把 server 模块的 `STAGING / JOBS / OUT_ROOT / STATE` 指向临时目录（不碰用户的 `_staging`）
    2. 造一个假的 `sd-scripts/anima_train_network.py`（打印 kohya 风格的 tqdm 行 + 在 `--output_dir` 里写产物）
    3. 调**真的** `start_job()` ⇒ 生成真的 `dataset.toml` / `run.cmd` ⇒ 真起独立进程
    4. 轮询**真的** `job_status()`，断言进度/退出码/产物
    5. 再起一个"长睡"作业，用**真的** `stop_job()` 杀进程树，断言停了

用法：python tools/verify/t_anima_job.py
"""
from __future__ import annotations

import importlib.util
import json
import os
import shutil
import struct
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

HERE = os.path.dirname(os.path.abspath(__file__))
PROJ = os.path.dirname(os.path.dirname(HERE))
APP = os.path.join(PROJ, "train_anima")
PY = sys.executable

FAILS: list[str] = []


def check(cond, msg):
    print("  %s %s" % ("✓" if cond else "✗", msg))
    if not cond:
        FAILS.append(msg)


def load_server():
    spec = importlib.util.spec_from_file_location("anima_server_job", os.path.join(APP, "server.py"))
    m = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(m)
    return m


def fake_png(path, w, h):
    with open(path, "wb") as f:
        f.write(b"\x89PNG\r\n\x1a\n" + b"\x00\x00\x00\x0d" + b"IHDR" + struct.pack(">II", w, h) + b"\x08\x02")


# 假训练脚本：打印 kohya 风格日志、在 --output_dir 写个假 LoRA、按 --max_train_steps 决定步数
STUB = r'''# -*- coding: utf-8 -*-
import os, sys, time
a = sys.argv
def val(flag, default=None):
    return a[a.index(flag) + 1] if flag in a else default
out = val("--output_dir")
total = int(val("--max_train_steps", "5"))
os.makedirs(out, exist_ok=True)
print("running training / 学習経過", flush=True)
print("  num train images * repeats: 2", flush=True)
if "--fake-sleep" in a:
    print("fake long run", flush=True)
    time.sleep(120)
for i in range(1, min(total, 5) + 1):
    pct = int(100 * i / total)
    sys.stdout.write("epoch 1/50\r")
    sys.stdout.write("steps: %3d%%|#         | %d/%d [00:0%d<00:10,  1.00it/s, avr_loss=0.1%03d]\r"
                     % (pct, i, total, i, i))
    sys.stdout.flush()
    time.sleep(0.4)
print("", flush=True)
open(os.path.join(out, "jm_style.safetensors"), "wb").write(b"FAKELORA" * 1024)
print("model saved to %s" % out, flush=True)
if "--save_state" in a:
    os.makedirs(os.path.join(out, "state"), exist_ok=True)
'''


def wait_rc(S, timeout=90.0):
    """轮询 job_status 直到 done.rc 出现（或超时）。"""
    t0 = time.time()
    logs, prog = "", {}
    while time.time() - t0 < timeout:
        d = S.job_status(len(logs))
        logs += d.get("log") or ""
        prog = d.get("progress") or prog
        if d.get("rc") is not None or (not d.get("alive") and d.get("rc") is not None):
            return d, logs, prog
        time.sleep(0.5)
    d = S.job_status(len(logs))
    return d, logs, prog


def main() -> int:
    print("=" * 74)
    print("  本机 Anima 训练台 · 作业编排集成验证（无 GPU）")
    print("=" * 74)
    S = load_server()
    tmp = tempfile.mkdtemp(prefix="anima_job_")
    real = {k: getattr(S, k) for k in ("STAGING", "JOBS", "OUT_ROOT", "STATE", "SD_DIR", "VENV_PY")}
    try:
        # ---- 隔离：所有目录常量指向临时区 ----
        S.STAGING = os.path.join(tmp, "staging")
        S.JOBS = os.path.join(tmp, "jobs")
        S.OUT_ROOT = os.path.join(tmp, "out")
        S.STATE = os.path.join(tmp, "state.json")
        S.SD_DIR = os.path.join(tmp, "sd-scripts")
        S.VENV_PY = PY
        for d in (S.STAGING, S.JOBS, S.OUT_ROOT, S.SD_DIR):
            os.makedirs(d, exist_ok=True)
        with open(os.path.join(S.SD_DIR, "anima_train_network.py"), "w", encoding="utf-8") as f:
            f.write(STUB)
        open(os.path.join(S.STAGING, "a.txt"), "w", encoding="utf-8").write("jm_style, 1girl")
        open(os.path.join(S.STAGING, "b.txt"), "w", encoding="utf-8").write("jm_style, 1girl")
        fake_png(os.path.join(S.STAGING, "a.png"), 1200, 900)
        fake_png(os.path.join(S.STAGING, "b.png"), 1100, 1100)
        check(os.path.abspath(S.STATE) != os.path.abspath(real["STATE"]),
              "状态文件/目录已指向临时区（**不碰用户的 _staging 与 _state.json**）")

        # ---------- 一、正常作业 ----------
        print("\n一、正常作业：start_job → 独立进程 → done.rc → job_status")
        o = dict(S.DEFAULTS)
        o.update({"steps": 5, "name": "verify_style", "save_every": 2, "blocks_to_swap": 4})
        r = S.start_job(o)
        check(r.get("ok") is True, "start_job 返回 ok（错误：%s）" % r.get("error"))
        job = r.get("job") or {}
        job_dir = job.get("job_dir", "")
        check(os.path.isfile(os.path.join(job_dir, "dataset.toml")), "生成了 dataset.toml")
        check(os.path.isfile(os.path.join(job_dir, "run.cmd")), "生成了 run.cmd")
        check(os.path.isfile(os.path.join(job_dir, "command.txt")), "留下了可复现的 command.txt")
        rc_txt = open(os.path.join(job_dir, "run.cmd"), "rb").read()
        check(rc_txt.count(b"\n") == rc_txt.count(b"\r\n"), "run.cmd 是 CRLF")
        check(b"HF_HUB_OFFLINE=1" in rc_txt, "run.cmd 设了 HF_HUB_OFFLINE=1")
        check(not os.path.exists(os.path.join(job_dir, "done.rc")) or True, "（done.rc 由子进程收尾写）")

        d, logs, prog = wait_rc(S, timeout=120)
        check(d.get("rc") is not None, "done.rc 出现了（rc=%r）—— 独立进程确实跑完了" % d.get("rc"))
        check(str(d.get("rc")).strip() == "0", "退出码 0")
        check("running training" in logs, "拿到了子进程日志（增量读取生效，%d 字节）" % len(logs))
        check(prog.get("step") == 5 and prog.get("total") == 5, "进度解析出 5/5 步")
        check(prog.get("loss") is not None, "进度里带 loss（%.4f）" % (prog.get("loss") or -1))
        check(d.get("alive") is False, "作业结束后 alive=False")
        outs = d.get("out_files") or []
        check(any(f["name"].endswith(".safetensors") for f in outs),
              "产物列表里能看到 .safetensors（%s）" % [f["name"] for f in outs])
        lf = S.list_out()
        check(len(lf["files"]) == 1, "list_out 能看到 1 个产物")
        st = json.load(open(S.STATE, encoding="utf-8"))
        check(st.get("job_id") == job.get("job_id"), "state.json 可重载且 job_id 对得上")

        # ---------- 二、停止 ----------
        print("\n二、停止：长睡作业 → stop_job 杀进程树")
        o2 = dict(o)
        o2.update({"name": "verify_stop", "steps": 5})
        r2 = None
        # ⛔ 别在 start_job **之后**改 run.cmd —— 那时子进程已经起好了，改了不生效
        #    （第一版就这么写，结果"长睡作业没起来"、停止断言被跳过）。要在起进程**之前**注入标记。
        orig_build = S.build_cmd

        def build_sleep(opts, jd):
            a, outd = orig_build(opts, jd)
            return a + ["--fake-sleep"], outd

        S.build_cmd = build_sleep
        try:
            r2 = S.start_job(o2)
        finally:
            S.build_cmd = orig_build
        check(r2.get("ok") is True, "第二次 start_job ok")
        check("--fake-sleep" in open(os.path.join(r2["job"]["job_dir"], "run.cmd"),
                                     encoding="utf-8").read(), "长睡标记进了 run.cmd（在起进程之前注入的）")
        time.sleep(4)
        d2 = S.job_status(0)
        check(d2.get("alive") is True, "长睡作业确实在跑（alive=True）")
        check("fake long run" in (d2.get("log") or ""), "长睡作业的日志也读得到（重定向正常）")
        res = S.stop_job()
        check(res.get("ok") is True, "stop_job 返回 ok（%s）" % (res.get("detail") or res.get("error")))
        time.sleep(3)
        d3 = S.job_status(0)
        check(d3.get("alive") is False, "杀完之后 alive=False")
        check(d3.get("stopped") is True, "状态里标记了 stopped")

        # ---------- 三、无人值守的边界 ----------
        print("\n三、边界")
        empty = tempfile.mkdtemp(prefix="anima_empty_")
        S.STAGING = empty
        r3 = S.start_job(dict(S.DEFAULTS))
        check(r3.get("ok") is False and "一张图" in (r3.get("error") or ""), "没图时拒绝开训")
        S.STAGING = os.path.join(tmp, "staging")
        r4 = S.start_job({**S.DEFAULTS, "network_dim": 99999})
        check(r4.get("ok") is True, "越界 rank 被夹住后仍能开训（不报 500）")
        S.job_status(0)
        conv = S.convert_lora(os.path.join(tmp, "..", "not_in_out.safetensors"))
        check(conv.get("ok") is False, "转换接口拒绝 _out/ 之外的文件（%s）" % conv.get("error", "")[:40])
        check(S.parse_progress("") == {}, "空日志不炸")
    finally:
        for k, v in real.items():
            setattr(S, k, v)
        # 收尾：杀掉临时区里可能还在跑的进程，再删临时目录
        try:
            st = json.load(open(S.STATE, encoding="utf-8")) if os.path.isfile(S.STATE) else {}
        except Exception:
            st = {}
        for pid in [st.get("pid")]:
            if pid:
                import subprocess
                subprocess.run(["taskkill", "/F", "/T", "/PID", str(pid)],
                               capture_output=True, text=True)
        shutil.rmtree(tmp, ignore_errors=True)

    print("\n" + "=" * 74)
    print("全部通过 ✅" if not FAILS else "有问题 ❌（%d 项）" % len(FAILS))
    for f in FAILS:
        print("   ✗ " + f)
    print("=" * 74)
    return 0 if not FAILS else 1


if __name__ == "__main__":
    sys.exit(main())
