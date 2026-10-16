# -*- coding: utf-8 -*-
"""显存上限实测探针 —— 找出 RTX 3080 10GB 上「参考图面积 / 输出画幅」的真实天花板。

为什么需要它：`resolution=0` 时 4K 参考图会撑爆显存。要设"预算闸"就得知道极限在哪，
而之前只有两个孤立数据点（3.57MP 参考图 OK、4.18MP 输出 OK）。

★ 2026-10-03 实测发现（很重要）：超限的后果**不是任务报错，而是整个 ComfyUI 进程被 Abort**。
  日志链：`Ran out of memory when regular VAE decoding`（回退分块解码）
        → `CUDA error: invalid argument` → `Fatal Python error: Aborted`。
  崩溃点在 `VAEDecode`，不在采样。⇒ 一次越界请求会把云卡服务整个打死，
  所以探针必须能**自动重启服务**，否则后面的实验全部跑不了。

设计要点：
  · **steps=1** —— 只为压显存，不要质量；OOM 发生在编码/解码，与步数无关。
  · 单条 SSH 隧道跑完全部，省掉反复握手（AutoDL 的 MaxStartups 很脆）。
  · 每个用例跑完探活 + 抓 comfy.log 的 OOM/崩溃标记；挂了就抓完日志再重启。
  · A 组只变输出画幅（t2i 无参考图）；B 组只变参考图面积（输出固定 1MP）。

用法：
    python probe_vram.py --phase A     # 只测输出画幅
    python probe_vram.py --phase B     # 只测参考图面积
"""
from __future__ import annotations

import argparse
import json
import os
import sys
import time
import uuid

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

import cloudgen as C
import cloud_sh

HERE = os.path.dirname(os.path.abspath(__file__))
OUT = os.path.join(HERE, "out", "probe")
STEPS = 1
PROMPT = "a plain test image, simple flat color background"
TIMEOUT = 300.0
LOG = "/root/autodl-tmp/comfy.log"
START_SH = os.path.join(HERE, "remote", "start_comfy.sh")


# ------------------------------------------------------------------ 服务健康 / 重启

def alive(t) -> bool:
    try:
        C.api_json(t, "/system_stats")
        return True
    except Exception:
        return False


def log_markers(cli, n: int = 6) -> str:
    out, _, _ = cloud_sh.run(
        cli,
        "grep -niE 'out of memory|invalid argument|Fatal Python error|Aborted' %s "
        "| tail -%d" % (LOG, n), timeout=60)
    return " ／ ".join(l.strip()[:90] for l in out.strip().splitlines())


def restart(cli) -> str:
    """用项目里的幂等启动脚本重启。⚠️ 它会清空 comfy.log，所以必须先抓日志。"""
    with open(START_SH, encoding="utf-8") as f:
        out, err, rc = cloud_sh.run(cli, f.read(), timeout=300)
    return (out.strip().splitlines() or [""])[0][:80]


# ------------------------------------------------------------------ 单个用例

def run_probe(t, wf: dict, out_path: str) -> dict:
    r = {"ok": False, "err": "", "secs": 0.0, "w": 0, "h": 0}
    C._selfcheck_autogrow(wf)
    try:
        pid = C.api_json(t, "/prompt",
                         {"prompt": wf, "client_id": uuid.uuid4().hex})["prompt_id"]
    except Exception as e:
        r["err"] = "提交失败（服务可能已挂）: %s" % str(e)[:120]
        return r
    t0 = time.time()
    while True:
        if time.time() - t0 > TIMEOUT:
            r["err"] = "超时 %.0fs" % TIMEOUT
            break
        try:
            hist = C.api_json(t, "/history/" + pid)
        except Exception as e:
            r["err"] = "轮询中断（服务大概率崩了）: %s" % str(e)[:100]
            break
        if hist:
            entry = hist.get(pid) or {}
            st = entry.get("status") or {}
            if st.get("status_str") == "error":
                flat = json.dumps(st.get("messages") or [], ensure_ascii=False)
                r["err"] = ("OOM" if "out of memory" in flat.lower() else "ERROR") + ": " + flat[:200]
                break
            imgs = [im for node in (entry.get("outputs") or {}).values()
                    for im in (node.get("images") or [])]
            if imgs:
                im = imgs[0]
                raw = C.api_get(t, "/view?filename=%s&subfolder=%s&type=%s" % (
                    im["filename"], im.get("subfolder", ""), im.get("type", "output")))
                os.makedirs(os.path.dirname(os.path.abspath(out_path)), exist_ok=True)
                with open(out_path, "wb") as f:
                    f.write(raw)
                try:
                    from PIL import Image
                    with Image.open(out_path) as pim:
                        r["w"], r["h"] = pim.size
                except Exception:
                    pass
                r["ok"] = True
                break
        time.sleep(1.5)
    r["secs"] = time.time() - t0
    return r


def run_case(t, cli, tag, wf, ref_mp, out_mp, order, results):
    r = run_probe(t, wf, os.path.join(OUT, "%02d_%s.png" % (order, tag)))
    r.update(tag=tag, ref_mp=ref_mp, out_mp=out_mp)
    if not r["ok"] and not r["err"]:
        r["err"] = "未知失败"
    r["crashed"] = (not r["ok"]) and (not alive(t))
    if not r["ok"]:
        r["markers"] = log_markers(cli)
        if r["crashed"]:
            r["restart"] = restart(cli)
    print("[%s] %-16s 参考%5.2fMP 输出%5.2fMP  %-13s %5.0fs%s"
          % ("OK  " if r["ok"] else "炸了", tag, ref_mp, out_mp,
             "%dx%d" % (r["w"], r["h"]) if r["ok"] else "-", r["secs"],
             "  ★服务被拖崩，已重启：" + r.get("restart", "") if r.get("crashed") else ""),
          flush=True)
    if r["err"]:
        print("        └─ %s" % r["err"][:180], flush=True)
    if r.get("markers"):
        print("        └─ 日志: %s" % r["markers"][:200], flush=True)
    results.append(r)
    return r


def make_ns(**kw):
    d = dict(C.DEFAULTS)
    d.update(prompt=PROMPT, negative="", seed=12345, prefix="probe", size=None, cmd="multi")
    d.update(kw)
    return argparse.Namespace(**d)


# ------------------------------------------------------------------ 各阶段

def phase_a(t, cli, results):
    print("=" * 78)
    print("A 组：只变**输出画幅**（t2i，无参考图，steps=%d）" % STEPS)
    print("=" * 78)
    for i, side in enumerate((2048, 3072, 3328, 3584), 1):
        a = make_ns(cmd="t2i", steps=STEPS, resolution=1024)
        a.size = (side, side)
        r = run_case(t, cli, "t2i_%d" % side, C.wf_t2i(a), 0.0,
                     side * side / 1048576.0, i, results)
        if r.get("crashed"):
            print("  （输出画幅上限到此为止，更大的不再试）", flush=True)
            break


def phase_b(t, cli, results):
    print("=" * 78)
    print("B 组：只变**参考图面积**（2 张参考图，输出固定 1024x1024 = 1MP）")
    print("=" * 78)
    refs_local = [os.path.join(HERE, "input", "pose_white.jpg"),
                  os.path.join(HERE, "input", "char_blue.jpg")]
    for p in refs_local:
        if not os.path.exists(p):
            sys.exit("缺参考图：%s" % p)
    if not alive(t):
        print("[重启服务] %s" % restart(cli), flush=True)
    uploaded = [C.upload_image(t, p, C.ascii_name(p, "probe_%d_" % i))
                for i, p in enumerate(refs_local)]
    print("[上传] %s" % uploaded, flush=True)

    n = 0
    for res in (1024, 1536, 2048, 2560):
        n += 1
        a = make_ns(cmd="multi", steps=STEPS, resolution=res)
        a.size = (1024, 1024)
        r = run_case(t, cli, "ref_%d" % res, C.wf_multi(a, uploaded),
                     2 * (res * res) / 1048576.0, 1.0, n, results)
        if r.get("crashed"):
            print("  （参考图上限到此为止，更大的不再试）", flush=True)
            break

    # 现实组合：不错的参考图 + 9:16 4MP 输出（就是用户日常会用的那一档）
    n += 1
    a = make_ns(cmd="multi", steps=STEPS, resolution=1536)
    a.size = (1536, 2720)
    run_case(t, cli, "combo_9x16", C.wf_multi(a, uploaded),
             2 * (1536 * 1536) / 1048576.0, 1536 * 2720 / 1048576.0, n, results)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--phase", default="both", choices=["A", "B", "both"])
    args = ap.parse_args()

    cli = cloud_sh.connect()
    t = C.Tunnel()
    results = []
    try:
        if args.phase in ("A", "both"):
            phase_a(t, cli, results)
        if args.phase in ("B", "both"):
            phase_b(t, cli, results)
    finally:
        t.close()
        cli.close()

    print()
    print("=" * 78)
    print("%-16s %8s %8s %-8s %8s  %s" % ("case", "参考MP", "输出MP", "结果", "耗时s", "说明"))
    for r in results:
        print("%-16s %8.2f %8.2f %-8s %8.0f  %s"
              % (r["tag"], r["ref_mp"], r["out_mp"], "OK" if r["ok"] else "FAIL",
                 r["secs"], "服务被拖崩" if r.get("crashed") else ""))
    out = os.path.join(OUT, "result.json")
    os.makedirs(OUT, exist_ok=True)
    with open(out, "w", encoding="utf-8") as f:
        json.dump(results, f, ensure_ascii=False, indent=1)
    print("\n结果已存 %s" % out)


if __name__ == "__main__":
    main()
