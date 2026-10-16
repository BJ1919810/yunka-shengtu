#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""并行下载 12 档 LoRA 权重（比逐个串行快得多）。

为什么要并行：实测 SFTP 单连接 ~4.7 MB/s，但**压缩包传输反而只有 0.78 MB/s**
（tar.gz 在这里没优势）。12 档 × 84 MB = 1 GB，串行要 ~4 分钟；
开 6 条连接并行能压到 1 分钟内。

只下 `pytorch_lora_weights.safetensors`（84 MB/档），
**不下** optimizer.bin / scheduler.bin / random_states.pkl（那是续跑用的，测风格不需要）。

用法：
    python tools/dl_loras.py [--steps 50,100,...] [--out 目录]
"""
from __future__ import annotations

import argparse
import concurrent.futures as cf
import os
import sys
import time

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(HERE)
sys.path.insert(0, ROOT)

import pathlib as _pl
for _r in (_pl.Path(__file__).resolve().parent, *_pl.Path(__file__).resolve().parents):
    if (_r / "paths.py").exists():
        sys.path.insert(0, str(_r))
        break

import paths  # noqa: E402  （路径唯一权威）

import cloud_sh  # noqa: E402

R_DIR = "/root/autodl-tmp/lora-out/bj_style"


def fetch(step: str, out_dir: str) -> tuple[str, bool, str]:
    """下某一档。每次用独立连接（paramiko 的 SFTPClient 不保证线程安全）。"""
    remote = "%s/checkpoint-%s/pytorch_lora_weights.safetensors" % (R_DIR, step)
    local = os.path.join(out_dir, "_raw_ckpt%s.safetensors" % step)
    t0 = time.time()
    try:
        cfg = cloud_sh.load_cfg()
        cli = cloud_sh.connect(cfg)
        sftp = cli.open_sftp()
        sftp.get(remote, local)
        sftp.close()
        cli.close()
        dt = time.time() - t0
        mb = os.path.getsize(local) / 1048576
        return step, True, "%.1f MB  %.1fs  %.2f MB/s" % (mb, dt, mb / max(dt, .01))
    except Exception as e:
        return step, False, "%s: %s" % (type(e).__name__, e)


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--steps", default="50,100,150,200,250,300,350,400,450,500,550,600")
    ap.add_argument("--out", default=str(paths.QWEN_LORA_DIR / "_tmp12"))
    ap.add_argument("--workers", type=int, default=6, help="并行连接数（别太大，SSH 会拒）")
    a = ap.parse_args()

    steps = [s.strip() for s in a.steps.split(",") if s.strip()]
    os.makedirs(a.out, exist_ok=True)

    print("=" * 72)
    print("并行下载 %d 档 → %s  （%d 条连接）" % (len(steps), a.out, a.workers))
    print("=" * 72)
    t0 = time.time()
    results = []
    with cf.ThreadPoolExecutor(max_workers=a.workers) as ex:
        futs = {ex.submit(fetch, s, a.out): s for s in steps}
        for f in cf.as_completed(futs):
            step, ok, msg = f.result()
            results.append((step, ok, msg))
            print("  %s ckpt-%-4s  %s" % ("✓" if ok else "✗", step, msg))
    dt = time.time() - t0

    good = sum(1 for _, ok, _ in results if ok)
    print()
    print("成功 %d / %d，总耗时 %.1fs（平均 %.2f MB/s）"
          % (good, len(steps), dt, good * 84 / max(dt, .01)))
    if good < len(steps):
        print("失败的：", ", ".join(s for s, ok, _ in results if not ok))
    return 0 if good == len(steps) else 1


if __name__ == "__main__":
    raise SystemExit(main())
