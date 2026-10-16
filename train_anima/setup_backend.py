# -*- coding: utf-8 -*-
"""一键装本机训练后端：clone sd-scripts + 建 venv + 补几个小依赖。

为什么这么装（都是本机实测结论，别"简化"）：
  1. **别动 git 全局代理** —— 本机 git 配了
     `http.https://github.com.proxy = http://127.0.0.1:7892`，而那个代理现在没在跑
     ⇒ 直接 clone 会报 `Failed to connect to github.com:443 over proxy`。
     本脚本用 `-c http.https://github.com.proxy=` **单次绕开**（GitHub 直连实测 1.9s 通），
     改的是这条命令，不是用户的全局配置。
  2. **venv 从「基础解释器」用 `--system-site-packages` 建**（位置见 `paths.py` 的
     `PY312`，或用 `ANIMA_BASE_PY` 覆盖）—— 那个解释器里
     已经有 torch 2.6.0+cu126 / diffusers / peft / accelerate / bitsandbytes / transformers，
     继承过来只补 4 个小包即可，**不用再下 2.5GB 的 torch**。
  3. **pip 走腾讯源** —— 本机实测最快（官方 PyPI 49KB/s，基本等于废）。

用法：`python train_anima/setup_backend.py`（幂等，可重复跑）
"""
from __future__ import annotations

import os
import subprocess
import sys

import pathlib as _pl
_here = _pl.Path(__file__).resolve()
for _r in (_here.parent, *_here.parents):          # 向上找仓库根（含 paths.py）
    if (_r / "paths.py").exists():
        sys.path.insert(0, str(_r))
        break
import paths  # noqa: E402  （路径唯一权威）

BASE_PY = str(paths.PY312)
ANIMA_ROOT = str(paths.ANIMA_ROOT)
SD_DIR = str(paths.ANIMA_SD_DIR)
VENV = os.environ.get("ANIMA_VENV", os.path.join(ANIMA_ROOT, "train-env"))
VENV_PY = os.path.join(VENV, "Scripts", "python.exe")
PIP_INDEX = os.environ.get("ANIMA_PIP_INDEX", "https://mirrors.cloud.tencent.com/pypi/simple")

# sd-scripts 在 Anima 这条路上会 import、而 Python312 里**没有**的小包
NEED = ["toml", "voluptuous", "imagesize", "ftfy"]

# ⛔ GitHub 直连**小请求通、大传输被墙重置**（2026-10-15 实测：ls-remote 成功，clone 报
#    "Recv failure: Connection was reset"）⇒ 必须带镜像兜底。
CLONE_URLS = [
    "https://github.com/kohya-ss/sd-scripts",
    "https://ghproxy.net/https://github.com/kohya-ss/sd-scripts",
    "https://gh-proxy.com/https://github.com/kohya-ss/sd-scripts",
    "https://gitclone.com/github.com/kohya-ss/sd-scripts",
]
TARBALL = "https://ghproxy.net/https://github.com/kohya-ss/sd-scripts/archive/refs/heads/main.tar.gz"


def run(cmd, **kw):
    print("  $ " + " ".join(cmd), flush=True)
    return subprocess.run(cmd, text=True, encoding="utf-8", errors="replace", **kw)


def clone_sd(target: str) -> bool:
    """挨个通道试；`-c http.https://github.com.proxy=` 是**单次**绕开本机那个没在跑的代理。"""
    import shutil
    script = os.path.join(target, "anima_train_network.py")
    for url in CLONE_URLS:
        if os.path.isdir(target):
            shutil.rmtree(target, ignore_errors=True)
        print("  → 试通道：%s" % url, flush=True)
        r = run(["git", "-c", "http.https://github.com.proxy=", "clone", "--depth", "1", url, target])
        if r.returncode == 0 and os.path.isfile(script):
            print("  ✓ clone 成功")
            return True
    # 兜底：整包 tar.gz（clone 走到一半被重置时用它）
    print("  → clone 全失败，改下 tar.gz：%s" % TARBALL, flush=True)
    import tarfile
    import tempfile
    tmp = os.path.join(tempfile.gettempdir(), "sd-scripts-main.tar.gz")
    r = run(["curl", "-L", "--noproxy", "*", "--max-time", "600", "-o", tmp, TARBALL])
    if r.returncode != 0 or not os.path.isfile(tmp):
        return False
    if os.path.isdir(target):
        shutil.rmtree(target, ignore_errors=True)
    with tarfile.open(tmp) as tf:
        root = tf.getnames()[0].split("/")[0]
        tf.extractall(os.path.dirname(target))
    os.rename(os.path.join(os.path.dirname(target), root), target)
    os.remove(tmp)
    return os.path.isfile(script)


def main() -> int:
    print("=" * 70)
    print("  本机 Anima 训练后端安装")
    print("=" * 70)
    if not os.path.isfile(BASE_PY):
        print("[x] 找不到基础解释器：%s（设 ANIMA_BASE_PY 覆盖）" % BASE_PY)
        return 1
    print("  基础解释器 : %s" % BASE_PY)
    print("  sd-scripts : %s" % SD_DIR)
    print("  venv       : %s" % VENV)
    print()

    # ---- 1. clone sd-scripts ----
    script = os.path.join(SD_DIR, "anima_train_network.py")
    if os.path.isfile(script):
        print("[1/3] sd-scripts 已就位，跳过 clone")
    else:
        print("[1/3] clone sd-scripts（浅克隆；单次绕开本机那个没在跑的 git 代理，失败自动换镜像）")
        os.makedirs(os.path.dirname(SD_DIR), exist_ok=True)
        if not clone_sd(SD_DIR):
            print("[x] 所有通道都失败。手动装法（任选）：")
            print('    git -c http.https://github.com.proxy= clone --depth 1 '
                  'https://ghproxy.net/https://github.com/kohya-ss/sd-scripts "%s"' % SD_DIR)
            print('    curl -L --noproxy "*" -o sd.tar.gz %s && tar -xzf sd.tar.gz' % TARBALL)
            return 1
    # Anima 支持是 v0.10.1 加进来的
    if not os.path.isfile(os.path.join(SD_DIR, "anima_train_network.py")):
        print("[x] 这个 sd-scripts 太旧：没有 anima_train_network.py（需 v0.10.1+）")
        return 1

    # ---- 2. venv ----
    if os.path.isfile(VENV_PY):
        print("[2/3] venv 已存在，跳过创建")
    else:
        print("[2/3] 建 venv（--system-site-packages ⇒ 直接继承 torch/diffusers/peft/accelerate）")
        r = run([BASE_PY, "-m", "venv", "--system-site-packages", VENV])
        if r.returncode != 0:
            print("[x] venv 创建失败")
            return 1

    # ---- 3. 补小依赖 ----
    print("[3/3] 补 sd-scripts 需要、而基础解释器里没有的小包：%s" % ", ".join(NEED))
    r = run([VENV_PY, "-m", "pip", "install", "-q", "-i", PIP_INDEX, *NEED])
    if r.returncode != 0:
        print("[!] pip 装失败（可能只是某个包在源里没有）—— 下面直接做 import 自检。")

    # ---- 自检：真把训练脚本 import 一遍 ----
    print()
    print("=" * 70)
    print("  自检")
    print("=" * 70)
    r = run([VENV_PY, "-c",
             "import torch, sys;"
             "print('torch %s | cuda %s | %s' % (torch.__version__, torch.cuda.is_available(),"
             " torch.cuda.get_device_name(0) if torch.cuda.is_available() else 'NO-GPU'))"],
            cwd=SD_DIR)
    r = run([VENV_PY, "-c", "import anima_train_network; print('anima_train_network import OK')"],
            cwd=SD_DIR)
    if r.returncode != 0:
        print()
        print("[!] import 失败 —— 看上面缺哪个模块，然后：")
        print('    "%s" -m pip install -i %s <模块名>' % (VENV_PY, PIP_INDEX))
        return 1
    for label, p in (("Anima 底模", os.path.join(ANIMA_ROOT, "diffusion_models",
                                                "anima-aesthetic-v1.1.safetensors")),
                     ("文本编码器", os.path.join(ANIMA_ROOT, "text_encoders", "qwen_3_06b_base.safetensors")),
                     ("VAE", os.path.join(ANIMA_ROOT, "vae", "qwen_image_vae.safetensors"))):
        print("  %s %-12s %s" % ("✓" if os.path.isfile(p) else "✗", label, p))
    print()
    print("装好了。双击项目根目录的 train_anima.cmd 即可。")
    return 0


if __name__ == "__main__":
    sys.exit(main())
