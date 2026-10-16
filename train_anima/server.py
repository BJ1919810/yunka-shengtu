# -*- coding: utf-8 -*-
"""本机 Anima 风格 LoRA 训练台（web 端）。

链路
    拖入「图片 + 同名 .txt」→ 体检 → 生成 kohya dataset TOML → 本机起
    `anima_train_network.py`（**独立子进程**，关掉网页照跑）→ 轮询日志/进度/显存
    → 训完取 LoRA（可选转 ComfyUI 版）

为什么用 kohya sd-scripts，而不是云卡那套 diffusers 官方脚本
    sd-scripts 从 **v0.10.1 起官方支持 Anima**（`anima_train_network.py`），而且专门
    给低显存做了一整套开关（`--blocks_to_swap` / `--cache_text_encoder_outputs` /
    `--qwen_image_vae_2d` …）⇒ 8GB 的 3070 才有戏。详见 `.skills/83`。

⛔ 本机三条硬约束（都是踩过的，别"优化"掉）
    1. **绝不加 `--xformers`** —— 本机那个 xformers 是给 torch 2.2.2+cpu 编的，
       一加载就警告/报错；用默认的 torch attention（必要时 `--split_attn`）。
    2. **必须 `HF_HUB_OFFLINE=1`** —— 否则收尾 `save_model_card()` 会卡着重试几分钟
       （不报错，只是白等；见 `.skills/70` §9.3 第 5 条）。
    3. **训练跑在独立子进程里**（`cmd /c run.cmd` + `done.rc`）—— 学云卡 `job.sh` 那套：
       网页/服务挂了训练照跑；「结束」判据是 `done.rc`，不是进程还在不在。

用法：双击 `train_anima.cmd`（或 `python train_anima/server.py --port 8791`）
"""
from __future__ import annotations

import base64
import json
import os
import re
import shutil
import subprocess
import sys
import threading
import time
import traceback
import urllib.parse
import zipfile
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer

import pathlib as _pl
_here = _pl.Path(__file__).resolve()
for _r in (_here.parent, *_here.parents):          # 向上找仓库根（含 paths.py）
    if (_r / "paths.py").exists():
        sys.path.insert(0, str(_r))
        break
import paths  # noqa: E402  （路径唯一权威）

HERE = os.path.dirname(os.path.abspath(__file__))
PROJECT = os.path.dirname(HERE)                     # 云卡生图/
# 暂存目录可用环境变量覆盖 ⇒ 测试时能隔离，不碰用户真正在用的 _staging
STAGING = os.environ.get("ANIMA_STAGING") or os.path.join(HERE, "_staging")
JOBS = os.path.join(HERE, "_jobs")                  # 每个 job 一个目录（run.cmd / train.log / done.rc）
OUT_ROOT = os.path.join(HERE, "_out")               # 产物：_out/<输出名>/
STATE = os.path.join(HERE, "_state.json")

# ---- 后端与模型（位置统一由 paths.py 解析，环境变量仍可覆盖）----
VENV_PY = str(paths.ANIMA_VENV_PY)
SD_DIR = str(paths.ANIMA_SD_DIR)
ANIMA_ROOT = str(paths.ANIMA_ROOT)
DIT = os.path.join(ANIMA_ROOT, "diffusion_models", "anima-aesthetic-v1.1.safetensors")
TE = os.path.join(ANIMA_ROOT, "text_encoders", "qwen_3_06b_base.safetensors")
VAE = os.path.join(ANIMA_ROOT, "vae", "qwen_image_vae.safetensors")
TURBO = os.path.join(ANIMA_ROOT, "diffusion_models", "anima-turbo-v1.1.safetensors")

IMG_EXT = (".png", ".jpg", ".jpeg", ".webp", ".bmp")
# 素材放大倍数超过它就该提醒（既有结论：≤1.2 倍最好 —— `.skills/70` §9.1-3 / `.skills/81`）
UPS_WARN = 1.2
OP_LOCK = threading.Lock()

# 浏览器总会自动请求 /favicon.ico；不回就是一条 404，在控制台里看着像"页面出错"。
# 给一个 1×1 透明 PNG（70 字节），顺手把那行噪音消掉。
FAVICON = base64.b64decode(
    "iVBORw0KGgoAAAANSUhEUgAAAAEAAAABCAYAAAAfFcSJAAAADUlEQVR42mP8/5+hHgAHggJ/PzWvWQAAAABJRU5ErkJggg==")

# 素材里的「排版词」——会被触发词一起吸走（.skills/81 §四）
LAYOUT_WORDS = ("竖排", "气泡", "对话框", "分格", "文字", "水印", "签名", "logo", "Logo")


def _ensure_dirs() -> None:
    for d in (STAGING, JOBS, OUT_ROOT):
        os.makedirs(d, exist_ok=True)


# ============================================================ 状态文件
def load_state() -> dict:
    try:
        with open(STATE, encoding="utf-8") as f:
            return json.load(f)
    except Exception:
        return {}


def save_state(st: dict) -> None:
    tmp = STATE + ".tmp"
    with open(tmp, "w", encoding="utf-8") as f:
        json.dump(st, f, ensure_ascii=False, indent=2)
    os.replace(tmp, STATE)


# ============================================================ 环境体检
def _py_ver(py: str, code: str) -> str:
    try:
        r = subprocess.run([py, "-c", code], capture_output=True, text=True,
                           encoding="utf-8", errors="replace", timeout=90)
        return (r.stdout or r.stderr or "").strip().splitlines()[-1] if (r.stdout or r.stderr) else ""
    except Exception as e:
        return "（探测失败：%s）" % type(e).__name__


def env_check() -> dict:
    """后端齐不齐 —— 不齐就别让用户白点「开始训练」。"""
    e: dict = {"ok": True, "items": [], "blockers": []}

    def add(name, ok, detail, blocker=True, fix=""):
        e["items"].append({"name": name, "ok": bool(ok), "detail": detail, "fix": fix})
        if not ok:
            e["ok"] = False
            if blocker:
                e["blockers"].append(name)

    script = os.path.join(SD_DIR, "anima_train_network.py")
    add("sd-scripts", os.path.isfile(script),
        SD_DIR if os.path.isfile(script) else "缺 %s" % script,
        fix="跑一次 `python train_anima/setup_backend.py`（自动 clone + 建 venv）")
    add("训练 venv", os.path.isfile(VENV_PY), VENV_PY,
        fix="跑一次 `python train_anima/setup_backend.py`")
    if os.path.isfile(VENV_PY):
        v = _py_ver(VENV_PY, "import torch,sys;print('torch %s | cuda %s | %s'%("
                             "torch.__version__, torch.cuda.is_available(),"
                             " torch.cuda.get_device_name(0) if torch.cuda.is_available() else 'NO-GPU'))")
        add("torch / CUDA", "cuda True" in v, v)
        b = _py_ver(VENV_PY, "import bitsandbytes;print('bitsandbytes %s'%bitsandbytes.__version__)")
        add("bitsandbytes（AdamW8bit 要用）", "bitsandbytes" in b and "失败" not in b, b,
            blocker=False, fix="venv 里 pip install bitsandbytes")
    for label, p in (("Anima 底模", DIT), ("文本编码器 Qwen3-0.6B", TE), ("VAE", VAE)):
        mb = "%.0f MB" % (os.path.getsize(p) / 2 ** 20) if os.path.isfile(p) else "缺文件"
        add(label, os.path.isfile(p), "%s  %s" % (os.path.basename(p), mb))
    # 显存：本机常态被桌面吃掉不少，训练前值得看一眼
    try:
        r = subprocess.run(["nvidia-smi", "--query-gpu=memory.used,memory.total,memory.free",
                            "--format=csv,noheader"], capture_output=True, text=True, timeout=20)
        e["gpu"] = (r.stdout or "").strip()
    except Exception:
        e["gpu"] = "n/a"
    return e


# ============================================================ 数据集体检
def _img_size(path: str):
    """只读文件头拿宽高（不 import PIL，服务端保持零依赖）。"""
    with open(path, "rb") as f:
        head = f.read(32)
        if head[:8] == b"\x89PNG\r\n\x1a\n":
            import struct
            w, h = struct.unpack(">II", head[16:24])
            return int(w), int(h)
        if head[:2] == b"\xff\xd8":                       # JPEG：扫 SOFn
            f.seek(2)
            while True:
                b = f.read(1)
                while b and b != b"\xff":
                    b = f.read(1)
                marker = f.read(1)
                while marker == b"\xff":
                    marker = f.read(1)
                if not marker:
                    return None
                m = marker[0]
                if m in (0xC0, 0xC1, 0xC2, 0xC3, 0xC5, 0xC6, 0xC7, 0xC9, 0xCA, 0xCB, 0xCD, 0xCE, 0xCF):
                    import struct
                    f.read(3)
                    h, w = struct.unpack(">HH", f.read(4))
                    return int(w), int(h)
                seg = f.read(2)
                if len(seg) < 2:
                    return None
                f.seek(int.from_bytes(seg, "big") - 2, 1)
        if head[:4] == b"RIFF" and head[8:12] == b"WEBP":  # WEBP（VP8X 简化处理）
            import struct
            f.seek(12)
            chunk = f.read(8)
            if chunk[:4] == b"VP8X":
                w = 1 + int.from_bytes(f.read(3), "little")
                h = 1 + int.from_bytes(f.read(3), "little")
                return w, h
    return None


def stage_check(trigger: str = "") -> dict:
    """配对 + 尺寸 + 放大倍数 + caption 前缀 + 排版词。"""
    files = sorted(os.listdir(STAGING)) if os.path.isdir(STAGING) else []
    imgs = [f for f in files if f.lower().endswith(IMG_EXT)]
    txts = [f for f in files if f.lower().endswith(".txt")]
    imgs.sort()
    pairs, no_cap, orphan_txt = [], [], []
    ups, shorts, shape_bad = [], [], []
    bad_prefix, layout_hits = [], []
    bom_caps = []
    for f in imgs:
        stem = os.path.splitext(f)[0]
        cap = stem + ".txt"
        if cap not in txts:
            no_cap.append(f)
        else:
            pairs.append(f)
            try:
                # ⛔ 这里必须**先查 BOM 再解码**：
                #   kohya 读 caption 用的是 `encoding="utf-8"`（**不去 BOM**）⇒ BOM 会**粘在触发词上**：
                #   2026-10-16 用户的训练记录 `ss_tag_frequency` 里就是 `\ufeffjm_style`（15/15 全带 BOM）
                #   ⇒ 训练时学到的触发词是「BOM+jm_style」，出图写 `jm_style` 时**对不上** ⇒ 风格不生效。
                #   ⚠️ 所以**不能**只把读取改成 utf-8-sig 把它藏起来（上一版就是这么错的）。
                with open(os.path.join(STAGING, cap), "rb") as fh:
                    raw = fh.read()
                if raw[:3] == b"\xef\xbb\xbf":
                    bom_caps.append(cap)
                    text = raw[3:].decode("utf-8", "replace").strip()
                else:
                    text = raw.decode("utf-8", "replace").strip()
            except OSError:
                text = ""
            # 触发词比对：忽略大小写 + 允许「触发词后面直接跟逗号/空格/中文逗号」
            if trigger and not text.lower().startswith(trigger.lower()):
                bad_prefix.append(cap)
            hit = [w for w in LAYOUT_WORDS if w in text]
            if hit:
                layout_hits.append("%s：%s" % (cap, "/".join(hit)))
        st = _img_size(os.path.join(STAGING, f))
        if st:
            w, h = st
            short, long_ = min(w, h), max(w, h)
            shorts.append(short)
            ups.append(max(1024.0 / w, 1024.0 / h))          # 不开分桶时被放大到 1024²
            if long_ / max(1, short) > 3:
                shape_bad.append("%s（%dx%d）" % (f, w, h))
    for t in txts:
        if os.path.splitext(t)[0] + ".txt" not in t or (os.path.splitext(t)[0] + ".png") not in imgs:
            if not any(os.path.splitext(i)[0] == os.path.splitext(t)[0] for i in imgs):
                orphan_txt.append(t)
    short_sorted = sorted(shorts)
    med = short_sorted[len(short_sorted) // 2] if short_sorted else 0
    res = {
        "n_img": len(imgs), "n_txt": len(txts), "n_pair": len(pairs),
        "total_bytes": sum(os.path.getsize(os.path.join(STAGING, f)) for f in files),
        "short_med": med, "short_min": short_sorted[0] if short_sorted else 0,
        "short_max": short_sorted[-1] if short_sorted else 0,
        "ups_med": round(sorted(ups)[len(ups) // 2], 2) if ups else 0,
        "ups_max": round(max(ups), 2) if ups else 0,
        "n_up_over_1_2": sum(1 for u in ups if u > UPS_WARN),
        "no_cap": no_cap[:12], "n_no_cap": len(no_cap),
        "orphan_txt": orphan_txt[:8], "n_orphan_txt": len(orphan_txt),
        "bad_prefix": bad_prefix[:8], "n_bad_prefix": len(bad_prefix),
        # ★ BOM 单列一项：kohya 不去 BOM，BOM 会粘在触发词上（2026-10-16 实测踩过）
        "bom_caps": bom_caps[:12], "n_bom": len(bom_caps),
        "layout_hits": layout_hits[:8], "n_layout_hits": len(layout_hits),
        "shape_bad": shape_bad[:8], "n_shape_bad": len(shape_bad),
    }
    w = []
    if not imgs:
        w.append("一张图都没有 —— 把图片和**同名 .txt** 一起拖进来（也可以在提示词里留空）")
    if res["n_no_cap"]:
        w.append("%d 张图没有同名 .txt ⇒ 这几张会走空 caption（等于纯无条件训练），建议补上"
                 % res["n_no_cap"])
    if res["n_bom"]:
        w.append("%d 条 caption 带 **UTF-8 BOM**（记事本/某些导出工具会加）—— ⛔ kohya **不去 BOM**："
                 "BOM 会**粘在触发词上**（实测训练记录 `ss_tag_frequency` 里就是 `\\ufeff%s`），"
                 "而出图时你写的是「%s」⇒ **触发词对不上、风格基本不生效**（只剩底模自己的味道）。"
                 "⇒ 把这些 .txt 重新存成「UTF-8 无 BOM」再训：`python tools/strip_bom.py`"
                 "（只去掉开头 3 字节，内容不动）"
                 % (res["n_bom"], trigger, trigger))
    if trigger and res["n_bad_prefix"]:
        w.append("%d 条 caption 没有以触发词「%s」开头 —— 风格 LoRA 惯例是每条都带（见《训练标注-DeepSeek规则.md》）"
                 % (res["n_bad_prefix"], trigger))
    if res["n_layout_hits"]:
        w.append("%d 条 caption 里有**排版词**（%s…）—— 这些会被触发词一起吸走，以后出图自带分格/气泡（.skills/81 §四）"
                 % (res["n_layout_hits"], "、".join(LAYOUT_WORDS[:4])))
    if res["ups_max"] and res["ups_max"] > UPS_WARN:
        w.append("不开分桶时最大会被放大 %.2f 倍（中位 %.2f）—— 既有结论是**放大 ≤1.2 倍最好**"
                 "（`.skills/70` §9.1-3 / `.skills/81`）⇒ 务必勾「按原图比例分桶」；"
                 "想看落桶明细跑 `python tools/pick_dataset.py <目录> --buckets`"
                 % (res["ups_max"], res["ups_med"]))
    if res["n_shape_bad"]:
        w.append("%d 张极端长宽比（>3:1）—— 就算开分桶也会被裁掉约一半" % res["n_shape_bad"])
    if imgs and len(imgs) > 20:
        w.append("现在 %d 张：风格 LoRA 的甜点区是 **10~15 张**（下限 8），多不等于好（.skills/81 §三）"
                 % len(imgs))
    res["warnings"] = w
    return res


def strip_bom_file(path: str) -> bool:
    """把 caption(.txt) 开头的 UTF-8 BOM 去掉（只删 3 字节，内容一个字节不动）。返回是否真改了。

    为什么必须自动做：kohya 读 caption 用 `encoding="utf-8"`，**不去 BOM** ⇒ BOM 会**粘在触发词上**：
    2026-10-16 实测，用户那版训练记录的 `ss_tag_frequency` 里就是 `\\ufeffjm_style`（**15/15 全带**），
    而出图时写的是干净的触发词 ⇒ **对不上、风格基本不生效**（只剩底模自己的味道）。详见 `.skills/81 §六`。
    ⛔ 非 UTF-8 的文件（如 GBK）**原样不动**，免得把别的编码弄坏。
    """
    try:
        with open(path, "rb") as fh:
            raw = fh.read()
    except OSError:
        return False
    if not raw.startswith(b"\xef\xbb\xbf"):
        return False
    body = raw[3:]
    try:
        body.decode("utf-8")
    except UnicodeDecodeError:
        return False
    try:
        with open(path, "wb") as fh:
            fh.write(body)
    except OSError:
        return False
    return True


def list_files() -> dict:
    files = []
    if os.path.isdir(STAGING):
        for f in sorted(os.listdir(STAGING)):
            p = os.path.join(STAGING, f)
            if os.path.isfile(p):
                files.append({"name": f, "size": os.path.getsize(p)})
    return {"files": files, "n_img": sum(1 for f in files if f["name"].lower().endswith(IMG_EXT)),
            "n_txt": sum(1 for f in files if f["name"].lower().endswith(".txt")),
            "total_bytes": sum(f["size"] for f in files)}


def free_name(name: str) -> str:
    base = os.path.basename(name.replace("\\", "/"))
    base = re.sub(r"[^\w\u4e00-\u9fff.\-]", "_", base) or "unnamed"
    dest = os.path.join(STAGING, base)
    n = 1
    while os.path.exists(dest):
        stem, ext = os.path.splitext(base)
        dest = os.path.join(STAGING, "%s_%d%s" % (stem, n, ext))
        n += 1
    return dest


# ============================================================ kohya 配置与命令
DEFAULTS = {
    "name": "jm_style", "trigger": "jm_style",
    "dit": "aesthetic", "steps": 600, "resolution": 1024, "buckets": True,
    "network_dim": 16, "network_alpha": 16, "seed": 42,
    "learning_rate": 0.0001, "lr_scheduler": "constant", "lr_warmup_steps": 0,
    "optimizer": "AdamW8bit", "batch": 1, "grad_accum": 1, "caption_dropout": 0.1,
    "blocks_to_swap": 16, "gradient_checkpointing": True, "cache_latents": True,
    "cache_te": True, "vae_2d": True, "vae_chunk": 64,
    # 存 checkpoint：每 200 步、留 2 个 —— 照 `.skills/70` §9.2 的参数基线（我原先写 100/3 是偏离）
    "save_every": 200, "save_last": 2, "save_state": True,
    "resume": "", "tensorboard": True,
    "timestep_sampling": "sigmoid", "flow_shift": 1.0, "max_grad_norm": 1.0,
}


def _f(v, default=0.0) -> float:
    try:
        return float(v)
    except (TypeError, ValueError):
        return default


def _i(v, default=0) -> int:
    try:
        return int(float(v))
    except (TypeError, ValueError):
        return default


def sanitize(raw: dict) -> tuple[dict | None, str]:
    o = dict(DEFAULTS)
    o.update({k: v for k, v in (raw or {}).items() if k in DEFAULTS})
    name = re.sub(r"[^\w\-]", "_", str(o["name"]).strip()) or "jm_style"
    o["name"] = name
    o["trigger"] = str(o["trigger"]).strip()
    if o.get("dit") not in ("aesthetic", "turbo"):
        return None, "底模只能选 aesthetic 或 turbo"
    o["steps"] = max(1, _i(o["steps"], 600))
    o["resolution"] = max(256, min(2048, _i(o["resolution"], 1024)))
    o["network_dim"] = max(1, min(256, _i(o["network_dim"], 16)))
    o["network_alpha"] = max(1, min(256, _i(o["network_alpha"], 16)))
    o["batch"] = max(1, min(8, _i(o["batch"], 1)))
    o["grad_accum"] = max(1, min(64, _i(o["grad_accum"], 1)))
    o["blocks_to_swap"] = max(0, min(26, _i(o["blocks_to_swap"], 16)))
    o["save_every"] = max(0, _i(o["save_every"], 100))
    o["save_last"] = max(1, min(50, _i(o["save_last"], 3)))
    o["vae_chunk"] = max(0, min(512, _i(o["vae_chunk"], 64)))
    o["caption_dropout"] = max(0.0, min(1.0, _f(o["caption_dropout"], 0.1)))
    o["learning_rate"] = max(0.0, _f(o["learning_rate"], 1e-4))
    o["lr_warmup_steps"] = max(0, _i(o["lr_warmup_steps"], 0))
    o["flow_shift"] = _f(o["flow_shift"], 1.0)
    o["max_grad_norm"] = _f(o["max_grad_norm"], 1.0)
    # ⛔ 只留**训练 venv 里真能 import 的**优化器（实测：lion_pytorch / prodigyopt 都没装）
    #    —— 下拉里放一个点了就报错的选项，比不放更糟。
    if o["optimizer"] not in ("AdamW8bit", "AdamW", "Adafactor"):
        o["optimizer"] = "AdamW8bit"
    if o["lr_scheduler"] not in ("constant", "constant_with_warmup", "cosine", "linear"):
        o["lr_scheduler"] = "constant"
    if o["timestep_sampling"] not in ("sigmoid", "shift", "uniform", "sigma", "flux_shift"):
        o["timestep_sampling"] = "sigmoid"
    # 续跑：kohya 的 --max_train_steps 也是**总步数**（.skills/70 §9.3-2）
    rs = str(o.get("resume") or "").strip()
    o["resume"] = rs if rs in ("", "latest") or rs.isdigit() or rs.startswith("train_state") else ""
    if not os.path.isdir(STAGING) or not any(f.lower().endswith(IMG_EXT) for f in os.listdir(STAGING)):
        return None, "本机暂存里一张图都没有 —— 先把图片和同名 .txt 拖进来"
    return o, ""


def make_toml(o: dict, path: str) -> str:
    """kohya 的 dataset 配置。⛔ image_dir 用**正斜杠**，反斜杠在 TOML 里要转义、最容易写错。"""
    res = (o["resolution"], o["resolution"])
    lines = [
        "[general]",
        "enable_bucket = %s" % ("true" if o["buckets"] else "false"),
        "bucket_no_upscale = %s" % ("true" if o["buckets"] else "false"),
        "min_bucket_reso = 512",
        "max_bucket_reso = %d" % max(1024, o["resolution"] * 2),
        "resolution = [%d, %d]" % res,
        'caption_extension = ".txt"',
        "shuffle_caption = false",
        "",
        "[[datasets]]",
        "batch_size = %d" % o["batch"],
        "enable_bucket = %s" % ("true" if o["buckets"] else "false"),
        "bucket_no_upscale = %s" % ("true" if o["buckets"] else "false"),
        "",
        "  [[datasets.subsets]]",
        '  image_dir = "%s"' % STAGING.replace("\\", "/"),
        "  num_repeats = 1",
        "  caption_dropout_rate = %.2f" % o["caption_dropout"],
        "",
    ]
    text = "\n".join(lines)
    with open(path, "w", encoding="utf-8", newline="\n") as f:
        f.write(text)
    return text


def build_cmd(o: dict, job_dir: str) -> tuple[list[str], str]:
    """返回 (kohya 参数列表, 输出目录)。"""
    out_dir = os.path.join(OUT_ROOT, o["name"])
    dit = TURBO if o["dit"] == "turbo" else DIT
    logdir = os.path.join(job_dir, "tb")
    a = [
        "--pretrained_model_name_or_path", dit,
        "--qwen3", TE,
        "--vae", VAE,
        "--dataset_config", os.path.join(job_dir, "dataset.toml"),
        "--output_dir", out_dir,
        "--output_name", o["name"],
        "--save_model_as", "safetensors",
        "--network_module", "networks.lora_anima",
        "--network_dim", str(o["network_dim"]),
        "--network_alpha", str(o["network_alpha"]),
        "--network_train_unet_only",          # 只训 DiT（TE 输出被缓存，不训 TE LoRA）
        "--learning_rate", repr(o["learning_rate"]),
        "--lr_scheduler", o["lr_scheduler"],
        "--lr_warmup_steps", str(o["lr_warmup_steps"]),
        "--optimizer_type", o["optimizer"],
        "--max_train_steps", str(o["steps"]),
        "--train_batch_size", str(o["batch"]),
        "--gradient_accumulation_steps", str(o["grad_accum"]),
        "--seed", str(o["seed"]),
        "--mixed_precision", "bf16",
        "--timestep_sampling", o["timestep_sampling"],
        "--discrete_flow_shift", repr(o["flow_shift"]),
        "--max_grad_norm", repr(o["max_grad_norm"]),
        "--max_data_loader_n_workers", "2",   # Windows 下别开 8 个 worker
        "--gradient_checkpointing",
        "--cache_latents",
        "--cache_text_encoder_outputs",
        "--vae_chunk_size", str(o["vae_chunk"]),
        "--save_every_n_steps", str(o["save_every"]) if o["save_every"] else "0",
        # ⛔ kohya 的 `--save_last_n_steps` 是**步数窗口**（"保留最近 N 步内的存档"），不是"留几个文件"！
        #   源码 `library/checkpoint_io.py:get_remove_step_no()`：
        #       remove_step_no = step_no - save_last_n_steps - 1
        #       remove_step_no -= remove_step_no % save_every_n_steps     ← 向下取到间隔的倍数后删除
        #   ⇒ 把它当"个数"直接传是**错的**：`save_every=300 / 留 3` 窗口只有 3 步，
        #     每次存档都把上一档删掉 ⇒ **永远只剩最新一档**
        #     （2026-10-16 实测：1500 步那次只剩 `-step00001500`，早期档全没了；由用户"把600步也加上"引出）。
        #   ⇒ 换算成窗口：要留 M 档（每隔 E 步一档）⇒ 窗口 = (M-1)*E；M=1 ⇒ 0（只留最新档）。
        "--save_last_n_steps",
        str(max(0, (o["save_last"] - 1) * o["save_every"])) if o["save_every"] else str(o["save_last"]),
    ]
    if o["vae_2d"]:
        a.append("--qwen_image_vae_2d")
        a.append("--vae_disable_cache")
    if o["blocks_to_swap"]:
        a += ["--blocks_to_swap", str(o["blocks_to_swap"])]
    if o["save_state"]:
        a.append("--save_state")
    if o["tensorboard"]:
        a += ["--logging_dir", logdir, "--log_with", "tensorboard"]
    if str(o.get("resume") or ""):
        # ⛔ kohya 的 --resume 传**输出目录**（取里面最新的 train_state-*），
        #    并且 --max_train_steps 是**总步数**；填小了会立刻结束（.skills/70 §9.3-2）
        a += ["--resume", out_dir]
    return a, out_dir


def _demo(a: list[str]) -> str:
    """把命令拼成能直接粘进 cmd 的一行（run.cmd 里也存这个）。"""
    def q(s: str) -> str:
        return '"%s"' % s if (" " in s or "\\" in s or ":" in s) else s
    return " ".join(q(x) for x in a)


def make_run_cmd(o: dict, job_dir: str, args: list[str]) -> str:
    """生成 run.cmd：**独立进程**跑训练，收尾写 done.rc（学云卡 job.sh 那套）。"""
    log = os.path.join(job_dir, "train.log")
    done = os.path.join(job_dir, "done.rc")
    lines = [
        "@echo off",
        "chcp 65001 >nul",
        "set PYTHONIOENCODING=utf-8",
        "set PYTHONUNBUFFERED=1",
        # ⛔ 见文件头硬约束 2：不设它，收尾 save_model_card() 会卡着重试几分钟
        "set HF_HUB_OFFLINE=1",
        "set HF_DATASETS_OFFLINE=1",
        'cd /d "%s"' % SD_DIR,
        '"%s" -u anima_train_network.py %s > "%s" 2>&1' % (VENV_PY, " ".join('"%s"' % x for x in args), log),
        # ⛔ 别写成 "...%ERRORLEVEL%..." % done —— `%E` 会被 % 格式化当成浮点格式符（实测报
        #    TypeError: must be real number, not str），这里必须用字符串拼接。
        'echo %ERRORLEVEL% > "' + done + '"',
    ]
    path = os.path.join(job_dir, "run.cmd")
    with open(path, "w", encoding="utf-8", newline="\r\n") as f:     # ⛔ .cmd 必须 CRLF
        f.write("\n".join(lines) + "\n")
    return path


# ============================================================ 起训 / 查状态 / 停
def start_job(raw: dict) -> dict:
    o, err = sanitize(raw)
    if err:
        return {"ok": False, "error": err}
    chk = stage_check(o["trigger"])
    if chk["n_img"] == 0:
        return {"ok": False, "error": "暂存里没有图片"}

    job_id = time.strftime("%m%d-%H%M%S")
    job_dir = os.path.join(JOBS, job_id)
    os.makedirs(job_dir, exist_ok=True)
    make_toml(o, os.path.join(job_dir, "dataset.toml"))
    args, out_dir = build_cmd(o, job_dir)

    if os.path.isdir(out_dir) and not str(o.get("resume") or ""):
        n = 1
        while os.path.isdir(os.path.join(OUT_ROOT, "%s_%s" % (o["name"], job_id[:6]))):
            n += 1
        o["name"] = "%s_%s" % (o["name"], job_id[:6])
        args, out_dir = build_cmd(o, job_dir)      # 名字撞了就换一个，别覆盖上次的 LoRA
    os.makedirs(out_dir, exist_ok=True)

    run_cmd = make_run_cmd(o, job_dir, args)
    with open(os.path.join(job_dir, "command.txt"), "w", encoding="utf-8") as f:
        f.write(_demo([VENV_PY, "-u", "anima_train_network.py"] + args) + "\n")
    with open(os.path.join(job_dir, "opts.json"), "w", encoding="utf-8") as f:
        json.dump({"opts": o, "args": args, "out_dir": out_dir}, f, ensure_ascii=False, indent=2)

    launch_log = open(os.path.join(job_dir, "launch.log"), "wb")
    try:
        # ★ 独立进程：关掉网页/服务照跑 ⇒ 不能让它挂在服务的控制台上。
        # ⛔⛔ 别用 DETACHED_PROCESS(0x8)：2026-10-15 实测，它让 run.cmd 里的 `> train.log`
        #    重定向**失效** —— done.rc 照写、产物照出、退出码照对，但 train.log 恒 0 字节
        #    ⇒ 界面上「日志」永远空白，且不报错（最难发现的一类）。
        # ✅ CREATE_NO_WINDOW(0x08000000)：给它一个**隐藏的**控制台，重定向正常，也不随本进程退出而死。
        CREATE_NO_WINDOW = 0x08000000
        CREATE_NEW_PROCESS_GROUP = 0x00000200
        p = subprocess.Popen(["cmd.exe", "/c", run_cmd], cwd=SD_DIR, stdin=subprocess.DEVNULL,
                             stdout=launch_log, stderr=subprocess.STDOUT,
                             creationflags=CREATE_NO_WINDOW | CREATE_NEW_PROCESS_GROUP,
                             close_fds=True)
    except Exception as e:
        launch_log.close()
        return {"ok": False, "error": "起进程失败：%s: %s" % (type(e).__name__, e)}
    finally:
        try:
            launch_log.close()
        except Exception:
            pass

    st = {"job_id": job_id, "job_dir": job_dir, "out_dir": out_dir, "pid": p.pid,
          "name": o["name"], "started": time.time(), "opts": o, "stopped": False}
    save_state(st)
    time.sleep(1.2)
    return {"ok": True, "job": st, "cmd": _demo([VENV_PY, "-u", "anima_train_network.py"] + args)}


def _alive(pid: int) -> bool:
    try:
        r = subprocess.run(["tasklist", "/FI", "PID eq %d" % pid, "/NH"],
                           capture_output=True, text=True, timeout=20)
        return str(pid) in (r.stdout or "")
    except Exception:
        return False


PROGRESS_RE = re.compile(r"(\d+)/(\d+)\s*\[")
LOSS_RE = re.compile(r"avr_loss=([0-9.]+)")
EPOCH_RE = re.compile(r"epoch\s+(\d+)/(\d+)")


def parse_progress(text: str) -> dict:
    p: dict = {}
    for line in reversed(text.splitlines()):
        if "avr_loss" in line or PROGRESS_RE.search(line):
            m = PROGRESS_RE.search(line)
            if m:
                p["step"], p["total"] = int(m.group(1)), int(m.group(2))
                p["pct"] = round(100.0 * p["step"] / max(1, p["total"]), 1)
            l = LOSS_RE.search(line)
            if l:
                p["loss"] = float(l.group(1))
            break
    for line in reversed(text.splitlines()[-400:]):
        e = EPOCH_RE.search(line)
        if e:
            p["epoch"], p["epochs"] = int(e.group(1)), int(e.group(2))
            break
    if "loss" not in p:
        for line in reversed(text.splitlines()[-400:]):
            m = re.search(r"loss[=: ]+([0-9.]+)", line)
            if m:
                p["loss"] = float(m.group(1))
                break
    return p


def job_status(offset: int) -> dict:
    st = load_state()
    if not st.get("job_id"):
        return {"ok": True, "has_job": False}
    job_dir = st["job_dir"]
    log_path = os.path.join(job_dir, "train.log")
    size = os.path.getsize(log_path) if os.path.isfile(log_path) else 0
    off = max(0, int(offset))
    if off > size:            # 日志被截断/换了 job
        off = 0
    chunk = ""
    if size > off:
        with open(log_path, "rb") as f:
            f.seek(off)
            chunk = f.read().decode("utf-8", "replace")
    alive = _alive(int(st["pid"])) if st.get("pid") else False
    rc = None
    rcp = os.path.join(job_dir, "done.rc")
    if os.path.isfile(rcp):
        try:
            rc = open(rcp, encoding="utf-8", errors="replace").read().strip()
        except OSError:
            rc = "?"
    out_files = []
    if os.path.isdir(st["out_dir"]):
        for f in sorted(os.listdir(st["out_dir"])):
            if f.endswith((".safetensors", ".pt")):
                p = os.path.join(st["out_dir"], f)
                out_files.append({"name": f, "path": p,
                                  "mb": round(os.path.getsize(p) / 2 ** 20, 1)})
    gpu = ""
    try:
        r = subprocess.run(["nvidia-smi", "--query-gpu=memory.used,memory.total,utilization.gpu",
                            "--format=csv,noheader"], capture_output=True, text=True, timeout=15)
        gpu = (r.stdout or "").strip()
    except Exception:
        pass
    return {"ok": True, "has_job": True, "job_id": st["job_id"], "name": st.get("name", ""),
            "out_dir": st["out_dir"], "alive": alive, "rc": rc, "stopped": bool(st.get("stopped")),
            "offset": size, "log": chunk, "elapsed": int(time.time() - st.get("started", time.time())),
            "progress": parse_progress(chunk) if chunk else {}, "gpu": gpu, "out_files": out_files}


def stop_job() -> dict:
    st = load_state()
    if not st.get("job_id"):
        return {"ok": False, "error": "没有任务"}
    pid = int(st.get("pid") or 0)
    if not pid:
        return {"ok": False, "error": "这个任务没记到 pid"}
    if not _alive(pid):
        return {"ok": False, "error": "进程已经退出了（看日志最后几行）"}
    try:
        subprocess.run(["taskkill", "/F", "/T", "/PID", str(pid)],
                       capture_output=True, text=True, timeout=60)
    except Exception as e:
        return {"ok": False, "error": "kill 失败：%s" % e}
    st["stopped"] = True
    save_state(st)
    return {"ok": True, "detail": "已停止（进程树一起杀了）。已存下的 checkpoint 还在 %s。" % st["out_dir"]}


# ============================================================ 产物 / 转换
def list_out() -> dict:
    files = []
    for name in sorted(os.listdir(OUT_ROOT)) if os.path.isdir(OUT_ROOT) else []:
        d = os.path.join(OUT_ROOT, name)
        if not os.path.isdir(d):
            continue
        for f in sorted(os.listdir(d)):
            if f.endswith(".safetensors"):
                p = os.path.join(d, f)
                files.append({"path": p, "name": "%s/%s" % (name, f),
                              "mb": round(os.path.getsize(p) / 2 ** 20, 1)})
    return {"files": files, "dir": OUT_ROOT}


def convert_lora(src: str) -> dict:
    """sd-scripts 版 → ComfyUI 版（networks/convert_anima_lora_to_comfy.py）。"""
    src = os.path.abspath(src)
    # 路径边界要按「目录 + 分隔符」比：光 startswith(_out) 会把 `_outEvil/x` 也放进来；
    # normcase 是为了 Windows 的大小写不敏感。
    out_root = os.path.normcase(os.path.abspath(OUT_ROOT)) + os.sep
    if not os.path.normcase(src).startswith(out_root):
        return {"ok": False, "error": "只允许转换 _out/ 下的文件"}
    if not os.path.isfile(src):
        return {"ok": False, "error": "文件不存在：%s" % src}
    script = os.path.join(SD_DIR, "networks", "convert_anima_lora_to_comfy.py")
    if not os.path.isfile(script):
        return {"ok": False, "error": "找不到转换脚本：%s" % script}
    stem, ext = os.path.splitext(src)
    dst = stem + "_comfy" + ext
    # ⛔ 别写成 [VENV_PY, script, ...]（脚本路径直跑）：Python 只会把 `networks/` 放进 sys.path，
    #    脚本里的 `import library.model_io` 立刻 ModuleNotFoundError（实测：
    #    "No module named 'library'"）⇒ 转换按钮看着有、点了就报错。
    # ✅ 正解：-m 从仓库根作为包导入（networks/ 没 __init__.py，靠 PEP420 命名空间包也能跑），
    #    再加 PYTHONPATH 兜一层。
    env = dict(os.environ)
    env["PYTHONPATH"] = SD_DIR + os.pathsep + env.get("PYTHONPATH", "")
    r = subprocess.run([VENV_PY, "-m", "networks.convert_anima_lora_to_comfy", src, dst],
                       cwd=SD_DIR, capture_output=True, text=True, env=env,
                       encoding="utf-8", errors="replace", timeout=900)
    if not os.path.isfile(dst):
        return {"ok": False, "error": "转换失败", "log": (r.stdout or "")[-1500:] + (r.stderr or "")[-1500:]}
    return {"ok": True, "dst": dst, "mb": round(os.path.getsize(dst) / 2 ** 20, 1),
            "log": (r.stdout or "")[-600:]}


# ============================================================ HTTP
class Handler(BaseHTTPRequestHandler):
    server_version = "TrainAnima/1.0"
    protocol_version = "HTTP/1.1"

    def log_message(self, fmt, *args):
        pass

    def _body(self) -> bytes:
        n = int(self.headers.get("Content-Length") or 0)
        buf = bytearray()
        while len(buf) < n:
            c = self.rfile.read(min(1 << 20, n - len(buf)))
            if not c:
                break
            buf += c
        return bytes(buf)

    def _json(self, obj, code=200):
        b = json.dumps(obj, ensure_ascii=False).encode("utf-8")
        self.send_response(code)
        self.send_header("Content-Type", "application/json; charset=utf-8")
        self.send_header("Content-Length", str(len(b)))
        self.send_header("Cache-Control", "no-store")
        self.end_headers()
        self.wfile.write(b)

    def _html(self):
        with open(os.path.join(HERE, "index.html"), "rb") as f:
            body = f.read()
        self.send_response(200)
        self.send_header("Content-Type", "text/html; charset=utf-8")
        self.send_header("Content-Length", str(len(body)))
        self.send_header("Cache-Control", "no-store")
        self.end_headers()
        self.wfile.write(body)

    def _q(self):
        return {k: v[0] for k, v in urllib.parse.parse_qs(urllib.parse.urlparse(self.path).query).items()}

    def do_GET(self):                                  # noqa: N802
        path = urllib.parse.urlparse(self.path).path
        try:
            if path in ("/", "/index.html"):
                return self._html()
            if path == "/favicon.ico":
                self.send_response(200)
                self.send_header("Content-Type", "image/png")
                self.send_header("Content-Length", str(len(FAVICON)))
                self.send_header("Cache-Control", "max-age=86400")
                self.end_headers()
                return self.wfile.write(FAVICON)
            if path == "/api/env":
                return self._json(env_check())
            if path == "/api/stage/list":
                d = list_files()
                d["check"] = stage_check(self._q().get("trigger", ""))
                return self._json(d)
            if path == "/api/defaults":
                return self._json(DEFAULTS)
            if path == "/api/job":
                off = self._q().get("offset", "0")
                return self._json(job_status(int(off) if off.isdigit() else 0))
            if path == "/api/out":
                return self._json(list_out())
            return self._json({"error": "no such endpoint: %s" % path}, 404)
        except Exception as e:
            traceback.print_exc()
            return self._json({"ok": False, "error": "%s: %s" % (type(e).__name__, e)}, 500)

    def do_POST(self):                                 # noqa: N802
        path = urllib.parse.urlparse(self.path).path
        q = self._q()
        try:
            if path == "/api/stage/put":               # 裸字节上传（不手搓 multipart）
                name = q.get("name") or "unnamed.png"
                body = self._body()
                if not body:
                    return self._json({"ok": False, "error": "空文件：%s" % name}, 400)
                with OP_LOCK:
                    dest = free_name(name)
                    with open(dest, "wb") as f:
                        f.write(body)
                    # ★ 上传即去 BOM：kohya 不去 BOM ⇒ BOM 会粘在触发词上（详见 .skills/81 §六）
                    stripped = strip_bom_file(dest) if dest.lower().endswith(".txt") else False
                return self._json({"ok": True, "saved": os.path.basename(dest), "bytes": len(body),
                                   "bom_stripped": bool(stripped)})
            if path == "/api/stage/zip":
                body = self._body()
                if not body:
                    return self._json({"ok": False, "error": "空 zip"}, 400)
                tmp = os.path.join(HERE, "_incoming.zip")
                with open(tmp, "wb") as f:
                    f.write(body)
                n, nbom = 0, 0
                with zipfile.ZipFile(tmp) as zf:
                    for info in zf.infolist():
                        nm = info.filename.replace("\\", "/")
                        if info.is_dir() or "__MACOSX" in nm or "/._" in nm:
                            continue
                        base = os.path.basename(nm)
                        if not base or base.startswith("."):
                            continue
                        if os.path.splitext(base)[1].lower() not in IMG_EXT + (".txt",):
                            continue
                        dest = free_name(base)
                        with zf.open(info) as src, open(dest, "wb") as dst:
                            shutil.copyfileobj(src, dst)
                        if dest.lower().endswith(".txt") and strip_bom_file(dest):
                            nbom += 1
                        n += 1
                os.remove(tmp)
                return self._json({"ok": True, "unpacked": n, "bom_stripped": nbom})
            if path == "/api/stage/strip-bom":
                # 兜底：把**已经在暂存里**的 caption 去 BOM（上传前就躺在里面的、或别处塞进来的）
                hit = []
                with OP_LOCK:
                    for f in sorted(os.listdir(STAGING)):
                        p = os.path.join(STAGING, f)
                        if f.lower().endswith(".txt") and os.path.isfile(p) and strip_bom_file(p):
                            hit.append(f)
                return self._json({"ok": True, "stripped": len(hit), "files": hit[:20]})
            if path == "/api/stage/clear":
                with OP_LOCK:
                    shutil.rmtree(STAGING, ignore_errors=True)
                    os.makedirs(STAGING, exist_ok=True)
                return self._json({"ok": True})
            if path == "/api/stage/delete":
                body = json.loads(self._body() or b"{}")
                killed = []
                for nm in body.get("names") or []:
                    p = os.path.join(STAGING, os.path.basename(str(nm)))
                    if os.path.isfile(p):
                        os.remove(p)
                        killed.append(os.path.basename(p))
                return self._json({"ok": True, "deleted": killed})
            if path == "/api/job/start":
                return self._json(start_job(json.loads(self._body() or b"{}")))
            if path == "/api/job/stop":
                return self._json(stop_job())
            if path == "/api/convert":
                body = json.loads(self._body() or b"{}")
                return self._json(convert_lora(str(body.get("src") or "")))
            return self._json({"error": "no such endpoint: %s" % path}, 404)
        except Exception as e:
            traceback.print_exc()
            return self._json({"ok": False, "error": "%s: %s" % (type(e).__name__, e)}, 500)


def main() -> None:
    _ensure_dirs()
    port = 8791
    if "--port" in sys.argv:
        port = int(sys.argv[sys.argv.index("--port") + 1])
    open_browser = "--open" in sys.argv
    print("=" * 66)
    print("  本机 Anima 风格 LoRA 训练台")
    print("=" * 66)
    print("  sd-scripts : %s%s" % (SD_DIR, "" if os.path.isfile(
        os.path.join(SD_DIR, "anima_train_network.py")) else "   ⛔ 缺（先跑 setup_backend.py）"))
    print("  venv       : %s%s" % (VENV_PY, "" if os.path.isfile(VENV_PY) else "   ⛔ 缺"))
    print("  暂存       : %s" % STAGING)
    print("  产物       : %s" % OUT_ROOT)
    print()
    print("  打开        http://127.0.0.1:%d" % port)
    print("  关掉这个窗口 = 关掉网页服务；**正在跑的训练不受影响**。")
    print("=" * 66)
    srv = ThreadingHTTPServer(("127.0.0.1", port), Handler)
    if open_browser:
        import webbrowser
        threading.Timer(1.2, lambda: webbrowser.open("http://127.0.0.1:%d" % port)).start()
    try:
        srv.serve_forever()
    except KeyboardInterrupt:
        print("\n[exit] 已停止")


if __name__ == "__main__":
    main()
