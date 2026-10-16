# -*- coding: utf-8 -*-
"""训练 web 端 —— 拖文件 → 调参数 → 点按钮就开训（Qwen-Image-2.1 LoRA）。

★ 核心诉求：「点完按钮，云卡那边就开跑；本机网页关掉也照样跑。」
  做法是远端用 `setsid` 另起一个**新会话**跑 job.sh，且 stdin/stdout/stderr
  全部重定向到文件或 /dev/null。SSH 通道有没有断，对它是无所谓的 ——
  本机这个网页只是「发令枪 + 看板」，随时关掉都行。
  ⚠️ 要停下来只能点网页上的「停止训练」（或去云端 kill），因为已经没人管着它了。

为什么用 stdlib 的 http.server 而不是 FastAPI/Flask：
  项目 requirements 只有 paramiko / Pillow / numpy，本机 venv 里没装 web 框架。
  零新依赖 ⇒ 双击 train_web.cmd 就能用，不需要先 pip install 一堆东西。

启动：
    python train_web/server.py            # 默认 127.0.0.1:7870
    python train_web/server.py 8888       # 换端口

⚠️ 只绑 127.0.0.1 —— 这台机器上的云端口令不能被局域网随便访问。
"""
from __future__ import annotations

import json
import os
import re
import shutil
import sys
import threading
import time
import traceback
import urllib.parse
import zipfile
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(HERE)
sys.path.insert(0, ROOT)
sys.path.insert(0, os.path.join(ROOT, "remote"))

import cloud_sh                      # 复用已有的 SSH 样板（含 MaxStartups 退避）
import prepare_dataset as ds         # 复用体检逻辑：本机 / 云端是同一份文件

DEFAULT_PORT = 7870
HOST = "127.0.0.1"

STAGING = os.path.join(HERE, "_staging")        # 本机暂存（拖进来的原图 + txt）
DOWNLOADS = os.path.join(HERE, "_downloads")    # 训完下载回来的 LoRA
STATE = os.path.join(HERE, "_state.json")       # 当前/最近一次 job 的元信息

# ---- 云端固定路径（改这里就够，见 .skills/70-训练环境与权重加载.md）----
R_TRAIN = "/root/autodl-tmp/lora-train"
R_DATA = "/root/autodl-tmp/dataset"
R_OUT_ROOT = "/root/autodl-tmp/lora-out"
R_RUNS = R_TRAIN + "/runs"
R_INCOMING = "/root/autodl-tmp/_incoming"
R_PY = "/root/autodl-tmp/train-env/bin/python"
R_ENV = "/root/autodl-tmp/train-env"          # ★ 注意：job.sh 里用的是"目录 + /bin/python"
                                             #   （2026-10-14：这里曾误传 R_PY，拼成 .../python/bin/python）
R_MODEL = "/root/autodl-tmp/Qwen-Image-2.1"
R_DIT = "/root/autodl-tmp/ComfyUI/models/diffusion_models/qwen_image_2.1_bf16.safetensors"
R_TE = "/root/autodl-tmp/ComfyUI/models/text_encoders/qwen3vl_8b_bf16.safetensors"

for _d in (STAGING, DOWNLOADS):
    os.makedirs(_d, exist_ok=True)

# 串行化「改云端状态」的操作：同时点两下按钮会互相踩。
OP_LOCK = threading.Lock()


# ============================================================ SSH
class SSH:
    """复用一个 paramiko 连接 + 互斥锁。

    ⚠️ 不用 cloud_sh.connect 的 sys.exit 行为 —— 那是给命令行写的，
       在 web 服务里抛 SystemExit 会把请求线程搞死还不给前端任何信息。
    """

    def __init__(self) -> None:
        self._cli = None
        self._lock = threading.Lock()

    def _get(self):
        if self._cli is None:
            cfg = cloud_sh.load_cfg()
            if not cfg["host"] or not cfg["pass"]:
                raise RuntimeError(
                    "cloud-comfy.config.json 里缺 ssh_host / ssh_password（或在环境变量里设）")
            try:
                self._cli = cloud_sh.connect(cfg, retries=3)
            except SystemExit as e:                      # cloud_sh 用 sys.exit 报错
                raise RuntimeError(str(e.code if isinstance(e.code, str) else e))
        return self._cli

    def reset(self) -> None:
        try:
            if self._cli is not None:
                self._cli.close()
        except Exception:
            pass
        self._cli = None

    def run(self, script: str, timeout: float = 600) -> tuple[str, str, int]:
        with self._lock:
            for attempt in (0, 1):
                try:
                    return cloud_sh.run(self._get(), script, timeout=timeout)
                except Exception:
                    self.reset()
                    if attempt:
                        raise
                    # ⚠️ 复用的连接放久了会被服务端断掉（实测 "Socket exception: 10054"），
                    #    轮询中间隔几分钟再点按钮就会撞上。重连一次即可，别把 500 丢给用户。
                    time.sleep(0.4)

    def _sftp(self, fn):
        with self._lock:
            for attempt in (0, 1):
                cli = self._get()
                try:
                    sftp = cli.open_sftp()
                    try:
                        return fn(sftp)
                    finally:
                        sftp.close()
                except Exception:
                    self.reset()
                    if attempt:
                        raise
                    time.sleep(0.4)

    def put(self, local: str, remote: str) -> int:
        def _do(sftp):
            sftp.put(local, remote)
            return sftp.stat(remote).st_size
        return self._sftp(_do)          # type: ignore[return-value]

    def get(self, remote: str, local: str) -> None:
        def _do(sftp):
            sftp.get(remote, local)
        self._sftp(_do)


SSH_CLIENT = SSH()


# ============================================================ 小工具
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


def parse_sections(out: str) -> dict:
    """把 `### KEY` 分段的输出解析成 dict（cloud_service.py 同款做法）。"""
    parts: dict[str, list[str]] = {}
    cur = None
    for line in out.splitlines():
        if line.startswith("### "):
            cur = line[4:].strip()
            parts[cur] = []
        elif cur is not None:
            parts[cur].append(line)
    return {k: "\n".join(v).strip() for k, v in parts.items()}


def stage_list() -> dict:
    """本机暂存目录的文件清单。"""
    if not os.path.isdir(STAGING):
        return {"n_img": 0, "n_txt": 0, "files": [], "bytes": 0}
    files = sorted(f for f in os.listdir(STAGING) if os.path.isfile(os.path.join(STAGING, f)))
    img_ext = (".png", ".jpg", ".jpeg", ".webp", ".bmp")
    imgs = [f for f in files if os.path.splitext(f)[1].lower() in img_ext]
    txts = [f for f in files if f.lower().endswith(".txt")]
    total = sum(os.path.getsize(os.path.join(STAGING, f)) for f in files)
    return {
        "n_img": len(imgs),
        "n_txt": len(txts),
        "total_bytes": total,          # ⚠️ 别叫 bytes：上传接口里 bytes 是"本次文件大小"，同名会看错
        "files": [{"name": f, "size": os.path.getsize(os.path.join(STAGING, f))}
                  for f in files[:400]],
    }


def free_name(base: str) -> str:
    """暂存目录里取一个不重名的文件名（同名自动加 __dupN）。"""
    base = os.path.basename(base).strip() or "unnamed"
    safe = re.sub(r"[^\w.\-\u4e00-\u9fff]+", "_", base)
    target = os.path.join(STAGING, safe)
    if not os.path.exists(target):
        return os.path.join(STAGING, safe)
    stem, ext = os.path.splitext(safe)
    k = 1
    while True:
        t = os.path.join(STAGING, "%s__dup%d%s" % (stem, k, ext))
        if not os.path.exists(t):
            return t
        k += 1


def make_zip(src_dir: str, zip_path: str) -> int:
    """把暂存目录打包（ZIP_STORED：PNG 已经压过了，再压只是白费 CPU）。"""
    n = 0
    with zipfile.ZipFile(zip_path, "w", zipfile.ZIP_STORED) as zf:
        for f in sorted(os.listdir(src_dir)):
            p = os.path.join(src_dir, f)
            if os.path.isfile(p):
                zf.write(p, arcname=f)
                n += 1
    return n


# ============================================================ job.sh 生成
JOB_SH_HEAD = """#!/usr/bin/env bash
# ⚠️ 本文件由 train_web 自动生成（{stamp}）—— 参数以网页为准，重新点按钮会覆盖。
# 手动改这里的默认值**不会**影响网页表单；要改默认值请改 train_web/server.py。
#
# 脱离 SSH 会话的关键在启动方式（见 server.py 的 start_job）：
#   setsid bash job.sh > train.log 2>&1 < /dev/null &
# 本脚本自己把 $$ 写进 job.pid，结束时把退出码写进 done.rc。
set -uo pipefail

JOB_DIR="$(cd "$(dirname "$0")" && pwd)"
echo $$ > "$JOB_DIR/job.pid"
rm -f "$JOB_DIR/done.rc"
trap 'echo 143 > "$JOB_DIR/done.rc"; exit 143' TERM INT

ENV={env}
TRAIN={train}
MODEL={model}
DATA={data}
OUT={out}

# 系统盘只有 ~10G ⇒ 缓存全部落到数据盘
export PIP_CACHE_DIR=/root/autodl-tmp/.pipcache
export HF_HOME=/root/autodl-tmp/hf
export HF_DATASETS_CACHE=/root/autodl-tmp/hf/datasets
export TMPDIR=/root/autodl-tmp/tmp
export PYTORCH_ALLOC_CONF=expandable_segments:True
# ⚠️ 海外 Hub 不通：**必须**设这个。训练跑完后脚本会无条件调 save_model_card()，
#    它内部要 HEAD 一次 huggingface.co 看 repo 是否存在 ⇒ 不设就会一直重试、白等几分钟
#    （2026-10-14 实测：20 步训练早跑完了，任务却因为它在重试而挂着不结束）。
#    只影响 huggingface_hub，不动本地加载路径（模型都是本地目录）。
export HF_HUB_OFFLINE=1
export HF_HUB_DISABLE_TELEMETRY=1
mkdir -p "$OUT" "$HF_HOME" "$TMPDIR"

# 前置检查：训练用的 python 找不到就立刻停，别等 python 命令报一句看不懂的错
if [ ! -x "$ENV/bin/python" ]; then
  echo "[x] 找不到训练用的 python：$ENV/bin/python"
  echo "    检查 server.py 的 R_ENV（应该是 venv 目录，不是 venv/bin/python）"
  echo 3 > "$JOB_DIR/done.rc"; exit 3
fi

echo "===================== 参数 ====================="
sed -n '/^# ---- 参数开始 ----/,/^# ---- 参数结束 ----/p' "$JOB_DIR/job.sh"
echo "================================================"
echo

echo "[1/3] 数据集体检"
"$ENV/bin/python" "$TRAIN/check_dataset.py" "$DATA" || true
echo
echo "[2/3] 图片 + 同名 txt → metadata.jsonl"
"$ENV/bin/python" "$TRAIN/make_metadata.py" "$DATA" || true
echo
echo "[3/3] 开始训练（Ctrl+C 无效，要停请用网页上的「停止训练」）"
cd "$TRAIN"
"""


# ★ 训完自动关机（2026-10-15）—— 由 build_job_sh 按需拼进脚本收尾
#   设计要点：
#   ① 用 `shutdown -h +1` **排程**（1 分钟后），不用 `( sleep 60; shutdown ) &`：
#      排程交给 init，不依赖本脚本进程存活；而且**能被 `shutdown -c` 取消**。
#   ② 一定要 `-h`（halt/poweroff）。⚠️ 裸 `shutdown` 在某些系统上等同于 `+1` 关机，
#      但语义不明确，写清楚更安全（2026-10-15 我用 `shutdown --help` 就误关了机）。
#   ③ 先 echo 一段，让 train.log 里留有"要关机了"的痕迹，方便事后排查为什么断连。
#   ④ 关机不影响数据：数据盘 /root/autodl-tmp 关机后仍在（同一实例）。
SHUTDOWN_BLOCK = """
# ---- 训完自动关机（网页上勾了「训完自动关机」）----
if command -v shutdown >/dev/null 2>&1; then
  echo "[shutdown] 训练已结束（rc=$rc），1 分钟后关机。"
  echo "[shutdown] 想取消：立刻 SSH 进来执行  shutdown -c"
  echo "[shutdown] 数据盘 /root/autodl-tmp 不受影响；要取 LoRA 请重新开机。"
  # ⚠️ 用 nohup + 子 shell，避免 shutdown 的排程被本脚本退出带走
  ( nohup shutdown -h +1 >/dev/null 2>&1 & ) || shutdown -h +1 || true
else
  echo "[shutdown] ⚠️ 找不到 shutdown 命令，跳过自动关机。"
fi
"""


def build_job_sh(o: dict, job_id: str) -> str:
    """把网页表单变成一份自解释的 shell 脚本。

    ★ 这里**显式写全**所有参数，不依赖 train_qwen21.sh 的默认值 ——
      这样「网页上看到的」就是「实际跑的」，日志头部也会原样打印出来。
      代价：改默认值要动 server.py 的 DEFAULTS（两处不同步的话以网页为准）。
    """
    # 演练模式不要污染正式的产物目录：单独丢进 /lora-out/_dry_<jobid>/
    if o.get("dry"):
        out_dir = R_OUT_ROOT + "/_dry_" + job_id
    else:
        out_dir = R_OUT_ROOT + "/" + o["name"]

    t = JOB_SH_HEAD.format(
        stamp=time.strftime("%F %T"), env=R_ENV, train=R_TRAIN,
        model=R_MODEL, data=R_DATA, out=out_dir)

    args = [
        '  --pretrained_model_name_or_path "$MODEL"',
        '  --dataset_name "$DATA"',
        "  --image_column image",
        "  --caption_column text",
        "  --instance_prompt %s" % sh(o["trigger"]),
        '  --output_dir "$OUT"',
        "  --resolution %d" % int(o["resolution"]),
        "  --train_batch_size %d" % int(o["batch"]),
        "  --gradient_accumulation_steps %d" % int(o["grad_accum"]),
        "  --rank %d" % int(o["rank"]),
        "  --lora_alpha %d" % int(o["lora_alpha"]),
    ]
    # ★ LoRA 作用面（2026-10-14）：只挂 4 个 attn 是脚本默认；加 img_mlp 三层后
    #   目标 128→224，与官方加速 LoRA 同构（见 .skills/79）。空 = 不传，走脚本默认。
    if o.get("lora_layers"):
        args.append("  --lora_layers %s" % sh(o["lora_layers"]))
    args += [
        "  --lora_dropout %s" % num(o["lora_dropout"]),
        "  --learning_rate %s" % num(o["learning_rate"]),
        "  --lr_scheduler %s" % o["lr_scheduler"],
        "  --lr_warmup_steps %d" % int(o["warmup"]),
        "  --max_train_steps %d" % int(o["steps"]),
        "  --checkpointing_steps %d" % int(o["ckpt_steps"]),
        "  --checkpoints_total_limit %d" % int(o["ckpt_limit"]),
        "  --caption_dropout %s" % num(o["caption_dropout"]),
        "  --mixed_precision %s" % o["mixed_precision"],
        "  --seed %d" % int(o["seed"]),
        "  --report_to tensorboard",
    ]
    if o["offload"]:
        args.append("  --offload")
    if o["gradient_checkpointing"]:
        args.append("  --gradient_checkpointing")
    if o["use_8bit_adam"]:
        args.append("  --use_8bit_adam")
    if o["cache_latents"]:
        args.append("  --cache_latents")
    if o.get("bnb"):
        args.append('  --bnb_quantization_config_path "$TRAIN/bnb_nf4.json"')
    # ★ 续跑（2026-10-14）：值已由 sanitize_opts 限成 latest / checkpoint-N。
    #   官方脚本会拼 output_dir + basename(值)，所以这里给什么都能找到（前提是
    #   start_job 没把 out_dir 改名 —— 那边有配套处理，见 .skills/78）。
    if o.get("resume"):
        args.append("  --resume_from_checkpoint %s" % sh(o["resume"]))
    # ★ 分桶（2026-10-14）：开了之后**小图分到小桶，不再上采样**。
    #   不开时：所有图被 BILINEAR 放大到 resolution² 再 RandomCrop —— 素材短边
    #   541 的图要放大 1.9 倍、还被裁掉 30%，线稿直接发虚（实测 35/38 张中招）。
    #   开了之后：bucket 上限 = 图片自身分辨率 ⇒ 不造假细节，但训练边长会变小。
    #   ⚠️ 开了就**忽略 --resolution**（帮手里写明了），所以只传开关不传 resolution 也行。
    if o.get("buckets"):
        args.append("  --use_aspect_ratio_buckets")

    # ⚠️ 续行符只加在「非最后一行」——否则 rc=$? 会被粘进 python 的命令行里
    joinline = " \\\n".join(args)

    if o.get("dry"):
        # 演练模式：不加载那 31G 权重，只把「体检 → metadata → 开跑 → 出日志 → 收尾」
        # 整条链路走一遍。第一次用这个页面时先跑它，确认流程通了再上真训练。
        train_block = '''# ---- 演练模式：不真训练，只走流程（不烧钱）----
echo "[dry] 演练：模拟 40 步，约 80 秒。这一轮不会加载模型、不会出 LoRA。"
for i in $(seq 1 40); do
  printf 'Steps: %3d%%|%s| %d/40 [00:0%d<00:00,  1.00it/s]\\r' \\
    $((i * 100 / 40)) "==========" $i $((i / 4))
  printf '{"loss": 0.%04d, "lr": 0.0001, "epoch": 0}\\n' $((999 - i * 7))
  sleep 2
done
echo
echo "[dry] 演练结束。上面 40 行是假的 loss —— 现在可以回网页看看产物/停止按钮。"
'''
    else:
        train_block = '"$ENV/bin/python" -u run_train_qwen21.py \\\n' + joinline + "\n"

    # ★ 训完自动关机：只在勾了、且**非演练模式**时挂上（演练不烧钱，没必要关）
    shutdown_block = SHUTDOWN_BLOCK if (o.get("shutdown") and not o.get("dry")) else ""

    body = ('# ---- 参数开始 ----\n'
            + train_block
            + '# ---- 参数结束 ----\n'
            + """rc=$?

echo
echo "===================== 训练结束 rc=$rc ====================="
if [ "$rc" -ne 0 ]; then
  echo "（非 0 退出：常见原因参见 .skills/70 §二/§八 —— 无卡模式被 OOM / 显存不够 / 无 CUDA）"
  tail -n 40 "$JOB_DIR/train.log" 2>/dev/null || true
fi
ls -la "$OUT" 2>/dev/null || true
echo "$rc" > "$JOB_DIR/done.rc"
""" + shutdown_block + """
exit "$rc"
""")
    return t + body


def sh(s: str) -> str:
    """单引号包裹，只防单引号 —— 用于触发词这种用户可控字符串。"""
    return "'" + str(s).replace("'", "'\"'\"'") + "'"


def num(v) -> str:
    """浮点参数：保证长得像数字，避免有人往表单里塞奇怪东西。"""
    f = float(v)
    return ("%f" % f).rstrip("0").rstrip(".") or "0"


# ============================================================ 接口实现
def api_ssh_check() -> dict:
    """云卡体检 —— 一次 SSH 拿齐，避免多次往返。

    ⚠️⛔ 判断「有没有训练在跑」**不能用 `pgrep -af`**（2026-10-14 踩）：
       这条 SSH 命令自己的 bash 命令行里就含 "run_train_qwen21.py" 之类的字样，
       pgrep 会匹配到自己 ⇒ 永远返回「有训练在跑」。
       `[r]un_...` 那个方括号技巧在这里**也救不了**：只要脚本里还有**别处**
       （比如 `ls .../run_train_qwen21.py`）出现过裸字符串就照样自匹配。
       ⇒ 改成扫 /proc 读 cmdline，并显式跳过 `$$`（就是执行本脚本的 bash 自己）。
    """
    script = r"""
has_proc() {
  for d in /proc/[0-9]*; do
    p=${d#/proc/}
    [ "$p" = "$$" ] && continue
    c=$(tr '\0' ' ' < "$d/cmdline" 2>/dev/null) || continue
    case "$c" in *"$1"*) return 0;; esac
  done
  return 1
}
echo "### HOST"; hostname
echo "### MEM"; cat /sys/fs/cgroup/memory.max 2>/dev/null || echo 0
echo "### GPU"; nvidia-smi --query-gpu=name,memory.used,memory.total,utilization.gpu --format=csv,noheader 2>/dev/null || echo "no-gpu"
echo "### DISK"; df -h /root/autodl-tmp 2>/dev/null | tail -1
echo "### ENV"; ls -d %s 2>/dev/null || echo "MISSING"
echo "### SCRIPT"; ls %s/train_dreambooth_lora_qwenimage21.py %s/run_train_qwen21.py 2>/dev/null || echo "MISSING"
echo "### MODEL"; ls -l %s 2>/dev/null | awk '{print $5}' || true
echo "### TE"; ls -l %s 2>/dev/null | awk '{print $5}' || true
echo "### HF"; ls -d %s/vae 2>/dev/null || echo "MISSING"
echo "### DATA"; ls -A %s 2>/dev/null | wc -l
echo "### PROCS"; has_proc "train_dreambooth_lora_qwenimage21.py" && echo "yes" || echo "none"
echo "### COMFY"; has_proc "main.py --port" && echo "yes" || echo "none"
""" % (R_PY, R_TRAIN, R_TRAIN, R_DIT, R_TE, R_MODEL, R_DATA)

    out, err, rc = SSH_CLIENT.run(script, timeout=90)
    s = parse_sections(out)

    def _int(v: str) -> int:
        try:
            return int(str(v).strip().split()[0])
        except Exception:
            return 0

    mem = _int(s.get("MEM", "0"))
    info = {
        "host": s.get("HOST", "?"),
        "mem_gb": round(mem / 1073741824.0, 1) if mem else 0,
        "has_gpu": s.get("GPU", "").lower() not in ("", "no-gpu"),
        "gpu": s.get("GPU", "n/a"),
        "disk": s.get("DISK", ""),
        "env_ok": "MISSING" not in (s.get("ENV") or "MISSING"),
        "script_ok": "MISSING" not in (s.get("SCRIPT") or "MISSING"),
        "model_ok": bool(_int(s.get("MODEL", "0"))),
        "model_gb": round(_int(s.get("MODEL", "0")) / 1073741824.0, 1),
        "te_ok": bool(_int(s.get("TE", "0"))),
        "hf_ok": "MISSING" not in (s.get("HF") or "MISSING"),
        "remote_dataset_count": _int(s.get("DATA", "0")),
        "training_now": (s.get("PROCS") or "none") != "none",
        "comfy_running": (s.get("COMFY") or "none") != "none",
        "rc": rc,
    }
    # 前置条件：无卡模式内存只有 2G，加载权重必被 OOM Killer 秒杀（见 .skills/70 §二）
    problems = []
    if not info["has_gpu"]:
        problems.append("没有 GPU —— 云卡大概处于**无卡模式**，训练起不来（去控制台切有卡模式）")
    if info["mem_gb"] and info["mem_gb"] < 30:
        problems.append("容器内存只有 %.1f G —— 无卡模式（有卡模式应该是 62G）" % info["mem_gb"])
    if not info["env_ok"]:
        problems.append("训练 venv 不在：%s" % R_PY)
    if not info["script_ok"]:
        problems.append("训练脚本不在：%s" % R_TRAIN)
    if not info["model_ok"]:
        problems.append("DiT 权重不在：%s" % R_DIT)
    if not info["te_ok"]:
        problems.append("文本编码器不在：%s" % R_TE)
    if not info["hf_ok"]:
        problems.append("伪 HF 目录不全（缺 vae）：%s" % R_MODEL)
    info["problems"] = problems
    info["ok"] = not problems
    if err.strip():
        info["stderr"] = err.strip()[-2000:]
    return info


def api_dataset_push(clear: bool, trigger: str) -> dict:
    """本机暂存 → 打包 → 上传 → 云端解包 + 体检。"""
    files = stage_list()
    if files["n_img"] == 0:
        return {"ok": False, "error": "暂存目录里一张图都没有（先把图片和同名 .txt 拖进来）"}

    job_id = time.strftime("%m%d-%H%M%S")
    zip_local = os.path.join(HERE, "_pack_%s.zip" % job_id)
    zip_remote = "%s/ds_%s.zip" % (R_INCOMING, job_id)

    try:
        n = make_zip(STAGING, zip_local)
        size_mb = round(os.path.getsize(zip_local) / 1048576, 1)

        # 远端目录先建好（prepare_dataset.py 也要先就位，保证云端脚本是最新版）
        SSH_CLIENT.run(
            "mkdir -p %s %s %s %s" % (R_INCOMING, R_RUNS, R_OUT_ROOT, R_DATA),
            timeout=60)
        SSH_CLIENT.put(os.path.join(ROOT, "remote", "prepare_dataset.py"),
                       R_TRAIN + "/prepare_dataset.py")

        got = SSH_CLIENT.put(zip_local, zip_remote)
        if got != os.path.getsize(zip_local):
            return {"ok": False, "error": "上传不完整：本地 %d B，云端 %d B"
                    % (os.path.getsize(zip_local), got)}

        cmd = '%s %s/prepare_dataset.py %s %s %s %s' % (
            R_PY, R_TRAIN, zip_remote, R_DATA, sh(trigger), "" if clear else "--keep")
        out, err, rc = SSH_CLIENT.run(cmd, timeout=900)

        # 拆出结构化结果（最后一行 ###JSON###{...}）
        rep = None
        for line in out.splitlines():
            if line.startswith("###JSON###"):
                try:
                    rep = json.loads(line[len("###JSON###"):])
                except Exception:
                    rep = None
        SSH_CLIENT.run("rm -f %s" % zip_remote, timeout=60)

        log = "\n".join(l for l in out.splitlines() if not l.startswith("###JSON###"))
        if rc != 0 or not rep or not rep.get("ok"):
            return {"ok": False, "error": "云端解包/体检失败（rc=%s）" % rc,
                    "log": log[-6000:], "stderr": err[-3000:]}
        return {"ok": True, "n_files": n, "zip_mb": size_mb, "check": rep.get("check"),
                "log": log[-6000:]}
    finally:
        try:
            os.remove(zip_local)
        except OSError:
            pass


def start_job(o: dict) -> dict:
    """上传 job.sh 并**脱离 SSH 会话**启动。

    ★ 断连照跑的三个保险：
      1. `setsid`  —— 新建会话，脱离 SSH 的登录会话（收到 SIGHUP 也波及不到它）
      2. `nohup`   —— 再挡一次 SIGHUP
      3. 三件套重定向 `> log 2>&1 < /dev/null` —— 不持有 SSH 通道

    ⚠️⛔ 还必须用**子 shell 包一层** `( ... & )`。2026-10-14 实测（9 种写法对比）：
       直接 `setsid nohup bash job.sh >log 2>&1 </dev/null &` 时，paramiko 的
       `stdout.read()` 会**一直阻塞到任务结束**才返回 —— 症状是点了「开始训练」网页
       转圈几小时（训练本身照跑，但接口不返回、浏览器会超时）。
       包成 `( ... & )` 后 0.2s 返回（`setsid --fork` 同样有效）。
       ⇒ 网页端的启动接口必须保持这个形状，别再"简化"回去。
    """
    job_id = time.strftime("%m%d-%H%M%S")
    job_dir = "%s/%s" % (R_RUNS, job_id)
    out_dir = ("%s/_dry_%s" % (R_OUT_ROOT, job_id) if o.get("dry")
               else "%s/%s" % (R_OUT_ROOT, o["name"]))

    script = build_job_sh(o, job_id)
    local_sh = os.path.join(HERE, "_job_%s.sh" % job_id)
    with open(local_sh, "w", encoding="utf-8", newline="\n") as f:   # ★ 必须 LF，CRLF 会让远端 bash 报 \r
        f.write(script)

    try:
        # ⛔⛔ 续跑时**绝对不能**撞名改名：官方脚本是 os.path.join(output_dir, basename(值))，
        #     改了名 ⇒ 去新目录里找不到 checkpoint ⇒ 只 print 一句
        #     "does not exist. Starting a new training run." 然后**从零重训**（静默！）。
        #     所以续跑时跳过这段，强制复用 R_OUT_ROOT/<name>。
        if o.get("resume"):
            # ⚠️ 别写成 `ls ... | tail -1 || echo "(无)"` —— 管道返回码来自 tail（成功），
            #    `||` 永远不会触发，缺 checkpoint 时会静默得到空串。用计数判定。
            chk, _e, _c = SSH_CLIENT.run(
                'mkdir -p {j} {d}; n=$(ls -d {d}/checkpoint-* 2>/dev/null | wc -l); echo "NCKPT=$n"; '
                'mx=$(ls -d {d}/checkpoint-* 2>/dev/null | sed "s/.*checkpoint-//" | sort -n | tail -1); '
                'echo "MAX=${{mx:-0}}"'
                .format(j=job_dir, d=out_dir), timeout=60)
            if "NCKPT=0" in chk:
                return {"ok": False,
                        "error": "续跑失败：%s 下没有任何 checkpoint-* 目录。"
                                 "请确认「输出目录名」填的是上次训练的那个名字（不是新名字）。"
                                 % out_dir}
            # ⛔ 兜底「总步数 ≤ 已跑步数」这个致命误填：官方脚本此时**不报错**，
            #    会 print 一堆正常日志然后立刻 break（一步都不跑），极难自己看出来。
            mm = re.search(r"MAX=(\d+)", chk)
            if mm and int(mm.group(1)) > 0:
                done_steps = int(mm.group(1))
                if int(o["steps"]) <= done_steps:
                    return {"ok": False,
                            "error": "续跑失败：训练步数（%d）必须**大于**已跑的 %d 步。"
                                     "「训练步数」填的是**总步数**（含已跑的）——"
                                     "想再跑 500 步就填 %d。"
                                     % (int(o["steps"]), done_steps, done_steps + 500)}
        else:
            # 如果同名输出目录已存在，说明名字撞了 —— 用带时间戳的，避免覆盖上次的 LoRA
            chk, _e, _c = SSH_CLIENT.run(
                'mkdir -p %s && if [ -e %s ]; then echo EXISTS; else echo NEW; fi'
                % (job_dir, out_dir), timeout=60)
            if "EXISTS" in chk and not o.get("dry"):
                o = dict(o)
                o["name"] = "%s_%s" % (o["name"], job_id)
                out_dir = "%s/%s" % (R_OUT_ROOT, o["name"])
                script = build_job_sh(o, job_id)
                with open(local_sh, "w", encoding="utf-8", newline="\n") as f:
                    f.write(script)
                SSH_CLIENT.run("mkdir -p %s" % out_dir, timeout=60)

        SSH_CLIENT.put(local_sh, job_dir + "/job.sh")
        SSH_CLIENT.run("chmod +x %s/job.sh" % job_dir, timeout=60)

        launch = (
            'cd {d} && ( setsid nohup bash job.sh > train.log 2>&1 < /dev/null & ) && '
            'echo LAUNCHED; sleep 1; cat {d}/job.pid 2>/dev/null || echo "no-pid"'
        ).format(d=job_dir)
        out, err, rc = SSH_CLIENT.run(launch, timeout=90)
        if "LAUNCHED" not in out:
            return {"ok": False, "error": "启动命令没回 LAUNCHED（rc=%s）" % rc,
                    "log": (out + err)[-3000:]}

        pid = ""
        for line in out.splitlines():
            if line.strip().isdigit():
                pid = line.strip()

        st = {"job_id": job_id, "job_dir": job_dir, "out_dir": out_dir,
              "name": o["name"], "pid": pid, "started": time.time(),
              "opts": o, "log_offset": 0}
        save_state(st)
        return {"ok": True, "job": st}
    finally:
        try:
            os.remove(local_sh)
        except OSError:
            pass


def job_status(offset: int) -> dict:
    """查当前 job：活着吗 / 日志增量 / 进度 / 产物。"""
    st = load_state()
    if not st.get("job_id"):
        return {"ok": True, "has_job": False}

    d = st["job_dir"]
    script = """
J={d}
echo "### ALIVE"
p=$(cat "$J/job.pid" 2>/dev/null || echo "")
echo "$p"
if [ -n "$p" ] && kill -0 "$p" 2>/dev/null; then echo 1; else echo 0; fi
echo "### RC"; cat "$J/done.rc" 2>/dev/null || echo "-"
echo "### SIZE"; wc -c < "$J/train.log" 2>/dev/null || echo 0
echo "### GPU"
nvidia-smi --query-gpu=memory.used,memory.total,utilization.gpu --format=csv,noheader 2>/dev/null || echo "n/a"
echo "### OUT"
ls -la {o} 2>/dev/null | tail -n +2 || true
echo "### SH"; cat "$J/job.sh" 2>/dev/null | head -80
echo "### LOG"
tail -c +{off} "$J/train.log" 2>/dev/null || true
""".format(d=d, o=st["out_dir"], off=max(1, int(offset) + 1))

    out, err, rc = SSH_CLIENT.run(script, timeout=120)
    s = parse_sections(out)

    lines = (s.get("ALIVE") or "").splitlines()
    pid = lines[0].strip() if lines else ""
    alive = len(lines) > 1 and lines[1].strip() == "1"
    try:
        size = int((s.get("SIZE") or "0").strip() or 0)
    except ValueError:
        size = 0

    log = s.get("LOG") or ""
    reset = False
    if offset > size:               # 日志被重写过（重新开跑）⇒ 从头再来
        log = ""
        reset = True

    res = {
        "ok": True, "has_job": True,
        "job_id": st["job_id"], "name": st.get("name"), "out_dir": st.get("out_dir"),
        "pid": pid, "alive": alive,
        "rc": None if (s.get("RC") or "-") == "-" else (s.get("RC") or "").strip(),
        "log": log, "size": size, "offset": size, "reset": reset,
        "gpu": (s.get("GPU") or "n/a").strip(),
        "out_files": [l for l in (s.get("OUT") or "").splitlines() if l.strip()][-20:],
        "started": st.get("started"),
        "opts": st.get("opts"),
        "finished": (not alive) and ((s.get("RC") or "-") != "-"),
        "progress": parse_progress(log),
        "elapsed": round(time.time() - float(st.get("started") or time.time()), 1),
    }
    if alive or res["rc"] is not None:
        st["log_offset"] = size
        save_state(st)
        res["sh"] = s.get("SH") or ""
    if err.strip():
        res["stderr"] = err.strip()[-1500:]
    return res


RE_STEP = re.compile(r"(\d+)\s*/\s*(\d+)\s*\[")
RE_LOSS = re.compile(r"['\"]loss['\"]\s*:\s*([0-9.eE+\-]+)")


def parse_progress(log: str) -> dict | None:
    """从 tqdm / diffusers 的日志里抠出「第几步 / 共几步 / loss」。"""
    steps = RE_STEP.findall(log)
    losses = RE_LOSS.findall(log)
    if not steps and not losses:
        return None
    cur = tot = None
    for a, b in steps:
        cur, tot = int(a), int(b)
    return {
        "step": cur, "total": tot,
        "pct": round(cur / tot * 100, 1) if (cur and tot) else None,
        "loss": float(losses[-1]) if losses else None,
    }


def stop_job() -> dict:
    """停训练：按进程组杀（setsid ⇒ PGID == job.pid，连带 python 一起）。"""
    st = load_state()
    if not st.get("job_id"):
        return {"ok": False, "error": "没有正在跑的 job"}
    d = st["job_dir"]
    script = """
J={d}
p=$(cat "$J/job.pid" 2>/dev/null || echo "")
if [ -z "$p" ]; then echo "NO_PID"; exit 0; fi
if ! kill -0 "$p" 2>/dev/null; then echo "ALREADY_DEAD"; exit 0; fi
kill -TERM -"$p" 2>/dev/null || kill -TERM "$p" 2>/dev/null
sleep 3
if kill -0 "$p" 2>/dev/null; then kill -KILL -"$p" 2>/dev/null || kill -KILL "$p" 2>/dev/null; sleep 1; fi
echo "STOPPED $p"
""".format(d=d)
    out, err, _rc = SSH_CLIENT.run(script, timeout=120)
    time.sleep(0.5)
    return {"ok": True, "detail": out.strip(), "stderr": err.strip()[-500:]}


def list_lora() -> dict:
    """列远端输出目录里的 LoRA。"""
    st = load_state()
    if not st.get("job_id"):
        return {"ok": True, "files": []}
    out, _e, _c = SSH_CLIENT.run(
        'find %s -maxdepth 2 -name "*.safetensors" -printf "%%s\\t%%p\\n" 2>/dev/null || true'
        % R_OUT_ROOT, timeout=90)
    files = []
    for line in out.splitlines():
        if "\t" in line:
            sz, p = line.split("\t", 1)
            try:
                files.append({"path": p.strip(), "mb": round(int(sz) / 1048576, 1)})
            except ValueError:
                pass
    files.sort(key=lambda x: x["path"])
    return {"ok": True, "files": files}


def download_lora(paths: list[str]) -> dict:
    """把选中的 LoRA 拖回本机（不能选目录外的文件）。"""
    st = load_state()
    tag = st.get("name") or st.get("job_id") or time.strftime("%m%d-%H%M%S")
    dest = os.path.join(DOWNLOADS, str(tag))
    os.makedirs(dest, exist_ok=True)
    got = []
    for p in paths:
        rp = str(p)
        if not rp.startswith(R_OUT_ROOT) or ".." in rp:
            return {"ok": False, "error": "拒绝下载 %s（只允许 %s 内的文件）" % (rp, R_OUT_ROOT)}
        local = os.path.join(dest, os.path.basename(rp))
        SSH_CLIENT.get(rp, local)
        got.append({"path": local, "mb": round(os.path.getsize(local) / 1048576, 1)})
    return {"ok": True, "dir": dest, "files": got}


# ============================================================ DEFAULTS
# ⚠️ 与 remote/train_qwen21.sh 保持一致；网页上改哪个都行，实际以网页表单为准。
DEFAULTS = {
    "name": "bj_style",
    "trigger": "bj_style",
    "steps": 800,
    "rank": 16,
    "lora_alpha": 16,
    "lora_dropout": 0.05,
    "learning_rate": 1e-4,
    "lr_scheduler": "constant",
    "warmup": 0,
    "resolution": 1024,
    "batch": 1,
    "grad_accum": 4,
    # ⚠️ caption_dropout 的真实语义（读了官方脚本 484 / 1556 / 1760 行）：
    #    随机把一部分样本的 caption **整条换成空字符串** ⇒ 这部分走"无条件"训练，
    #    官方 help 原文是 "trained unconditionally. Improves classifier-free guidance."，
    #    **官方默认 0.0**。它**不是**"让触发词独立学会风格"（触发词靠"每条 caption 都以它开头"来学）。
    #    我们的模板给 0.1，属于"想要更稳的 CFG"的取舍，不是必需。
    "caption_dropout": 0.1,
    "ckpt_steps": 200,
    "ckpt_limit": 2,
    "seed": 42,
    "mixed_precision": "bf16",
    "offload": True,
    "gradient_checkpointing": True,
    "use_8bit_adam": True,
    "cache_latents": True,
    "bnb": False,
    "dry": False,
    # ★ 续跑（2026-10-14 加）。取值：
    #   ""            = 不续，从零训（默认）
    #   "latest"      = 用输出目录里步数最大的那个 checkpoint
    #   "checkpoint-N"= 指定某个（只填**目录名**，别填绝对路径 —— 官方脚本只取 basename
    #                   再拼 output_dir，传路径没用，见 .skills/78）
    # ⚠️⛔ 续跑有两条静默陷阱，改这里之前务必读 .skills/78：
    #   ① steps 是**总步数**（含已跑的）——想从 500 再跑 500 要填 1000；填 500 会立刻结束
    #   ② 输出目录**必须是原来那个**（不能撞名改名，否则找不到 checkpoint ⇒ 悄悄从零重训）
    "resume": "",
    # ★ LoRA 作用面（2026-10-14 加）。详见 .skills/79-LoRA作用面与rank.md
    #   背景：官方脚本默认只挂 ["to_k","to_q","to_v","to_out.0"]（4 个 attn 投影），
    #   完全**不动 img_mlp** ⇒ 128 个目标；官方加速 LoRA 是 224 个（多了 img_mlp 三种）。
    #   目标太少 ⇒ 梯度只好把每个张量拧得更狠 ⇒「浓度足、作用面窄」。
    #   加宽后与官方加速 LoRA **完全同构**，转换脚本和 ComfyUI 都不用改。
    # ⚠️ 必须写**全限定名**：PEFT 用 `key.endswith("."+名)` 匹配，只写 "proj" 会连
    #    txt_mlp.proj 一起命中（实测过）。2.1 是单流无 txt_mlp，但换版本就会静默训错层。
    # ⚠️ 别照抄 Qwen-Image 1.0 的 `img_mlp.net.0.proj` / `add_q_proj` —— 2.1 上这些
    #    名字不存在（1.0 是双流 + 通用 FeedForward，2.1 是单流 + 具名 SwiGLU）。
    "lora_layers": "to_k,to_q,to_v,to_out.0",
    # ★ 分桶（2026-10-14 加）。false = 所有图放大到 resolution² 再 RandomCrop；
    #   true = 按原图长宽比分桶，**小图分到小桶、不上采样**（官方 help：
    #   "smaller images are assigned to a smaller bucket instead of being upscaled"）。
    #   什么时候开：素材短边普遍 <1024（我们的 38 张里 35 张放大 >1.5 倍）。
    #   代价：训练边长变小 ⇒ 学到的细节层次降低；且开启后 --resolution 被忽略。
    #   什么时候不开：素材本来就 ≥1024，或想让所有图统一到同一尺度。
    "buckets": False,
    # ★ 训完自动关机（2026-10-15 加）。勾上 ⇒ job.sh 收尾时排程关机。
    #   AutoDL 上 `shutdown` **实测有效**（2026-10-15 03:0x 误触发，机器真停了）。
    #   ⚠️ 数据盘 /root/autodl-tmp **关机后仍在**（同一实例，见 .skills/00），
    #      所以 LoRA / checkpoint 不会丢；但**要下载 LoRA 就得再开机**。
    #   ⚠️ 用的是 `shutdown -h +1`（排程 1 分钟，走 init 而非本脚本的进程）——
    #      比 `( sleep; shutdown ) &` 稳，且能被 `shutdown -c` 取消。
    "shutdown": False,
}

ALLOWED_SCHED = ("constant", "constant_with_warmup", "linear", "cosine",
                 "cosine_with_restarts", "polynomial")

# LoRA 作用面白名单：必须是 diffusers 2.1 的真实模块名（配 PEFT 的 endswith 语义）
#   推荐档 = 全部 7 个（attn 4 + img_mlp 3），与官方 acc8 同构
LORA_LAYER_PRESETS = {
    "to_k", "to_q", "to_v", "to_out.0",
    "img_mlp.gate_layer", "img_mlp.proj", "img_mlp.out",
    # 下面三个在 2.1 上不存在（1.0 才有），留着仅为换模型时不报错
    "txt_mlp.gate_layer", "txt_mlp.proj", "txt_mlp.out",
}


def sanitize_opts(raw: dict) -> tuple[dict | None, str]:
    """表单 → 安全参数字典。任何一项不合法都明确报错，不猜。"""
    o = {**DEFAULTS, **{k: v for k, v in (raw or {}).items() if k in DEFAULTS}}

    name = re.sub(r"[^\w.\-]+", "_", str(o["name"]).strip()) or "bj_style"
    if name in (".", ".."):
        return None, "输出目录名不合法"
    o["name"] = name[:60]

    trig = str(o["trigger"]).strip()
    if not trig or len(trig) > 80:
        return None, "触发词不能为空，且不要超过 80 字符"
    o["trigger"] = trig

    def rng(key, lo, hi, kind=int):
        try:
            v = kind(o[key])
        except (TypeError, ValueError):
            raise ValueError("%s 不是合法数字：%r" % (key, o[key]))
        if not (lo <= v <= hi):
            raise ValueError("%s 超出范围（应在 %s ~ %s）：%r" % (key, lo, hi, v))
        return v

    try:
        o["steps"] = rng("steps", 1, 200000)
        o["rank"] = rng("rank", 1, 512)
        o["lora_alpha"] = rng("lora_alpha", 1, 512)
        o["lora_dropout"] = rng("lora_dropout", 0.0, 1.0, float)
        o["learning_rate"] = rng("learning_rate", 1e-7, 1e-2, float)
        o["warmup"] = rng("warmup", 0, 10000)
        o["resolution"] = rng("resolution", 256, 2048)
        o["batch"] = rng("batch", 1, 16)
        o["grad_accum"] = rng("grad_accum", 1, 64)
        o["caption_dropout"] = rng("caption_dropout", 0.0, 1.0, float)
        o["ckpt_steps"] = rng("ckpt_steps", 1, 200000)
        o["ckpt_limit"] = rng("ckpt_limit", 1, 50)
        o["seed"] = rng("seed", 0, 2 ** 31 - 1)
    except ValueError as e:
        return None, str(e)

    if o["lr_scheduler"] not in ALLOWED_SCHED:
        return None, "不认识的 lr_scheduler：%r" % o["lr_scheduler"]
    if o["mixed_precision"] not in ("no", "fp16", "bf16"):
        return None, "不认识的 mixed_precision：%r" % o["mixed_precision"]
    for k in ("offload", "gradient_checkpointing", "use_8bit_adam", "cache_latents",
              "bnb", "dry", "buckets", "shutdown"):
        o[k] = bool(o[k])

    # ---- LoRA 作用面（2026-10-14）----
    # ⛔ 刻意只放行**白名单里的全限定名**。原因（PEFT 0.21.2 tuners_utils 2377 行）：
    #      target_module_found = any(key.endswith("." + t) for t in target_modules)
    #    是 endswith ⇒ 只写 "proj" 会把 img_mlp.proj 和 txt_mlp.proj 一起挂上（实测过）。
    #    写错不报错、只是悄悄训了别的层，最难查 ⇒ 宁可在这里直接拒。
    layers = str(o.get("lora_layers") or "").strip()
    if layers:
        parts = [p.strip() for p in layers.split(",") if p.strip()]
        bad = [p for p in parts if p not in LORA_LAYER_PRESETS]
        if bad:
            return None, ("LoRA 作用面里有不认识的模块：%s。\n"
                          "2.1 只支持这 7 个：to_k,to_q,to_v,to_out.0,"
                          "img_mlp.gate_layer,img_mlp.proj,img_mlp.out\n"
                          "（必须写全限定名；只写 proj 之类会误伤 txt_mlp）" % "、".join(bad))
        if not parts:
            return None, "LoRA 作用面为空，请至少留一个模块"
        if len(set(parts)) != len(parts):
            return None, "LoRA 作用面有重复项：%s" % layers
        o["lora_layers"] = ",".join(parts)
    else:
        o["lora_layers"] = ""

    # ---- 续跑（2026-10-14）----
    # 只接受三种形态：空 / "latest" / "checkpoint-<数字>"。
    # ⛔ 故意**不允许**填绝对路径：官方脚本是 os.path.join(output_dir, basename(值))，
    #    传路径照样去 output_dir 下找（见 .skills/78 坑 1），放行只会给人"我填了路径"的错觉。
    r = str(o.get("resume") or "").strip()
    if r:
        if r != "latest":
            m = re.fullmatch(r"checkpoint-(\d+)", r)
            if not m:
                return None, ("续跑只能填 latest 或 checkpoint-<步数>（只填目录名，不要路径），"
                              "收到 %r" % r)
        if o["dry"]:
            return None, "演练模式不能续跑（演练不加载权重、也不产生真 checkpoint），请取消勾选"
        if o["name"] != o["name"].strip():        # 理论上前已处理，防御性
            return None, "输出目录名不合法"
    o["resume"] = r
    return o, ""


# ============================================================ HTTP
class Handler(BaseHTTPRequestHandler):
    server_version = "TrainWeb/1.0"
    protocol_version = "HTTP/1.1"

    # ---------- 基础 ----------
    def log_message(self, fmt, *args):        # 别把每个轮询都刷到控制台
        pass

    def _body(self) -> bytes:
        n = int(self.headers.get("Content-Length") or 0)
        buf = bytearray()
        while len(buf) < n:
            chunk = self.rfile.read(min(1 << 20, n - len(buf)))
            if not chunk:
                break
            buf += chunk
        return bytes(buf)

    def _json(self, obj, code: int = 200) -> None:
        body = json.dumps(obj, ensure_ascii=False).encode("utf-8")
        self.send_response(code)
        self.send_header("Content-Type", "application/json; charset=utf-8")
        self.send_header("Content-Length", str(len(body)))
        self.send_header("Cache-Control", "no-store")
        self.end_headers()
        self.wfile.write(body)

    def _html(self) -> None:
        p = os.path.join(HERE, "index.html")
        try:
            with open(p, "rb") as f:
                body = f.read()
        except OSError as e:
            self._json({"error": "读不到 index.html: %s" % e}, 500)
            return
        self.send_response(200)
        self.send_header("Content-Type", "text/html; charset=utf-8")
        self.send_header("Content-Length", str(len(body)))
        self.send_header("Cache-Control", "no-store")
        self.end_headers()
        self.wfile.write(body)

    def _q(self) -> dict:
        q = urllib.parse.urlparse(self.path).query
        return {k: v[0] for k, v in urllib.parse.parse_qs(q).items()}

    # ---------- GET ----------
    def do_GET(self):                                    # noqa: N802
        path = urllib.parse.urlparse(self.path).path
        try:
            if path in ("/", "/index.html"):
                return self._html()
            if path == "/api/ssh":
                return self._json(api_ssh_check())
            if path == "/api/stage/list":
                d = stage_list()
                trig = self._q().get("trigger", "")
                try:
                    d["check"] = ds._check(STAGING, trig)
                except Exception as e:
                    d["check"] = {"error": "%s: %s" % (type(e).__name__, e)}
                return self._json(d)
            if path == "/api/job":
                off = self._q().get("offset", "0")
                return self._json(job_status(int(off) if off.isdigit() else 0))
            if path == "/api/lora":
                return self._json(list_lora())
            if path == "/api/defaults":
                return self._json(DEFAULTS)
            return self._json({"error": "no such endpoint: %s" % path}, 404)
        except Exception as e:
            traceback.print_exc()
            return self._json({"ok": False, "error": "%s: %s" % (type(e).__name__, e)}, 500)

    # ---------- POST ----------
    def do_POST(self):                                   # noqa: N802
        path = urllib.parse.urlparse(self.path).path
        q = self._q()
        try:
            # 拖进来的是文件夹时，浏览器会给每个文件单独发一次；走裸字节最省事，
            # 不用手搓 multipart 解析。
            if path == "/api/stage/put":
                name = q.get("name") or "unnamed.png"
                body = self._body()
                if not body:
                    return self._json({"ok": False, "error": "空文件：%s" % name}, 400)
                dest = free_name(name)
                with open(dest, "wb") as f:
                    f.write(body)
                return self._json({"ok": True, "saved": os.path.basename(dest),
                                   "bytes": len(body), **stage_list_files_only()})

            if path == "/api/stage/zip":
                body = self._body()
                if not body:
                    return self._json({"ok": False, "error": "空 zip"}, 400)
                tmp = os.path.join(HERE, "_incoming.zip")
                with open(tmp, "wb") as f:
                    f.write(body)
                n = 0
                with zipfile.ZipFile(tmp) as zf:
                    for info in zf.infolist():
                        nm = info.filename.replace("\\", "/")
                        if info.is_dir() or "__MACOSX" in nm or "/._" in nm:
                            continue
                        base = os.path.basename(nm)
                        if not base or base.startswith("."):
                            continue
                        if os.path.splitext(base)[1].lower() not in (
                                ".png", ".jpg", ".jpeg", ".webp", ".bmp", ".txt"):
                            continue
                        with zf.open(info) as src, open(free_name(base), "wb") as dst:
                            shutil.copyfileobj(src, dst)
                        n += 1
                os.remove(tmp)
                return self._json({"ok": True, "unpacked": n, **stage_list_files_only()})

            if path == "/api/stage/clear":
                with OP_LOCK:
                    shutil.rmtree(STAGING, ignore_errors=True)
                    os.makedirs(STAGING, exist_ok=True)
                return self._json({"ok": True, **stage_list_files_only()})

            if path == "/api/stage/delete":
                body = json.loads(self._body() or b"{}")
                names = body.get("names") or []
                killed = []
                for nm in names:
                    p = os.path.join(STAGING, os.path.basename(str(nm)))
                    if os.path.isfile(p):
                        os.remove(p)
                        killed.append(os.path.basename(p))
                return self._json({"ok": True, "deleted": killed, **stage_list_files_only()})

            if path == "/api/data/push":
                body = json.loads(self._body() or b"{}")
                if not OP_LOCK.acquire(blocking=False):
                    return self._json({"ok": False, "error": "另一个云端操作正在进行，稍等"}, 409)
                try:
                    r = api_dataset_push(bool(body.get("clear", True)),
                                         str(body.get("trigger", "")))
                finally:
                    OP_LOCK.release()
                return self._json(r)

            if path == "/api/job/start":
                body = json.loads(self._body() or b"{}")
                opts, msg = sanitize_opts(body)
                if opts is None:
                    return self._json({"ok": False, "error": msg}, 400)
                if not OP_LOCK.acquire(blocking=False):
                    return self._json({"ok": False, "error": "另一个云端操作正在进行，稍等"}, 409)
                try:
                    st = job_status(0)
                    if st.get("has_job") and st.get("alive"):
                        return self._json({"ok": False, "error":
                                           "已经有一个训练在跑了（job %s）——先停止它" % st["job_id"]}, 409)
                    r = start_job(opts)
                finally:
                    OP_LOCK.release()
                return self._json(r)

            if path == "/api/job/stop":
                if not OP_LOCK.acquire(blocking=False):
                    return self._json({"ok": False, "error": "另一个云端操作正在进行，稍等"}, 409)
                try:
                    r = stop_job()
                finally:
                    OP_LOCK.release()
                return self._json(r)

            if path == "/api/lora/download":
                body = json.loads(self._body() or b"{}")
                if not OP_LOCK.acquire(blocking=False):
                    return self._json({"ok": False, "error": "另一个云端操作正在进行，稍等"}, 409)
                try:
                    r = download_lora(body.get("paths") or [])
                finally:
                    OP_LOCK.release()
                return self._json(r)

            return self._json({"error": "no such endpoint: %s" % path}, 404)
        except Exception as e:
            traceback.print_exc()
            return self._json({"ok": False, "error": "%s: %s" % (type(e).__name__, e)}, 500)


def stage_list_files_only() -> dict:
    d = stage_list()
    d.pop("files", None)
    return d


def main() -> None:
    port = DEFAULT_PORT
    for a in sys.argv[1:]:
        if a.isdigit():
            port = int(a)
    open_browser = "--open" in sys.argv

    for s in (sys.stdout, sys.stderr):
        try:
            s.reconfigure(encoding="utf-8", errors="replace")     # type: ignore[attr-defined]
        except Exception:
            pass

    print("=" * 62)
    print("  Qwen-Image-2.1 LoRA 训练台")
    print("=" * 62)
    cfg = cloud_sh.load_cfg()
    print("  云卡    : %s:%s  (口令%s)"
          % (cfg["host"] or "(未配置)", cfg["port"],
             "已设置" if cfg["pass"] else "**缺失**"))
    print("  打开    : http://%s:%d" % (HOST, port))
    print("  暂存    : %s" % STAGING)
    print("  产物    : %s" % DOWNLOADS)
    print()
    print("  ★ 点「开始训练」后云端独立开跑，本页面关掉也不影响；")
    print("    想停下来要点本页面的「停止训练」。")
    print("  Ctrl+C 退出本服务（不会影响云端正在跑的训练）")
    print("=" * 62)

    try:
        httpd = ThreadingHTTPServer((HOST, port), Handler)
    except OSError as e:
        print("[x] 端口 %d 起不来：%s" % (port, e))
        print("    多半是训练台已经在跑了 —— 直接开 http://%s:%d 就行。" % (HOST, port))
        print("    想另起一个：train_web.cmd 8888")
        sys.exit(1)
    httpd.daemon_threads = True
    if open_browser:
        def _open() -> None:
            try:
                import webbrowser
                webbrowser.open("http://%s:%d" % (HOST, port))
            except Exception:
                pass
        threading.Timer(1.0, _open).start()          # 等服务先 listen 上，免得打开是空白页
    try:
        httpd.serve_forever()
    except KeyboardInterrupt:
        print("\n[退出] 本机服务已停止（云端训练不受影响）")
    finally:
        httpd.server_close()
        SSH_CLIENT.reset()


if __name__ == "__main__":
    main()
