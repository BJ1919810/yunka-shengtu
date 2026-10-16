# -*- coding: utf-8 -*-
"""云卡生图 CLI —— 通过 SSH 隧道直连 AutoDL 上的 ComfyUI（Qwen-Image-2.1，原生 RGBA + 多参考图）。

与笑樱项目彻底分离：这是独立项目，不依赖笑樱任何文件。

云卡侧环境：
    ComfyUI   /root/autodl-tmp/ComfyUI
    Python    /root/autodl-tmp/envs/comfy/bin/python
    端口      6006（127.0.0.1，SSH 隧道进来，不占公网）
    GPU       RTX 3080 / 10GB ⇒ 量化必选 + 文本编码器靠DynamicVRAM 换出

⭐ Qwen-Image-2.1 相对旧版（Qwen-Image 2512）是**架构重做**，不是版本迭代：
    DiT            20B → 7B（32 层 Single-Stream）
    文本编码器     Qwen2.5-VL 7B → Qwen3-VL 8B
    VAE            qwen_image_vae → 64 通道 RGBA VAE（16倍压缩）
    编码节点       CLIPTextEncode → **TextEncodeQwenImage21**（一次搞定正/负条件+latent）
   ⇒ 旧的 UNETLoader + CLIPLoader(type=qwen_image) + EmptySD3LatentImage 那套**已作废**。

五个能力：
    t2i     文生图（--transparent 走原生 RGBA）
    i2i     单图编辑（等价 multi 的1 张特例，--denoise 仍可用）
    multi   多参考图编辑（最多 10 张，可混 mask）
    inpaint 局部重绘（mask 白区重画，其余原样贴回）
    alpha   保留：**黑白蒙版法**仍有用（2.1 抠图是生成式的、要抽卡，做确定性抠图更稳）
    info    查候选（模型/采样器/节点是否可用）

用法：
    # 文生图
    python cloudgen.py t2i --prompt "..." [--out a.png]
    # 透明背景 PNG（原生 alpha 通道，不是抠图）
    python cloudgen.py t2i --prompt "一个绿色小恐龙图标" --transparent --out logo.png
    # 单图编辑
    python cloudgen.py i2i --src a.png --prompt "改成夜景" --out b.png
    # 多参考图合成（最多 10 张）
    python cloudgen.py multi --refs a.png,b.png,c.png --prompt "三个人一起合照" --out c.png
    # 局部重绘
    python cloudgen.py inpaint --src a.png --mask-rect "-240,-130,-1,-1" --out b.png
    # 透明 PNG（蒙版法，确定性）
    python cloudgen.py alpha --color c.png --mask m.png --out t.png
    # 查云卡状态
    python cloudgen.py info [--node TextEncodeQwenImage21]

启动云卡 ComfyUI（幂等，已在跑就直接返回）：
    python cloud_sh.py --file remote/start_comfy.sh

口令来源：环境变量 COMFY_SSH_* 或同目录 cloud-comfy.config.json
"""
from __future__ import annotations

import argparse
import http.client
import json
import math
import mimetypes
import os
import sys
import time
import uuid

import paramiko

HERE = os.path.dirname(os.path.abspath(__file__))
CFG_PATH = os.path.join(HERE, "cloud-comfy.config.json")
OUT_DIR = os.path.join(HERE, "out")

DEFAULTS = {
    # ⛔ 这三个是 Comfy-Org/Qwen-Image-2.1 的 repack 文件名，不是原始 diffusers 仓库
    "unet": "qwen_image_2.1_bf16.safetensors",
    "clip": "qwen3vl_8b_bf16.safetensors",
    "vae": "qwen_image_2.1_vae_bf16.safetensors",
    # 官方模板值：steps 25 / cfg 1 / euler / simple，1024×1024
    # ⚠️ cfg 必须 ≥1。旧版的 2.5 那套直接照搬会出噪声。
    "width": 1024, "height": 1024,
    # ⚠️ 官方模板给 25 步；实测 8 步细节就开始软，10 步是速度/质量的平衡点（约省 60% 时间）
    "steps": 10, "cfg": 1.0, "shift": 3.5,
    "sampler": "euler", "scheduler": "simple",
    "negative": "",
    # 2026-10-10：默认 auto（先用空闲显存、不够落内存；官方 QwenImage21Cache 的 default）
    # ⚠️ 旧值 cpu 是 10GB 卡时代的结论，32G 卡上白白吃亏
    "cache_device": "auto",
    "cache_dtype": "default",
    # 参考图送进文本编码器时的统一边长；0 = 保持每张原图尺寸。
    # ★ 编辑场景现在**不再**自动置 0（见 REF_DEFAULT_RES 的注释），这里有意义：
    #   节点 TextEncodeQwenImage21 自己的 default 就是 1024。
    "resolution": 1024,
    # 指定画幅（--aspect）用的总像素预算；1.0 ≈ 1024²，2K 档给 4.0
    "megapixels": 1.0,
}


# ------------------------------------------------- 显存预算（★ 实测得出，别凭感觉改）
#
# ⛔ 为什么必须做这个闸：`TextEncodeQwenImage21` 的 `resolution` 就是这个模型自带的
#    "云卡侧降质开关"（节点 tooltip 原话:"Reference images are resized to about
#    resolution x resolution pixels ... 0 keeps each reference at its own size"）。
#    编辑场景曾经被我们默认置成 0 ⇒ 4K 参考图原样进模型，三条通路一起膨胀：
#      · VLM 视觉编码（Qwen3-VL 8B）
#      · VAE 编码 → 参考图 latent 拼进 DiT 序列（prefix KV 的一部分）
#      · 输出 latent 也跟着 image_1 变大
#    2026-10-03 实测 3840×2160（8.29MP）直接把 10GB 卡打爆。
#
# ⛔ 更严重的是**炸的方式**：不是任务报错，而是整个 ComfyUI 进程 Abort ——
#      Ran out of memory when regular VAE decoding（回退分块解码）
#      → CUDA error: invalid argument → Fatal Python error: Aborted
#    一次越界请求会把云卡服务整个打死，别的任务一起陪葬。所以必须提交前拦。
#
# 实测边界（RTX 3080 10GB，steps=1 —— OOM 发生在编码/解码，与步数无关）：
#     参考 2.00MP + 输出 1.00MP → OK          参考 4.50MP + 输出 1.00MP → OK
#     参考 3.57MP + 输出 4.18MP → OK（真实跑通过的那一档）
#     参考 4.50MP + 输出 3.98MP → OOM         参考 8.00MP + 输出 1.00MP → OOM
#     输出 9.00MP（无参考图）→ OK 但 66s/步    输出 12.25MP → 进程被 Abort
# ⇒ 参考图每 MP 明显比输出每 MP 贵（多走 VLM + VAE + prefix KV 两道工序）。
# ⇒ 两条闸，都留了余量：REF_MAX_MP 卡参考图，TOTAL_MAX_MP 卡「参考图 + 输出」。
# ★ 2026-10-10 云卡换 4080S 32GB 后实测：1.05MP→22s | 4.19MP→57s | 9.44MP→171s | 16.8MP→417s
#   ⇒ 都不 OOM，但 3K 以上太慢 ⇒ **输出封顶 2K（2048² = 4.0MP）**（用户拍板）
# ⚠️ 输出与参考图是【两条独立额度】：OUTPUT 只夹输出，REF 只管参考图，TOTAL 防两项都顶格。
# ⚠️ 这份闸与 comfy_web/server.py 里那份是【两份实现】，改一处必须改另一处。
OUTPUT_MAX_MP = 4.0     # 输出画幅上限（2048²）
REF_MAX_MP = 4.0        # 参考图合计上限
TOTAL_MAX_MP = 8.0      # 参考图 + 输出 的合计上限
REF_DEFAULT_RES = 1024   # 编辑场景的统一降质档（= 节点自带 default）
REF_MIN_RES = 320        # 参考图档位下限，再低就没意义了


# ------------------------------------------------------------------ 配置 / 隧道

def load_cfg() -> dict:
    cfg: dict = {}
    if os.path.exists(CFG_PATH):
        with open(CFG_PATH, encoding="utf-8") as f:
            cfg = json.load(f)
    out = {
        "host": os.environ.get("COMFY_SSH_HOST") or cfg.get("ssh_host", ""),
        "port": int(os.environ.get("COMFY_SSH_PORT") or cfg.get("ssh_port", 22)),
        "user": os.environ.get("COMFY_SSH_USER") or cfg.get("ssh_user", "root"),
        "pass": os.environ.get("COMFY_SSH_PASS") or cfg.get("ssh_password", ""),
        "comfy": int(os.environ.get("COMFY_REMOTE_PORT") or cfg.get("remote_comfy_port", 6006)),
    }
    if not out["host"] or not out["pass"]:
        sys.exit("[配置缺失] 需要 ssh_host / ssh_password（%s 或 COMFY_SSH_* 环境变量）" % CFG_PATH)
    return out


class Tunnel:
    """SSH direct-tcpip → 云卡 127.0.0.1:6006（ComfyUI）。"""

    def __init__(self) -> None:
        self.cfg = load_cfg()
        # ⚠️ AutoDL 的 sshd 会回一句纯文本 "Exceeded MaxStartups\r\n"（未认证并发连接达上限，
        #    通常是别的隧道/扫描占着）。它不是 SSH banner ⇒ paramiko 抛
        #    "Error reading SSH protocol banner"，**云卡其实是好的**。退避重连即可。
        #    2026-10-02 实测 6 次里有 2 次撞上。
        last: Exception | None = None
        for attempt in range(6):
            self.ssh = paramiko.SSHClient()
            self.ssh.set_missing_host_key_policy(paramiko.AutoAddPolicy())
            try:
                self.ssh.connect(self.cfg["host"], self.cfg["port"], self.cfg["user"], self.cfg["pass"],
                                 timeout=25, look_for_keys=False, allow_agent=False)
                break
            except Exception as e:
                last = e
                try:
                    self.ssh.close()
                except Exception:
                    pass
                if attempt == 5:
                    sys.exit("[SSH 连接失败，重试 6 次仍不通] %s: %s" % (type(e).__name__, e))
                time.sleep(0.4 * (attempt + 1))
        assert last is None or self.ssh is not None
        self._transport = self.ssh.get_transport()

    def channel(self) -> paramiko.Channel:
        return self._transport.open_channel(
            "direct-tcpip", ("127.0.0.1", self.cfg["comfy"]), ("127.0.0.1", 0))

    def conn(self) -> http.client.HTTPConnection:
        factory = self.channel

        class _Conn(http.client.HTTPConnection):
            def connect(self) -> None:      # noqa: D401
                self.sock = factory()

        return _Conn("127.0.0.1", self.cfg["comfy"], timeout=1800)

    def close(self) -> None:
        try:
            self.ssh.close()
        except Exception:
            pass


# ------------------------------------------------------------------ HTTP 小工具

def api_get(t: Tunnel, path: str) -> bytes:
    c = t.conn()
    try:
        c.request("GET", path)
        r = c.getresponse()
        data = r.read()
        if r.status != 200:
            raise RuntimeError("GET %s → %s %s" % (path, r.status, data[:300]))
        return data
    finally:
        c.close()


def api_json(t: Tunnel, path: str, body: dict | None = None) -> dict:
    c = t.conn()
    try:
        if body is None:
            c.request("GET", path)
        else:
            c.request("POST", path, body=json.dumps(body).encode(),
                      headers={"Content-Type": "application/json"})
        r = c.getresponse()
        data = r.read()
        if r.status != 200:
            raise RuntimeError("%s → %s %s" % (path, r.status, data[:400]))
        return json.loads(data or b"{}")
    finally:
        c.close()


def ascii_name(src_path: str, pre: str) -> str:
    """中文名经 HTTP→JSON 多层转码易坏，统一 ASCII 化。"""
    n = pre + os.path.basename(src_path).replace(" ", "_")
    if not n.isascii():
        n = pre + uuid.uuid4().hex[:10] + (os.path.splitext(src_path)[1] or ".png")
    return n


def upload_image(t: Tunnel, src_path: str, name: str) -> str:
    """上传源图到云卡 ComfyUI/input/，返回可用文件名（走 /upload/image，不是 SFTP）。"""
    with open(src_path, "rb") as f:
        raw = f.read()
    boundary = "----xy" + uuid.uuid4().hex
    ctype = mimetypes.guess_type(src_path)[0] or "image/png"
    body = b"".join([
        ("--%s\r\n" % boundary).encode(),
        ('Content-Disposition: form-data; name="image"; filename="%s"\r\n' % name).encode(),
        ("Content-Type: %s\r\n\r\n" % ctype).encode(),
        raw,
        ("\r\n--%s--\r\n" % boundary).encode(),
    ])
    c = t.conn()
    try:
        c.request("POST", "/upload/image", body=body,
                  headers={"Content-Type": "multipart/form-data; boundary=%s" % boundary})
        r = c.getresponse()
        data = json.loads(r.read() or b"{}")
        if r.status != 200:
            raise RuntimeError("上传失败 %s %s" % (r.status, data))
        return data.get("name") or name
    finally:
        c.close()


def _seed_of(entry: dict) -> int:
    """从 /history 的 entry 里取回实际使用的种子。

    ⚠️ **两个坑叠在一起**（2026-10-02 实测）：
      1) 旧代码写死 `prompt[2].get("seed")`。0.38 之后 entry["prompt"] 变成
         `[num, uuid, nodes_dict, extra_dict, outputs_list]` 共 5 个元素，
         `[2]` 恰好还是节点字典 —— 这一条侥幸没炸，但下面一条炸了。
      2) **seed 不在节点顶层，而在节点["6"]["inputs"]["seed"]**。
         顶层找 "seed" 永远返回 None ⇒ 日志回显 -1，让人以为"钉了 seed 也复现不了"。
      所以改成：遍历节点字典 → 找有 inputs.seed 的那个。
    """
    p = entry.get("prompt")
    nodes = None
    if isinstance(p, list):
        for x in p:
            # 节点字典的特征：值全是 dict，且里面套着 class_type
            if isinstance(x, dict) and any(
                    isinstance(v, dict) and "class_type" in v for v in x.values()):
                nodes = x
                break
    elif isinstance(p, dict):
        nodes = p
    if not nodes:
        return -1
    for nid, node in nodes.items():
        if not isinstance(node, dict):
            continue
        inputs = node.get("inputs")
        if isinstance(inputs, dict) and "seed" in inputs:
            try:
                return int(inputs["seed"])
            except (TypeError, ValueError):
                pass
    return -1


def split_refs(chunks: list[str]) -> list[str]:
    """把 `--refs` 收集到的多个片段摊平成一维路径列表。

    `--refs` 必须用 `action="append"`（**不能**用 `nargs="+"`：后者在重复传参时会被
    argparse 覆盖成只剩最后一个，2026-10-03 因此整轮实验结论跑偏）。所以这里要同时兼容：
        --refs a.png --refs b.png   → ['a.png', 'b.png']
        --refs a.png,b.png          → ['a.png,b.png'] → 拆成 ['a.png', 'b.png']
    """
    flat: list[str] = []
    for chunk in chunks:
        for piece in str(chunk).split(","):
            piece = piece.strip()
            if piece:
                flat.append(piece)
    return flat


# ------------------------------------------------- 画幅（输出尺寸由谁决定）

# Qwen-Image-2.1 支持的比例（官方 README 的 2K 表就是这几档）。
ASPECTS: dict[str, tuple[int, int]] = {
    "1:1": (1, 1), "4:3": (4, 3), "3:4": (3, 4), "3:2": (3, 2), "2:3": (2, 3),
    "16:9": (16, 9), "9:16": (9, 16), "21:9": (21, 9),
}


def calc_size(aspect: str, megapixels: float, multiple: int = 32) -> tuple[int, int]:
    """按比例 + 总像素预算算宽高，取整到 multiple 的倍数（Qwen 建议 32）。

    ⚠️ 与官方 diffusers 示例的 WH_RATIO_TO_SIZE 同一套口径（1MP ≈ 1024²）。
       官方 2K 档 9:16 给 1536×2752，本函数给 1536×2720 —— 差在取整策略，可接受；
       要精确值就直接用 --width/--height 绕过它。
    """
    if aspect not in ASPECTS:
        sys.exit("[参数错误] --aspect 只支持：%s" % "、".join(ASPECTS))
    rw, rh = ASPECTS[aspect]
    total = float(megapixels) * 1024 * 1024
    w = max(multiple, int(round(math.sqrt(total * rw / rh) / multiple) * multiple))
    h = max(multiple, int(round((total / w) / multiple) * multiple))
    return w, h


def image_mp(path: str) -> float:
    """读图片面积（MP）。只在**显式要"保持原尺寸"**时才需要 —— 那种写法不知道尺寸就没法拦。"""
    from PIL import Image
    with Image.open(path) as im:
        w, h = im.size
    return w * h / 1048576.0


def ref_paths_of(a: argparse.Namespace) -> list[str]:
    """列出这次会作为「参考图」送进模型的本地文件。"""
    if a.cmd == "multi":
        return list(a.refs or [])
    if a.cmd == "i2i":
        return [a.src]
    if a.cmd == "inpaint":
        # 2.1 的 inpaint = 原图 + mask 两张参考图；mask 还没生成时按原图同尺寸估
        return [a.src, a.mask if getattr(a, "mask", None) else a.src]
    return []


def clamp_size_to(mp_cap: float, size: tuple[int, int]) -> tuple[int, int]:
    """把 (w,h) **等比**缩到 mp_cap 以内，宽高对齐 32 的倍数。

    ⚠️ 必须等比（2026-10-03 用户特别提过）。
    ⚠️ 不能简单地逐轴 floor：那样 4096×2304 会变成 1920×1056，比例误差 2.3%，
       肉眼能看出歪。这里在 32 的倍数网格上小范围搜索，挑「面积不超上限 且
       比例最接近原比例」的那一对 —— 面积是硬约束，比例尽量准。
    """
    w, h = int(size[0]), int(size[1])
    cap = mp_cap * 1048576.0
    if mp_cap <= 0 or w * h <= cap:
        return (w, h)
    ideal_w = max(1, int(w * math.sqrt(cap / (w * h))) // 32)
    best = None
    for nw in range(max(1, ideal_w - 6), ideal_w + 7):
        W = nw * 32
        nh = int((cap / W) // 32)
        for cand in (nh, nh - 1):
            if cand < 1:
                continue
            H = cand * 32
            err = abs((W / H) / (w / h) - 1)
            if best is None or err < best[0] - 1e-12:
                best = (err, W, H)
    if best is None:                       # 理论上不会走到：总能找到足够小的一对
        return (max(32, w // 32 * 32), max(32, h // 32 * 32))
    return (best[1], best[2])


def plan_ref_res(n_refs: int, out_mp: float | None, want: int) -> int:
    """算该给 TextEncodeQwenImage21 的 resolution（参考图的统一面积档）。

    out_mp=None 表示"输出跟随第一张参考图" ⇒ 输出面积 ≈ resolution²，
    这时约束是 (n+1)·N² ≤ TOTAL；否则是 n·N² ≤ min(REF_MAX, TOTAL - 输出)。
    """
    if n_refs <= 0 or want <= 0:
        return want
    if out_mp is None:
        n2_max = min(REF_MAX_MP / n_refs, TOTAL_MAX_MP / (n_refs + 1))
    else:
        n2_max = min(REF_MAX_MP / n_refs, max(0.0, TOTAL_MAX_MP - out_mp) / n_refs)
    n_cap = int(math.sqrt(max(0.0, n2_max) * 1048576.0))
    return max(REF_MIN_RES, min(want, n_cap) // 32 * 32)


def guard_budget(a: argparse.Namespace) -> None:
    """★ 提交前的显存预算闸：超限就自动降 + 打印原因（用户选的"自动降 + 提示"）。

    优先级：**先保住参考图档位，让输出画幅让位**。反过来（先满足输出）会把参考图
    挤到 320 档、参考图几乎等于没给，编辑效果反而更差。
    实测依据见文件顶部常量注释。
    """
    refs = ref_paths_of(a)
    n = len(refs)
    explicit_res = "--resolution" in sys.argv

    if a.cmd == "t2i":
        if a.resolution <= 0:
            print("[提示] t2i 无参考图，--resolution 0 无意义，改用 %d" % DEFAULTS["resolution"])
            a.resolution = DEFAULTS["resolution"]
        # 无参考图 ⇒ 唯一变量就是输出，直接用 OUTPUT_MAX_MP（2K 封顶）
        if getattr(a, "size", None):
            mp = a.size[0] * a.size[1] / 1048576.0
            if mp > OUTPUT_MAX_MP:
                ns = clamp_size_to(OUTPUT_MAX_MP, a.size)
                print("[显存预算] 输出 %.2fMP > %.1fMP 上限 ⇒ 等比缩到 %dx%d（保持比例）"
                      % (mp, OUTPUT_MAX_MP, ns[0], ns[1]))
                a.size = ns
        else:
            mp = a.resolution ** 2 / 1048576.0
            if mp > OUTPUT_MAX_MP:
                res = max(32, int(math.sqrt(OUTPUT_MAX_MP * 1048576.0)) // 32 * 32)
                print("[显存预算] 输出 %.2fMP > %.1fMP 上限 ⇒ resolution %d → %d"
                      % (mp, OUTPUT_MAX_MP, a.resolution, res))
                a.resolution = res
        return

    want = a.resolution if explicit_res else REF_DEFAULT_RES
    out_mp = (a.size[0] * a.size[1] / 1048576.0) if getattr(a, "size", None) else None

    # ---- 显式要"保持原尺寸"（resolution=0）：不知道真实尺寸就没法拦
    if want <= 0 and n:
        try:
            areas = [image_mp(p) for p in refs]
        except Exception as e:
            print("[警告] 读不到参考图尺寸（%s）⇒ 按最坏情况处理，改用 %d"
                  % (str(e)[:60], REF_DEFAULT_RES))
            areas = []
        ref_mp = sum(areas) if areas else float("inf")
        # 跟随时输出面积 ≈ **最大的那张**参考图（latent 只认 image_1，不是总和）
        out_est = out_mp if out_mp is not None else (max(areas) if areas else 0.0)
        limit = min(REF_MAX_MP, max(0.5, TOTAL_MAX_MP - out_est))
        if ref_mp <= limit:
            return                      # 原尺寸本就在安全区 ⇒ 保持保真
        print("[显存预算] 参考图合计 %.2fMP 超预算（上限 %.2fMP，%d 张）"
              % (ref_mp, limit, n))
        want = REF_DEFAULT_RES

    # ---- 第 1 步：参考图想要的档位（只受"参考图自身"与"跟随输出"的约束）
    res0 = plan_ref_res(n, None, want)
    ref_mp0 = n * res0 ** 2 / 1048576.0

    # ---- 第 2 步：剩下的面积才给输出；不够就把输出**等比**缩掉
    if out_mp is not None:
        # 输出同时受两条约束：自身的 2K 上限、以及"给参考图让路"后的合计余额
        room = min(OUTPUT_MAX_MP, TOTAL_MAX_MP - ref_mp0)
        if out_mp > room:
            ns = clamp_size_to(max(0.3, room), a.size)
            print("[显存预算] %d 张参考图要 %.2fMP，输出 %.2fMP 超上限 %.1fMP "
                  "⇒ 输出等比缩到 %dx%d（保持比例）"
                  % (n, ref_mp0, out_mp, room, ns[0], ns[1]))
            a.size = ns
    if res0 != a.resolution:
        print("[显存预算] resolution %d → %d：%d 张参考图统一缩到约 %.2fMP/张"
              % (a.resolution, res0, n, res0 ** 2 / 1048576.0))
    a.resolution = res0

    # ---- 兜底总检（上面理应已满足；万一漏了就再夹一次，绝不放超限请求出去）
    out_now = (a.size[0] * a.size[1] / 1048576.0) if getattr(a, "size", None) \
        else a.resolution ** 2 / 1048576.0
    ref_mp = n * a.resolution ** 2 / 1048576.0
    _cap = min(OUTPUT_MAX_MP, TOTAL_MAX_MP - ref_mp)
    if getattr(a, "size", None) and (out_now > OUTPUT_MAX_MP or ref_mp + out_now > TOTAL_MAX_MP):
        ns = clamp_size_to(max(0.3, _cap), a.size)
        print("[显存预算] 兜底：参考图 %.2fMP + 输出 %.2fMP 超上限 ⇒ 输出缩到 %dx%d"
              % (ref_mp, out_now, ns[0], ns[1]))
        a.size = ns


def apply_size(wf: dict, a: argparse.Namespace, latent_ref: list) -> list:
    """★ 指定画幅时，用 EmptyLatentImage 顶掉 TextEncodeQwenImage21 自带的 latent。

    为什么必须绕：`TextEncodeQwenImage21` 的 latent 输出被**硬编码**成「第一张参考图的
    尺寸」，节点 tooltip 原话 —— *Empty latent on the first reference image's size,
    to match with sampling as any other size shifts the edit.*
    ⇒ 想自由指定画幅，只能从外面塞一个指定宽高的空 latent。
    **官方 image_edit 模板正是这么接的**（EmptyLatentImage + ComfySwitchNode 二选一），
    这里用客户端判断代替那个 switch 节点，工作流更干净。

    ⚠️ 单位是**像素**，不要自己除 2：`EmptyLatentImage.generate` 返回的 latent 字典里
       带了 `downscale_ratio_spacial = 8`，`common_ksampler` 会把它交给
       `fix_empty_latent_channels`；后者发现 8 ≠ Qwen 的 16，就用 nearest-exact 把
       latent 空间尺寸**折半** ⇒ 最终 H/16 × W/16，解码即 W×H。通道数由
       `repeat_to_batch_size` 从 4 补齐到 64。

    ⚠️ 节点 tooltip 的另一半警告仍然有效：latent 尺寸与参考图尺寸差太多时，
       编辑内容会有偏移 ⇒ 指定画幅后建议**同时把参考图缩放到接近目标尺寸**
       （`--resolution` 会被用来统一参考图边长，别留 0）。
    """
    size = getattr(a, "size", None)
    if not size:
        return latent_ref
    # EmptyLatentImage 的 width/height 约束是 step=8，先对齐免得拿到意外尺寸
    w, h = (max(8, int(round(int(v) / 8.0)) * 8) for v in size)
    if (w, h) != tuple(int(v) for v in size):
        print("[画幅] 已对齐到 8 的倍数：%sx%s → %dx%d" % tuple(list(size) + [w, h]))
    wf["9"] = {"class_type": "EmptyLatentImage",
               "inputs": {"width": w, "height": h, "batch_size": 1}}
    return ["9", 0]


def _selfcheck_autogrow(wf: dict) -> None:
    """⛔ Autogrow 输入必须是**扁平点号键**，嵌套写法会被引擎静默丢弃。

    2026-10-03 的尺寸 bug：`"images": {"image_1": [...]}` 既不生效也不报错，
    节点收到 `{}` ⇒ 参考图全丢 + latent 退回 1024²。这里做提交前自检，
    让同类错误以后**必定炸出来**，而不是悄悄出图。
    """
    for nid, node in wf.items():
        if not isinstance(node, dict):
            continue
        for key, val in (node.get("inputs") or {}).items():
            if isinstance(val, dict):
                raise SystemExit(
                    "[工作流自检失败] 节点 %s(%s) 的输入 %r 是嵌套 dict。\n"
                    "  Autogrow（如 TextEncodeQwenImage21.images）必须展开成扁平点号键：\n"
                    "    \"images.image_1\": [\"r0\", 0]   ← 正确\n"
                    "    \"images\": {\"image_1\": [...]}   ← 会被静默忽略\n"
                    % (nid, node.get("class_type"), key))


def run_workflow(t: Tunnel, wf: dict, out_path: str, poll_s: float = 2.0,
                 timeout_s: float = 3600) -> tuple[str, int]:
    """提交工作流 → 轮询 → 取回第一张图。返回 (云卡文件名, 实际种子)。

    ⚠️ 2.1 是7B 单流DiT，10GB 卡靠 DynamicVRAM 反复换入换出，比旧版慢很多，
    默认超时给到 3600s。
    """
    _selfcheck_autogrow(wf)
    pid = api_json(t, "/prompt", {"prompt": wf, "client_id": uuid.uuid4().hex})["prompt_id"]
    t0 = time.time()
    seen = False
    last_note = 0.0
    while True:
        if time.time() - t0 > timeout_s:
            raise TimeoutError("等待出图超时（%.0fs）" % timeout_s)
        hist = api_json(t, "/history/" + pid)
        if hist:
            entry = hist.get(pid) or {}
            st = (entry.get("status") or {})
            if st.get("status_str") == "error" or st.get("completed") is False and st.get("messages"):
                msgs = st.get("messages") or []
                if any(m[0] == "execution_error" for m in msgs):
                    raise RuntimeError("ComfyUI 执行报错：%s" % json.dumps(msgs, ensure_ascii=False)[:800])
            outputs = entry.get("outputs") or {}
            imgs = [im for node in outputs.values() for im in (node.get("images") or [])]
            if imgs:
                im = imgs[0]
                q = "/view?filename=%s&subfolder=%s&type=%s" % (
                    im["filename"], im.get("subfolder", ""), im.get("type", "output"))
                raw = api_get(t, q)
                os.makedirs(os.path.dirname(os.path.abspath(out_path)) or ".", exist_ok=True)
                with open(out_path, "wb") as f:
                    f.write(raw)
                return im["filename"], _seed_of(entry)
            seen = True
        else:
            # 排队中（不在 history =还在跑或排队）
            if not seen and time.time() - last_note > 20:
                last_note = time.time()
                print("  ...生成中 %.0fs" % (time.time() - t0), flush=True)
        time.sleep(poll_s)


# ------------------------------------------------------------------ 工作流

def wf_base(a: argparse.Namespace) -> dict:
    """2.1 公共骨架：加载器 + 缓存节点。

    ⚠️ 踩过的坑：**加载器不用换**，仍是 `UNETLoader`（其显示名就叫
      "Load Diffusion Model"，只是内部class 名沿用旧叫法）。变的只有模型文件名：
      CLIPLoader 的 type 仍为 'qwen_image'，但权重换成 Qwen3-VL 8B；
      VAELoader 不变，但权重换成 64 通道 RGBA VAE。
      （我一开始误以为要换成 `LoadDiffusionModel`，云卡 970 个节点里根本没这东西。）
    """
    wf: dict = {
        "1": {"class_type": "UNETLoader",
              "inputs": {"unet_name": a.unet, "weight_dtype": "default"}},
        "2": {"class_type": "CLIPLoader",
              "inputs": {"clip_name": a.clip, "type": "qwen_image", "device": "default"}},
        "3": {"class_type": "VAELoader", "inputs": {"vae_name": a.vae}},
        # KV cache 节点（device 默认 auto：先用空闲显存，不够再落内存）
        "4": {"class_type": "QwenImage21Cache",
              "inputs": {"model": ["1", 0], "device": a.cache_device,
                         "dtype": a.cache_dtype}},
    }
    return wf


def wf_t2i(a: argparse.Namespace) -> dict:
    """纯文生图：无参考图，TextEncodeQwenImage21 输出 1024×1024 空白 latent。"""
    wf = wf_base(a)
    # ⚠️ width/height 由 prompt 里的画幅描述决定模型行为，这里靠 prompt 文本 + 事后 resize；
    #    无参考图时 TextEncodeQwenImage21 的 latent 就是 resolution × resolution 的空白。
    wf["5"] = {"class_type": "TextEncodeQwenImage21",
               "inputs": {"clip": ["2", 0], "prompt": a.prompt,
                          "negative_prompt": a.negative, "vae": ["3", 0],
                          "resolution": a.resolution}}
    latent_ref = apply_size(wf, a, ["5", 2])
    wf["6"] = {"class_type": "KSampler",
               "inputs": {"model": ["4", 0], "positive": ["5", 0], "negative": ["5", 1],
                          "latent_image": latent_ref, "seed": a.seed, "steps": a.steps,
                          "cfg": a.cfg, "sampler_name": a.sampler,
                          "scheduler": a.scheduler, "denoise": 1.0}}
    wf["7"] = {"class_type": "VAEDecode", "inputs": {"samples": ["6", 0], "vae": ["3", 0]}}
    wf["8"] = {"class_type": "SaveImage",
               "inputs": {"images": ["7", 0], "filename_prefix": a.prefix}}
    return wf


def wf_multi(a: argparse.Namespace, uploaded: list[str]) -> dict:
    """多参考图编辑：TextEncodeQwenImage21 的 images 走 Autogrow（image_1..image_16）。

    ⛔⛔ **API 格式下 Autogrow 必须写成「扁平点号键」**：`"images.image_1": ["r0", 0]`。
       写成嵌套的 `"images": {"image_1": [...]}` 会被引擎**静默丢弃**——既不生效也不报错：
         · `validate_prompt` 只遍历 `required ∪ optional`，而 Autogrow 容器名 `images`
           故意不在其中（见 `_io.Autogrow._expand_schema_for_dynamic` 首行注释
           "purposely do not include self in out_dict"）⇒ 嵌套键直接被忽略。
         · 展开逻辑判定槽位是否接线用的是 `expected_id = "images.image_1" in live_inputs`
           ⇒ 嵌套写法命中不了，`dynamic_paths["images"]` 落成 EMPTY_DICT。
         · `build_nested_inputs` 于是给节点送去 `images={}` ⇒ 参考图一张都进不去。
       2026-10-03 之前三轮实验（exp3/5/6）就栽在这里：实际是纯文生图，讨论"身份保持"毫无意义。

    ⚠️ 输出尺寸由**第一张参考图**决定（latent = image_1 尺寸，resolution=0 时四舍五入到
       32 的倍数）。要出 9:16，就把 9:16 的那张放 image_1。
    ⚠️ denoise 在 2.1 里不再走 KSampler.denoise——编辑强度由"参考图 + 指令"控制，
       KSampler.denoise 固定 1.0。
    """
    wf = wf_base(a)
    tex: dict = {"clip": ["2", 0], "prompt": a.prompt,
                 "negative_prompt": a.negative, "vae": ["3", 0],
                 "resolution": a.resolution}
    for i, name in enumerate(uploaded):
        nid = "r%d" % i
        wf[nid] = {"class_type": "LoadImage", "inputs": {"image": name}}
        # Autogrow 的键 = 「容器名.子名」，即 images.image_1 / images.image_2 …（点号，扁平）
        tex["images.image_%d" % (i + 1)] = [nid, 0]
    wf["5"] = {"class_type": "TextEncodeQwenImage21", "inputs": tex}
    latent_ref = apply_size(wf, a, ["5", 2])
    wf["6"] = {"class_type": "KSampler",
               "inputs": {"model": ["4", 0], "positive": ["5", 0], "negative": ["5", 1],
                          "latent_image": latent_ref, "seed": a.seed, "steps": a.steps,
                          "cfg": a.cfg, "sampler_name": a.sampler,
                          "scheduler": a.scheduler, "denoise": 1.0}}
    wf["7"] = {"class_type": "VAEDecode", "inputs": {"samples": ["6", 0], "vae": ["3", 0]}}
    wf["8"] = {"class_type": "SaveImage",
               "inputs": {"images": ["7", 0], "filename_prefix": a.prefix}}
    return wf


def wf_inpaint(a: argparse.Namespace, image_name: str, mask_name: str) -> dict:
    """局部重绘：原图 + mask 作为两张参考图，靠 prompt 指定改哪里。

    ⚠️ 2.1 的编辑语义和旧版完全不同：
       旧版 = VAEEncodeForInpaint + KSampler.denoise，靠 mask 的 latent 约束；
       2.1 = **把原图和 mask 当两张参考图喂进去**，用自然语言说"把这里改成…"。
       模型内部支持独立 mask 参考图（官方的"原图+掩码双输入"玩法）。
    ⚠️ 返回整图，mask 外经 VAE 往返有损 → 调用方仍需 `paste_back` 贴回原像素。
    """
    wf = wf_base(a)
    wf["r0"] = {"class_type": "LoadImage", "inputs": {"image": image_name}}
    wf["r1"] = {"class_type": "LoadImage", "inputs": {"image": mask_name}}
    wf["5"] = {"class_type": "TextEncodeQwenImage21",
               "inputs": {"clip": ["2", 0], "prompt": a.prompt,
                          "negative_prompt": a.negative, "vae": ["3", 0],
                          "resolution": a.resolution,
                          # ⛔ Autogrow 必须扁平点号键；嵌套写法会被引擎静默丢弃（见 wf_multi 注释）
                          "images.image_1": ["r0", 0], "images.image_2": ["r1", 0]}}
    latent_ref = apply_size(wf, a, ["5", 2])
    wf["6"] = {"class_type": "KSampler",
               "inputs": {"model": ["4", 0], "positive": ["5", 0], "negative": ["5", 1],
                          "latent_image": latent_ref, "seed": a.seed, "steps": a.steps,
                          "cfg": a.cfg, "sampler_name": a.sampler,
                          "scheduler": a.scheduler, "denoise": 1.0}}
    wf["7"] = {"class_type": "VAEDecode", "inputs": {"samples": ["6", 0], "vae": ["3", 0]}}
    wf["8"] = {"class_type": "SaveImage",
               "inputs": {"images": ["7", 0], "filename_prefix": a.prefix}}
    return wf


def make_rect_mask(src_path: str, rect: list[int], out_path: str, dilate: int = 0) -> str:
    """生成矩形 mask（黑底白块）。rect = [x0, y0, x1, y1]，dilate 再向外扩多少像素。"""
    from PIL import Image, ImageDraw
    with Image.open(src_path) as im:
        w, h = im.size
    x0, y0, x1, y1 = rect
    m = Image.new("L", (w, h), 0)
    ImageDraw.Draw(m).rectangle(
        [max(0, x0 - dilate), max(0, y0 - dilate),
         min(w, x1 + dilate), min(h, y1 + dilate)], fill=255)
    os.makedirs(os.path.dirname(os.path.abspath(out_path)) or ".", exist_ok=True)
    m.save(out_path)
    return out_path


def paste_back(src_path: str, result_path: str, mask_path: str, out_path: str,
               feather: float = 0.0, tail_pct: float | None = None,
               keep_alpha: bool = True) -> dict:
    """把重绘结果只贴回 mask 区域，mask 外 100% 保留原图像素。

    feather>0 时对 mask 做高斯羽化，避免硬接缝。
    keep_alpha=True（默认）时**保留源图的 alpha 通道**——透明区的 RGB 残留被抹掉、
    但透明度不变；若把 alpha 一起贴回会把透明区弄成不透明（去水印场景的经典事故）。
    tail_pct 用于「右下角去水印」这类场景的自动定量：统计结果图最后 N% 行里
    比原图变化的像素数。
    """
    from PIL import Image, ImageFilter
    import numpy as np
    base = Image.open(src_path)
    mode0 = base.mode
    base_a = base.getchannel("A") if "A" in base.getbands() else None
    base_rgb = base.convert("RGB")
    res = Image.open(result_path).convert("RGB")
    if res.size != base_rgb.size:
        res = res.resize(base_rgb.size, Image.LANCZOS)
    m = Image.open(mask_path).convert("L")
    if m.size != base_rgb.size:
        m = m.resize(base_rgb.size, Image.NEAREST)
    if feather > 0:
        m = m.filter(ImageFilter.GaussianBlur(feather))

    merged = base_rgb.copy()
    merged.paste(res, (0, 0), m)

    if keep_alpha and base_a is not None:
        out_im = merged.convert("RGBA")
        out_im.putalpha(base_a)          # alpha 原样保留
        out_im.save(out_path)
    elif mode0 == "RGB":
        merged.save(out_path)
    else:
        merged.convert(mode0).save(out_path)

    st = {}
    if tail_pct:
        a = np.asarray(base_rgb.convert("L"), dtype=np.int16)
        b = np.asarray(merged.convert("L"), dtype=np.int16)
        cut = int(base_rgb.size[1] * (1 - tail_pct / 100.0))
        st["tail_changed_px"] = int((np.abs(a - b)[cut:] > 8).sum())
        st["tail_min"] = int(a[cut:].min())
        st["tail_max"] = int(a[cut:].max())
    return st


def wf_alpha(color_name: str, mask_name: str, prefix: str,
             threshold: float | None, invert: bool) -> dict:
    """黑白蒙版法：颜色层 + 白剪影蒙版 → (二值化) → JoinImageWithAlpha → RGBA PNG。

    ⚠️ 云卡 ComfyUI 没有 `ImageThreshold`，但有 extra 节点 **`ThresholdMask(mask, value)`**。
    `value` 是 0~1 的阈值（作用在 MASK 上）。

    ⛔ **2.1 有原生 RGBA 生成能力，但这条链路仍然保留**：
       2.1 的透明图是"生成式"的（靠 prompt 描述 + 抽卡），确定性差；
       而蒙版法是数学上的，遇到"已有成图 + 想要精确透明背景"这类需求反而更靠谱、更省算力。
    """
    wf: dict = {
        "1": {"class_type": "LoadImage", "inputs": {"image": color_name}},
        "2": {"class_type": "LoadImage", "inputs": {"image": mask_name}},
        "3": {"class_type": "ImageToMask", "inputs": {"image": ["2", 0], "channel": "red"}},
    }
    mask_ref: list = ["3", 0]
    if threshold is not None:
        wf["4"] = {"class_type": "ThresholdMask",
                   "inputs": {"mask": mask_ref, "value": threshold}}
        mask_ref = ["4", 0]
    if invert:
        wf["5"] = {"class_type": "InvertMask", "inputs": {"mask": mask_ref}}
        mask_ref = ["5", 0]
    wf["6"] = {"class_type": "JoinImageWithAlpha",
               "inputs": {"image": ["1", 0], "alpha": mask_ref}}
    wf["7"] = {"class_type": "SaveImage",
               "inputs": {"images": ["6", 0], "filename_prefix": prefix}}
    return wf


# ------------------------------------------------------------------ 子命令

def cmd_info(t: Tunnel, a: argparse.Namespace) -> None:
    if a.node:
        d = api_json(t, "/object_info/" + a.node)
        if not d:
            print("节点不存在：%s" % a.node)
            return
        print(json.dumps(d, ensure_ascii=False, indent=2)[:3000])
        return
    oi = api_json(t, "/object_info")
    print("ComfyUI 节点总数：%d" % len(oi))
    for key, label in [("UNETLoader", "unet_name"), ("CLIPLoader", "clip_name"),
                       ("VAELoader", "vae_name")]:
        opts = oi.get(key, {}).get("input", {}).get("required", {}).get(label)
        print("%s.%s: %s" % (key, label, (opts or [[]])[0]))
    need = ["TextEncodeQwenImage21", "QwenImage21Cache", "KSampler", "UNETLoader",
            "ThresholdMask", "JoinImageWithAlpha", "ImageToMask", "InvertMask"]
    for cls in need:
        print("节点 %-22s %s" % (cls, "OK" if cls in oi else "缺失"))
    if "TextEncodeQwenImage21" not in oi:
        print("  ⛔ 缺 TextEncodeQwenImage21 ⇒ 云端 ComfyUI 版本 < 0.37.0，2.1 跑不了")
    names = [n for n in oi if "alpha" in n.lower() or "mask" in n.lower()]
    print("含 alpha/mask 的节点：%s" % ", ".join(sorted(names)[:40]))


def _report_alpha(path: str) -> None:
    """出图后报一下 alpha 通道实况—— 2.1 的原生 RGBA 是否真生效，就看这里。"""
    from PIL import Image, ImageStat
    with Image.open(path) as im:
        print("[完成] %s  mode=%s size=%s" % (path, im.mode, im.size))
        if im.mode == "RGBA":
            alpha = im.getchannel("A")
            hist = alpha.histogram()
            total = im.size[0] * im.size[1]
            trans, opaq = hist[0], hist[255]
            mid = total - trans - opaq
            print("       原生 alpha：全透明 %d (%.1f%%) / 全不透明 %d (%.1f%%) / 中间灰 %d (%.1f%%)" % (
                trans, 100 * trans / total, opaq, 100 * opaq / total, mid, 100 * mid / total))
            if mid > 0:
                print("       有中间灰 ⇒ 模型输出的是**连续 alpha**（2.1 原生透明，不是抠图）")
        else:
            print("       无 alpha 通道（本次输出是普通 RGB 图）")


def _common_generate(t: Tunnel, a: argparse.Namespace) -> None:
    if not a.seed or a.seed < 0:
        a.seed = int(time.time() * 1000) % (2 ** 31)
    print("[提交] %s steps=%d cfg=%s seed=%d resolution=%d cache=%s/%s" % (
        a.cmd, a.steps, a.cfg, a.seed, a.resolution, a.cache_device, a.cache_dtype))
    print("       画幅：%s" % ("%dx%d（EmptyLatentImage 指定）" % a.size
                               if getattr(a, "size", None) else "跟随 image_1 尺寸"))
    print("       模型：%s + %s + %s" % (a.unet, a.clip, a.vae))
    t0 = time.time()

    if a.cmd == "multi":
        uploaded = []
        for i, p in enumerate(a.refs):
            if not os.path.exists(p):
                sys.exit("参考图不存在：%s" % p)
            up = upload_image(t, p, ascii_name(p, "xy_m%d_" % i))
            uploaded.append(up)
            print("[上传] %s → 云卡 input/%s" % (os.path.basename(p), up))
        if len(uploaded) > 10:
            print("[警告] 传了 %d 张参考图，官方上限 10，超出会被节点截断" % len(uploaded))
        wf = wf_multi(a, uploaded)
    elif a.cmd == "i2i":
        if not os.path.exists(a.src):
            sys.exit("源图不存在：%s" % a.src)
        uploaded_name = upload_image(t, a.src, ascii_name(a.src, "xy_i2i_"))
        print("[上传] %s → 云卡 input/%s" % (os.path.basename(a.src), uploaded_name))
        wf = wf_multi(a, [uploaded_name])
    else:
        wf = wf_t2i(a)

    fn, seed = run_workflow(t, wf, a.out)
    _report_alpha(a.out)
    print("       耗时 %.0fs  种子 %d" % (time.time() - t0, seed))


def cmd_inpaint(t: Tunnel, a: argparse.Namespace) -> None:
    if not os.path.exists(a.src):
        sys.exit("源图不存在：%s" % a.src)
    if not a.seed or a.seed < 0:
        a.seed = int(time.time() * 1000) % (2 ** 31)
    if a.mask:
        if not os.path.exists(a.mask):
            sys.exit("mask 不存在：%s" % a.mask)
        mask_path = a.mask
    else:
        from PIL import Image
        with Image.open(a.src) as im:
            w, h = im.size
        r = [int(v) for v in a.mask_rect.split(",")]
        if len(r) != 4:
            sys.exit("--mask-rect 需要 x0,y0,x1,y1 四个整数")
        # 支持负值：负 = 从右/下边缘回数（-240,-130,-1,-1）
        def fix(v, n):
            return v if v >= 0 else n + v
        r = [fix(r[0], w), fix(r[1], h), fix(r[2], w), fix(r[3], h)]
        mask_path = os.path.join(os.path.dirname(os.path.abspath(a.out)), "_automsk.png")
        make_rect_mask(a.src, r, mask_path, dilate=a.dilate)
        print("[mask] 矩形 %s（原始尺寸 %dx%d）→ %s" % (r, w, h, mask_path))

    iname = upload_image(t, a.src, ascii_name(a.src, "xy_inp_src"))
    mname = upload_image(t, mask_path, ascii_name(mask_path, "xy_inp_msk"))
    print("[提交] inpaint（原图+mask 双参考图）steps=%d cfg=%s" % (a.steps, a.cfg))
    print("       指令：%s" % a.prompt)
    t0 = time.time()
    wf = wf_inpaint(a, iname, mname)
    fn, seed = run_workflow(t, wf, a.out)
    t_q = time.time() - t0

    if a.paste and os.path.exists(a.out):
        st = paste_back(a.src, a.out, mask_path, a.out, feather=a.feather, tail_pct=8)
        print("[贴回] 羽化 %.1fpx，mask 外保留原始像素" % a.feather)
        print("       右下 8%% 行内变动像素 %d，原图该区亮度 min/max=%s/%s" % (
            st.get("tail_changed_px", -1), st.get("tail_min"), st.get("tail_max")))
    from PIL import Image
    with Image.open(a.out) as im:
        mode, wh = im.mode, im.size
    print("[完成] %.0fs（含上传/轮询/回传）  %s  %dx%d mode=%s %.1fMB  seed=%d" % (
        t_q, a.out, wh[0], wh[1], mode, os.path.getsize(a.out) / 1048576, seed))


def cmd_alpha(t: Tunnel, a: argparse.Namespace) -> None:
    for p in (a.color, a.mask):
        if not os.path.exists(p):
            sys.exit("缺文件：%s" % p)
    cname = upload_image(t, a.color, "xy_color" + os.path.splitext(a.color)[1].lower())
    mname = upload_image(t, a.mask, "xy_mask" + os.path.splitext(a.mask)[1].lower())
    wf = wf_alpha(cname, mname, a.prefix, a.threshold, a.invert)
    print("[提交] alpha 合成 threshold=%s invert=%s" % (a.threshold, a.invert))
    fn, _ = run_workflow(t, wf, a.out)
    from PIL import Image, ImageStat
    with Image.open(a.out) as im:
        print("[完成] %s  mode=%s size=%s" % (a.out, im.mode, im.size))
        if im.mode == "RGBA":
            alpha = im.getchannel("A")
            hist = alpha.histogram()
            total = im.size[0] * im.size[1]
            print("       alpha 直方图：透明 %d (%.1f%%) / 不透明 %d (%.1f%%) / 中间灰 %d (%.1f%%)" % (
                hist[0], 100 * hist[0] / total, hist[255], 100 * hist[255] / total,
                total - hist[0] - hist[255],
                100 * (total - hist[0] - hist[255]) / total))
            print("       不透明区平均亮度：%.1f" % ImageStat.Stat(im.convert("RGB")).mean[0])


def default_out(prefix: str, ext: str = ".png") -> str:
    """未指定 --out 时，按 前缀_时间戳 落到本项目 out/。"""
    stamp = time.strftime("%Y%m%d-%H%M%S")
    return os.path.join(OUT_DIR, "%s_%s%s" % (prefix, stamp, ext))


def main() -> None:
    ap = argparse.ArgumentParser(description="云卡生图 CLI（Qwen-Image-2.1 / 原生RGBA / 多参考图）")
    sub = ap.add_subparsers(dest="cmd", required=True)

    def add_gen(p: argparse.ArgumentParser) -> None:
        p.add_argument("--prompt", required=True,
                       help="指令。透明图/局部编辑请把要求写进prompt（如'透明背景的图标'）")
        p.add_argument("--negative", default=DEFAULTS["negative"])
        p.add_argument("--out", default=None,
                       help="输出路径；省略则落到本项目 out/<prefix>_时间戳.png")
        p.add_argument("--width", type=int, default=DEFAULTS["width"],
                       help="输出宽（像素）。**显式给出即改用 EmptyLatentImage 指定画幅**；"
                            "不给时：有参考图 ⇒ 跟随第一张参考图；无参考图 ⇒ resolution²")
        p.add_argument("--height", type=int, default=DEFAULTS["height"],
                       help="输出高（像素），规则同 --width")
        p.add_argument("--aspect", default=None,
                       help="按比例指定画幅：%s（配合 --megapixels）" % "/".join(ASPECTS))
        p.add_argument("--megapixels", type=float, default=DEFAULTS["megapixels"],
                       help="--aspect 的总像素预算：1.0≈1024²，2K 档给 4.0")
        p.add_argument("--steps", type=int, default=DEFAULTS["steps"],
                       help="采样步数。默认 10；官方模板 25。追求细节可拉到 20~25，"
                            "出草稿用 8 也行")
        p.add_argument("--cfg", type=float, default=DEFAULTS["cfg"],
                       help="⚠️ 2.1 必须 >=1，低于 1 出噪声（官方模板给 1）")
        p.add_argument("--shift", type=float, default=DEFAULTS["shift"])
        p.add_argument("--sampler", default=DEFAULTS["sampler"])
        p.add_argument("--scheduler", default=DEFAULTS["scheduler"])
        p.add_argument("--seed", type=int, default=-1)
        p.add_argument("--prefix", default="xy")
        p.add_argument("--unet", default=DEFAULTS["unet"])
        p.add_argument("--clip", default=DEFAULTS["clip"])
        p.add_argument("--vae", default=DEFAULTS["vae"])
        p.add_argument("--resolution", type=int, default=DEFAULTS["resolution"],
                       help="参考图统一缩放的面积档（≈开方后的边长）。编辑场景默认 %d，"
                            "0=保持每张原图尺寸（只在参考图较小时才安全）；"
                            "超显存预算会自动下调并提示，见 REF_MAX_MP。"
                            "输出尺寸由 image_1 决定（t2i 无参考图时由本参数唯一定尺寸）"
                            % REF_DEFAULT_RES)
        p.add_argument("--cache-device", dest="cache_device", default=DEFAULTS["cache_device"],
                       choices=["auto", "gpu", "cpu", "off"],
                       help="prefix KV cache 放哪（auto=按剩余显存自动选，官方默认）")
        p.add_argument("--cache-dtype", dest="cache_dtype", default=DEFAULTS["cache_dtype"],
                       choices=["default", "int8", "int4"],
                       help="cache 精度；显存吃紧可用 int8")

    p1 = sub.add_parser("t2i", help="文生图"); add_gen(p1)
    p1.add_argument("--transparent", action="store_true",
                    help="透明图预设：自动追加透明背景描述词（模型仍需抽卡，非确定性）")

    p2 = sub.add_parser("i2i", help="单图编辑"); add_gen(p2)
    p2.add_argument("--src", required=True)
    p2.add_argument("--denoise", type=float, default=0.6,
                    help="⚠️ 2.1 已弃用此参数（编辑强度改由 prompt 表达），仅作兼容保留")

    p5 = sub.add_parser("multi", help="多参考图编辑（最多 10 张）"); add_gen(p5)
    # ⚠️ 必须用 action="append"，**不能**用 nargs="+"。
    #    nargs="+" 下重复写 `--refs a.png --refs b.png` 会被 argparse 覆盖成只剩最后一个
    #    （2026-10-03 踩过：明明传了两张，png_prompt 回捞只有一张，image_N 全错位）。
    #    append 之后每个 --refs 独立收集，再由 main 里拆逗号，两种写法都支持。
    p5.add_argument("--refs", action="append", required=True,
                    help="参考图路径，可重复传（--refs a.png --refs b.png），"
                         "也支持逗号分隔（--refs a.png,b.png），最多 10 张")

    p3 = sub.add_parser("alpha", help="黑白蒙版法合成透明 PNG（确定性，不走模型）")
    p3.add_argument("--color", required=True)
    p3.add_argument("--mask", required=True)
    p3.add_argument("--out", required=True)
    p3.add_argument("--prefix", default="xy_alpha")
    p3.add_argument("--threshold", type=float, default=None,
                    help="先做二值化再取 alpha（0~1），默认不二值化")
    p3.add_argument("--invert", action="store_true", help="蒙版黑白反相")

    p4 = sub.add_parser("info")
    p4.add_argument("--node", default=None)

    p6 = sub.add_parser("inpaint", help="局部重绘：原图+mask 当双参考图，靠 prompt 指明改哪")
    add_gen(p6)
    p6.add_argument("--src", required=True)
    p6.add_argument("--mask", default=None, help="mask 文件（白=要改）")
    p6.add_argument("--mask-rect", default=None,
                    help="不提供 mask 时自动生成矩形，格式 x0,y0,x1,y1；负值＝从右/下边缘回数")
    p6.add_argument("--dilate", type=int, default=0, help="矩形 mask 再外扩像素")
    p6.add_argument("--grow", type=int, default=16, help="（2.1 已不用，保留兼容）")
    p6.add_argument("--feather", type=float, default=4.0, help="贴回时的羽化半径")
    p6.add_argument("--no-paste", dest="paste", action="store_false",
                    help="不贴回，直接用云端整图输出（默认贴回）")

    a = ap.parse_args()
    # --refs 是 action="append"（见 add_gen 处的注释），每次出现给一个列表项。
    # 这里把每项内部的逗号再拆开，于是两种写法都成立：
    #   --refs a.png --refs b.png      （append 收到 ['a.png','b.png']）
    #   --refs a.png,b.png            （append 收到 ['a.png,b.png']，需要再拆）
    if a.cmd == "multi" and isinstance(a.refs, list):
        a.refs = split_refs(a.refs)
        if not a.refs:
            sys.exit("[参数错误] --refs 至少要有一张参考图")
        if len(a.refs) > 10:
            print("[警告] 传了 %d 张参考图，官方上限 10，超出会被节点截断" % len(a.refs))
        # ⚠️ 打印出来核对 —— 传错顺序会导致 <image1> 对不上号，之前踩过。
        print("[参考图 %d 张，顺序即 image1..imageN]" % len(a.refs))
        for i, p in enumerate(a.refs):
            print("   image_%-2d ← %s" % (i + 1, p))
    if getattr(a, "out", None) is None and a.cmd in ("t2i", "i2i", "inpaint", "multi"):
        a.out = default_out(a.prefix)
        print("[输出] %s" % a.out)
    # 透明图预设：把要求写进 prompt，而不是加参数（2.1 是靠文本控制输出的）
    if getattr(a, "transparent", False):
        a.prompt = a.prompt.rstrip() + (
            "，透明背景，PNG 格式，主体完整不裁切，无地面无阴影，"
            "纯色打光，居中构图")
    # ---- 画幅：显式指定就顶掉节点自带的 latent（理由见 apply_size 的注释）
    a.size = None
    if hasattr(a, "width"):
        if "--width" in sys.argv or "--height" in sys.argv:
            a.size = (a.width, a.height)
        elif a.aspect:
            a.size = calc_size(a.aspect, a.megapixels)
        if a.size:
            print("[画幅] %dx%d（显式指定 ⇒ 用 EmptyLatentImage，不再跟随参考图）" % a.size)
    # ---- 显存预算闸（★ 必须在提交前跑：超限会把云卡服务整个打死，理由见常量注释）
    # info / alpha 不送图不采样，没有 resolution 也没有预算概念 ⇒ 直接跳过。
    # （2026-10-05：`cloudgen.py info` 之前直接崩在这儿 —— info 子命令没注册这些参数。）
    if a.cmd not in ("info", "alpha"):
        guard_budget(a)
    t = Tunnel()
    try:
        if a.cmd == "info":
            cmd_info(t, a)
        elif a.cmd == "alpha":
            cmd_alpha(t, a)
        elif a.cmd == "inpaint":
            cmd_inpaint(t, a)
        else:
            _common_generate(t, a)
    finally:
        t.close()


if __name__ == "__main__":
    main()