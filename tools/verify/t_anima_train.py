# -*- coding: utf-8 -*-
"""本机 Anima 训练台 —— 离线回归验证（不加载模型、不占 GPU）。

用法：python tools/verify/t_anima_train.py

覆盖：
  ① 两端同步：index.html 的 KEYS 与 server.py 的 DEFAULTS **键集合必须一致**
     （这套项目吃过"改了前端忘了后端"的亏 —— 云卡那边的 LoRA 单选表要 3 处同步）
  ② dataset TOML：能解析、关键键在、**image_dir 必须是正斜杠**
  ③ 命令拼装：该有的 flag 都在、**绝不能出现 `--xformers`**（本机 xformers 是坏的）、
     `--max_train_steps` = 步数、续跑要追加 `--resume <out_dir>`
  ④ run.cmd：CRLF、有 HF_HUB_OFFLINE=1、收尾写 done.rc
  ⑤ 日志解析：kohya 的 tqdm 行 / epoch 行能读出 step/total/loss/pct
  ⑥ 数据集体检：配对、放大倍数、缺 caption、排版词、极端长宽比
  ⑦ train_anima.cmd：ASCII-only + CRLF（cmd.exe 会被 LF 骗）
"""
from __future__ import annotations

import importlib.util
import json
import os
import re
import shutil
import struct
import subprocess
import sys
import tempfile
import time
import tomllib

# ⛔ 本机控制台默认是 GBK：脚本自己把 stdout/stderr 切成 UTF-8，免得调用者忘了
#    `set PYTHONIOENCODING=utf-8` 就撞 UnicodeEncodeError（本项目老坑，见 .skills/20）
for _s in (sys.stdout, sys.stderr):
    try:
        _s.reconfigure(encoding="utf-8", errors="replace")
    except Exception:
        pass

HERE = os.path.dirname(os.path.abspath(__file__))
PROJ = os.path.dirname(os.path.dirname(HERE))          # 云卡生图/
APP = os.path.join(PROJ, "train_anima")

FAILS: list[str] = []


def check(cond, msg):
    print("  %s %s" % ("✓" if cond else "✗", msg))
    if not cond:
        FAILS.append(msg)


def load_server():
    spec = importlib.util.spec_from_file_location("anima_server", os.path.join(APP, "server.py"))
    m = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(m)
    return m


def make_fake_png(path, w, h):
    """只为体检逻辑造一个头部正确的假 PNG（服务端只读 IHDR 宽高）。"""
    with open(path, "wb") as f:
        f.write(b"\x89PNG\r\n\x1a\n" + b"\x00\x00\x00\x0d" + b"IHDR" + struct.pack(">II", w, h) + b"\x08\x02")


def main() -> int:
    print("=" * 74)
    print("  本机 Anima 训练台 · 离线回归")
    print("=" * 74)
    S = load_server()

    # ---------- ① 两端同步 ----------
    print("\n一、前后端字段同步（index.html KEYS ↔ server.py DEFAULTS）")
    html = open(os.path.join(APP, "index.html"), encoding="utf-8").read()
    m = re.search(r"const KEYS = \[(.*?)\];", html, re.S)
    check(bool(m), "index.html 里有 KEYS 数组")
    keys = set(re.findall(r'"([^"]+)"', m.group(1))) if m else set()
    defaults = set(S.DEFAULTS.keys())
    check(keys == defaults,
          "KEYS 与 DEFAULTS 键集合一致（只在前端：%s ；只在后端：%s）"
          % (sorted(keys - defaults), sorted(defaults - keys)))
    for k in sorted(defaults):
        check(('id="%s"' % k) in html, 'index.html 里有 id="%s" 的控件' % k)
    js = open(os.path.join(APP, "index.html"), encoding="utf-8").read()
    check("srcDrop" not in js and "document.getElementById" in js, "前端用 $() 取控件（无残留变量）")

    # ---------- ② TOML ----------
    print("\n二、dataset TOML")
    o = dict(S.DEFAULTS)
    o["buckets"] = True
    o["caption_dropout"] = 0.1
    tmp = tempfile.mkdtemp(prefix="anima_verify_")
    tpath = os.path.join(tmp, "dataset.toml")
    text = S.make_toml(o, tpath)
    d = tomllib.loads(text)
    check(d["general"]["enable_bucket"] is True, "enable_bucket = true")
    check(d["general"]["bucket_no_upscale"] is True, "bucket_no_upscale = true（小图不放大）")
    check(d["general"]["caption_extension"] == ".txt", 'caption_extension = ".txt"')
    check(d["general"]["resolution"] == [1024, 1024], "resolution = [1024, 1024]")
    sub = d["datasets"][0]["subsets"][0]
    check("\\" not in sub["image_dir"], "image_dir 用正斜杠（反斜杠在 TOML 里要转义，最容易写错）")
    check(sub["image_dir"] == S.STAGING.replace("\\", "/"), "image_dir 指向暂存目录")
    check(abs(sub["caption_dropout_rate"] - 0.1) < 1e-9, "caption_dropout_rate 下到 subset")
    check(d["datasets"][0]["batch_size"] == 1, "batch_size 下到 dataset")

    # ---------- ③ 命令 ----------
    print("\n三、训练命令拼装")
    args, out_dir = S.build_cmd(o, tmp)
    j = " ".join(args)

    def has(flag, val=None):
        if flag not in args:
            return False
        return val is None or (args[args.index(flag) + 1] == val)

    for flag, val in [
        ("--pretrained_model_name_or_path", S.DIT), ("--qwen3", S.TE), ("--vae", S.VAE),
        ("--network_module", "networks.lora_anima"), ("--network_dim", "16"),
        ("--network_alpha", "16"), ("--mixed_precision", "bf16"),
        ("--max_train_steps", "600"), ("--optimizer_type", "AdamW8bit"),
        ("--blocks_to_swap", "16"), ("--vae_chunk_size", "64"),
        ("--timestep_sampling", "sigmoid"), ("--learning_rate", "0.0001"),
    ]:
        check(has(flag, val), "有 %s %s" % (flag, val))
    for flag in ("--gradient_checkpointing", "--cache_latents", "--cache_text_encoder_outputs",
                 "--network_train_unet_only", "--qwen_image_vae_2d", "--save_state"):
        check(flag in args, "有 %s" % flag)
    check("--xformers" not in args, "⛔ 绝不带 --xformers（本机 xformers 是给 torch 2.2.2+cpu 编的）")
    check("--attn_mode" not in args or args[args.index("--attn_mode") + 1] == "torch",
          "注意力要么默认，要么显式 torch")
    check(out_dir == os.path.join(S.OUT_ROOT, o["name"]), "输出目录 = _out/<输出名>")
    check(args[args.index("--dataset_config") + 1].endswith("dataset.toml"), "dataset_config 指向 job 目录里的 toml")

    # ★ 「最多留几档」的换算（2026-10-16 修）：kohya 的 `--save_last_n_steps` 是**步数窗口**，
    #   不是"留几个文件"（源码 `get_remove_step_no`：`step_no - N - 1` 再向下取到间隔的倍数后删）。
    #   ⛔ 曾经直接把"个数"当窗口传 ⇒ `save_every=300 / 留 3` 窗口只有 3 步
    #      ⇒ **永远只剩最新一档**（用户 1500 步那次只剩 `-step00001500`）。
    #   现在台里换算成窗口 = (档数-1)*间隔；下面连"kohya 会删到哪一步"一起复算。
    def _kohya_remove(step_no, every, window):
        """逐字复刻 library/checkpoint_io.py:get_remove_step_no()。"""
        r = step_no - window - 1
        r -= r % every
        return None if r < 0 else r

    for (steps, every, keep, want) in ((1500, 300, 6, 1500), (1500, 300, 2, 300),
                                       (600, 200, 2, 200), (1500, 300, 1, 0)):
        o2 = dict(o)
        o2.update({"steps": steps, "save_every": every, "save_last": keep})
        a3, _ = S.build_cmd(o2, tmp)
        got = int(a3[a3.index("--save_last_n_steps") + 1])
        check(got == want, "留 %d 档（每 %d 步）⇒ --save_last_n_steps=%d（期望 %d）"
              % (keep, every, got, want))
        saved = list(range(every, steps + 1, every))
        cut = _kohya_remove(steps, every, got)
        kept = [s for s in saved if cut is None or s > cut]
        check(len(kept) == min(keep, len(saved)),
              "  → 跑到 %d 步实留 %d 档 %s（期望 %d 档）" % (steps, len(kept), kept, min(keep, len(saved))))

    o2 = dict(o); o2["resume"] = "latest"
    a2, out2 = S.build_cmd(o2, tmp)
    check("--resume" in a2 and a2[a2.index("--resume") + 1] == out2,
          "续跑时 --resume 指向**输出目录**（kohya 取里面最新的 train_state）")
    check(a2[a2.index("--max_train_steps") + 1] == str(o2["steps"]),
          "续跑时步数仍是**总步数**（填小了会立刻结束且不报错）")
    o3 = dict(o); o3["dit"] = "turbo"
    a3, _ = S.build_cmd(o3, tmp)
    check(a3[a3.index("--pretrained_model_name_or_path") + 1] == S.TURBO, "turbo 档换底模")

    # ---------- ④ run.cmd ----------
    print("\n四、run.cmd（独立进程 + done.rc）")
    run_cmd = S.make_run_cmd(o, tmp, args)
    raw = open(run_cmd, "rb").read()
    check(raw.count(b"\r\n") > 5 and raw.count(b"\n") == raw.count(b"\r\n"),
          "run.cmd 是 CRLF（cmd.exe 对 LF 会误解析）")
    rt = raw.decode("utf-8")
    check("HF_HUB_OFFLINE=1" in rt, "设了 HF_HUB_OFFLINE=1（否则收尾 save_model_card 白等几分钟）")
    check("done.rc" in rt and "%ERRORLEVEL%" in rt, "收尾把退出码写进 done.rc")
    check(S.VENV_PY in rt, "用训练 venv 的 python")
    check('cd /d "%s"' % S.SD_DIR in rt, "cwd 切到 sd-scripts（networks.lora_anima 才 import 得到）")

    # ---------- ⑤ 日志解析 ----------
    print("\n五、kohya 日志解析")
    log = ("running training / 学習経過\n"
           "  num train images * repeats / 学習画像の数×繰り返し回数: 12\n"
           "epoch 1/50\n"
           "steps:   3%|▎         | 18/600 [00:21<11:23,  1.17it/s, avr_loss=0.1342]\n")
    p = S.parse_progress(log)
    check(p.get("step") == 18 and p.get("total") == 600, "读出 step/total = 18/600")
    check(abs(p.get("pct", 0) - 3.0) < 0.05, "算出百分比 3.0%")
    check(abs(p.get("loss", 0) - 0.1342) < 1e-6, "读出 loss 0.1342")
    check(p.get("epoch") == 1 and p.get("epochs") == 50, "读出 epoch 1/50")
    check(S.parse_progress("nothing here") == {}, "没有进度行时返回空（不炸）")

    # ---------- ⑥ 数据集体检 ----------
    print("\n六、数据集体检")
    # ⛔ 绝不能动用户的真 `_staging`（2026-10-16 修：以前会把它改名成 _staging_verify_bak 再还原，
    #    中途崩了就把用户的素材留在那儿了）⇒ 把 S.STAGING 指到临时目录
    _real_stage = S.STAGING
    stage = tempfile.mkdtemp(prefix="anima_stage_check_")
    S.STAGING = stage
    try:
        make_fake_png(os.path.join(stage, "a.png"), 1200, 900)      # 短边 900，放大 ×1.14
        make_fake_png(os.path.join(stage, "b.png"), 600, 800)       # 短边 600，放大 ×1.71
        make_fake_png(os.path.join(stage, "wide.png"), 2400, 400)   # 6:1 极端条
        make_fake_png(os.path.join(stage, "c.png"), 1024, 1024)     # 与 BOM caption 配对
        open(os.path.join(stage, "a.txt"), "w", encoding="utf-8").write("jm_style, 1girl, 银发")
        open(os.path.join(stage, "b.txt"), "w", encoding="utf-8").write("没有触发词，还有竖排文字和气泡")
        # ★ BOM：记事本存的 .txt 常带它；**kohya 不去 BOM** ⇒ BOM 会粘在触发词上
        #   （2026-10-16 用户那次训练 `ss_tag_frequency` 里就是 `\ufeffjm_style`，15/15 全带）
        open(os.path.join(stage, "c.txt"), "wb").write(
            b"\xef\xbb\xbf" + "jm_style, 1girl, BOM 测试".encode("utf-8"))
        c = S.stage_check("jm_style")
        check(c["n_img"] == 4 and c["n_txt"] == 3, "数出 4 图 / 3 txt")
        check(c["n_pair"] == 3, "配对 3 组")
        check(c["n_no_cap"] == 1, "认出 1 张缺 caption（wide.png）")
        check(abs(c["ups_max"] - 2.56) < 0.02, "最大放大倍数 = 1024/400 = 2.56（实测 %.2f）" % c["ups_max"])
        check(c["n_up_over_1_2"] == 2, "放大 >1.2 倍的有 2 张（阈值照 .skills/70 §9.1-3）")
        check(c["n_bad_prefix"] == 1, "认出 1 条没以触发词开头（带 BOM 那条已剥 BOM 再比，不算误报）")
        check(c["n_bom"] == 1 and c["bom_caps"] == ["c.txt"], "★ 认出 1 条带 UTF-8 BOM 的 caption")
        check(any("BOM" in w for w in c["warnings"]), "★ warnings 明确提示「BOM 会让触发词对不上」")
        check(c["n_layout_hits"] == 1, "认出 1 条 caption 含排版词（竖排/气泡）")
        check(c["n_shape_bad"] == 1, "认出 1 张极端长宽比（6:1）")
        lf = S.list_files()
        check(lf["n_img"] == 4 and lf["n_txt"] == 3, "list_files 统计一致")
        empty = S.sanitize(dict(S.DEFAULTS))
        check(empty[0] is not None, "有图时 sanitize 通过")
    finally:
        import shutil as _sh
        S.STAGING = _real_stage
        _sh.rmtree(stage, ignore_errors=True)
        check(not os.path.isdir(_real_stage + "_verify_bak"), "★ 没在真暂存旁边留下 _verify_bak")
    o_bad = dict(S.DEFAULTS); o_bad["network_dim"] = "999"
    ok, _ = S.sanitize(o_bad)
    check(ok is None or ok["network_dim"] <= 256, "越界的 rank 被夹到合法区间")

    # ---------- ⑦ 启动器 ----------
    print("\n七、train_anima.cmd / setup_backend.py")
    cmd = os.path.join(PROJ, "train_anima.cmd")
    check(os.path.isfile(cmd), "train_anima.cmd 在项目根目录")
    b = open(cmd, "rb").read()
    check(b.count(b"\n") == b.count(b"\r\n"), "train_anima.cmd 全是 CRLF（裸 LF：%d）" % (b.count(b"\n") - b.count(b"\r\n")))
    try:
        b.decode("ascii")
        check(True, "train_anima.cmd 是 ASCII-only")
    except UnicodeDecodeError as e:
        check(False, "train_anima.cmd 有非 ASCII 字符：%s" % e)
    check("server.py" in b.decode("ascii") and "setup_backend.py" in b.decode("ascii"),
          "启动器指向 server.py 与 setup_backend.py")
    check(os.path.isfile(os.path.join(APP, "setup_backend.py")), "setup_backend.py 在")
    check("--system-site-packages" in open(os.path.join(APP, "setup_backend.py"), encoding="utf-8").read(),
          "setup 用 --system-site-packages（继承 torch，不重下 2.5GB）")

    # ---------- ⑧ 环境体检不炸 ----------
    print("\n八、环境体检（只读探测）")
    e = S.env_check()
    check(isinstance(e.get("items"), list) and len(e["items"]) >= 5, "env_check 返回 %d 项" % len(e.get("items", [])))
    print("      当前状态：%s" % ("齐备 ✅" if e["ok"] else "缺 " + "、".join(e["blockers"])))

    # ---------- ⑨ 参数名与本地 sd-scripts 对齐 ----------
    print("\n九、参数名对齐（拿本地真实 sd-scripts 源码核对，防版本差）")
    sd = S.SD_DIR
    if not os.path.isfile(os.path.join(sd, "anima_train_network.py")):
        print("  （sd-scripts 没装，跳过 —— 先跑 train_anima/setup_backend.py）")
    else:
        blob = []
        for root, dirs, files_ in os.walk(sd):
            dirs[:] = [d for d in dirs if d not in (".git", "__pycache__")]
            for f in files_:
                if f.endswith(".py"):
                    try:
                        blob.append(open(os.path.join(root, f), encoding="utf-8", errors="replace").read())
                    except OSError:
                        pass
        src = "\n".join(blob)
        flags = sorted({a for a in args if a.startswith("--")})
        missing = [f for f in flags if ('"%s"' % f) not in src and ("'%s'" % f) not in src]
        check(not missing, "所有 flag 都在本地 sd-scripts 里（缺：%s）" % missing)
        check(os.path.isfile(os.path.join(sd, "networks", "lora_anima.py")),
              "networks/lora_anima.py 在（--network_module networks.lora_anima）")
        check(os.path.isfile(os.path.join(sd, "networks", "convert_anima_lora_to_comfy.py")),
              "networks/convert_anima_lora_to_comfy.py 在（产物转 ComfyUI 版要用）")

    # ---------- ⑩ TOML 过 **训练器同款 schema** ----------
    print("\n十、TOML 过 sd-scripts 训练器同款的 schema（不是只有我能 parse）")
    env_py = S.VENV_PY
    if not os.path.isfile(env_py) or not os.path.isfile(os.path.join(sd, "library", "config_util.py")):
        print("  （训练 venv / sd-scripts 不在，跳过 —— 先跑 train_anima/setup_backend.py）")
    else:
        # ⛔ 关键：sanitizer 开关必须与 **train_network.py:924** 逐位一致
        #    `ConfigSanitizer(True, True, args.masked_loss, True)` = (dreambooth, finetuning, controlnet, dropout)
        #    最后那个 support_dropout 为 False 时，`caption_dropout_rate` 会变成
        #    "extra keys not allowed" ⇒ 开训瞬间就被拒。本项就是为了钉死这一点。
        probe = os.path.join(tempfile.gettempdir(), "anima_toml_schema_probe.py")
        with open(probe, "w", encoding="utf-8") as f:
            f.write(
                "# -*- coding: utf-8 -*-\n"
                "import argparse, dataclasses, json, sys\n"
                "sys.path.insert(0, %r)\n"
                "from library import config_util as cu\n"
                "cfg = cu.load_user_config(%r)\n"
                "san = cu.ConfigSanitizer(True, True, False, True)\n"
                "bp = cu.BlueprintGenerator(san).generate(cfg, argparse.Namespace())\n"
                "ds = bp.dataset_group.datasets[0]\n"
                "sub = ds.subsets[0]\n"
                "def pick(obj, names):\n"
                "    o = obj.params\n"
                "    return {n: getattr(o, n, None) for n in names}\n"
                "print('###' + json.dumps({\n"
                "    'dataset': pick(ds, ['resolution','enable_bucket','bucket_no_upscale',\n"
                "                        'min_bucket_reso','max_bucket_reso','batch_size']),\n"
                "    'subset': pick(sub, ['image_dir','num_repeats','caption_dropout_rate',\n"
                "                        'caption_extension','shuffle_caption']),\n"
                "}, ensure_ascii=False, default=str))\n" % (sd, tpath))
        r = subprocess.run([env_py, probe], cwd=sd, capture_output=True, text=True,
                           encoding="utf-8", errors="replace", timeout=300)
        marker = [l for l in (r.stdout or "").splitlines() if l.startswith("###")]
        if r.returncode != 0 or not marker:
            tail = ((r.stderr or "") + (r.stdout or "")).strip().splitlines()
            check(False, "sd-scripts 的 schema 拒了我们的 TOML：%s"
                  % " / ".join(x.strip() for x in tail[-3:]))
        else:
            got = json.loads(marker[-1][3:])
            d, sub2 = got["dataset"], got["subset"]
            check(list(d["resolution"] or []) == [1024, 1024], "dataset.resolution = [1024,1024]（实际 %r）" % d["resolution"])
            check(d["enable_bucket"] is True, "dataset.enable_bucket = true")
            check(d["bucket_no_upscale"] is True, "dataset.bucket_no_upscale = true（小图不放大）")
            check(d["min_bucket_reso"] == 512 and d["max_bucket_reso"] == 2048,
                  "dataset.min/max_bucket_reso = 512/2048（实际 %r/%r）" % (d["min_bucket_reso"], d["max_bucket_reso"]))
            check(d["batch_size"] == 1, "dataset.batch_size = 1")
            check(sub2["image_dir"] == S.STAGING.replace("\\", "/"), "subset.image_dir 没被改写")
            check(sub2["num_repeats"] == 1, "subset.num_repeats = 1")
            check(abs(float(sub2["caption_dropout_rate"]) - 0.1) < 1e-9,
                  "subset.caption_dropout_rate = 0.1（support_dropout 必须为 True）")
            check(sub2["caption_extension"] == ".txt", "general 的 caption_extension 继承下来了")

    # ---------- ⑪ flag 名 = 真 argparse 认的 flag ----------
    print("\n十一、flag 名对真 argparse（比源码 grep 更硬）")
    help_txt = os.path.join(tempfile.gettempdir(), "anima_help_verify.txt")
    if os.path.isfile(env_py) and os.path.isfile(os.path.join(sd, "anima_train_network.py")):
        if not os.path.isfile(help_txt) or os.path.getsize(help_txt) < 5000:
            with open(help_txt, "w", encoding="utf-8") as f:
                subprocess.run([env_py, "anima_train_network.py", "--help"], cwd=sd,
                               stdout=f, stderr=subprocess.STDOUT, timeout=600)
    if os.path.isfile(help_txt) and os.path.getsize(help_txt) > 5000:
        accepted = set(re.findall(r"(--[a-z0-9_]+)", open(help_txt, encoding="utf-8", errors="replace").read()))
        mine = sorted({a for a in args if a.startswith("--")})
        not_accepted = [f for f in mine if f not in accepted]
        check(not not_accepted, "我们发的 %d 个 flag 全被 argparse 接受（不在：%s）" % (len(mine), not_accepted))
        check(len(accepted) > 150, "argparse 选项池够大（%d 个）—— 说明确实读到了 --help" % len(accepted))
    else:
        print("  （没拿到 --help 输出，跳过）")
    # 转换脚本：必须能作为**包**跑（-m），脚本路径直跑会 ModuleNotFoundError
    conv_mod = "networks.convert_anima_lora_to_comfy"
    if os.path.isfile(env_py) and os.path.isfile(os.path.join(sd, "networks", "convert_anima_lora_to_comfy.py")):
        env2 = dict(os.environ)
        env2["PYTHONPATH"] = sd + os.pathsep + env2.get("PYTHONPATH", "")
        r = subprocess.run([env_py, "-m", conv_mod, "--help"], cwd=sd, capture_output=True, text=True,
                           env=env2, encoding="utf-8", errors="replace", timeout=300)
        h = (r.stdout or "") + (r.stderr or "")
        check("src_path" in h and "dst_path" in h, "转换脚本吃两个位置参数 src_path / dst_path（与我调用一致）")
        check("--reverse" in h, "转换脚本有 --reverse（反向转换用）")
        check('"-m", "networks.convert_anima_lora_to_comfy"' in
              open(os.path.join(APP, "server.py"), encoding="utf-8").read(),
              "server.py 用 -m 调转换脚本（脚本路径直跑会 ModuleNotFoundError: library）")
    # alpha 落盘：kohya 把 alpha 注册成 buffer ⇒ 进 state_dict ⇒ 文件里带 alpha
    la = os.path.join(sd, "networks", "lora_anima.py")
    if os.path.isfile(la):
        t = open(la, encoding="utf-8", errors="replace").read()
        check('register_buffer("alpha"' in t, "lora_anima 把 alpha 注册成 buffer（⇒ 存进 safetensors）")
        check("modules_alpha" in t, "有 per-module alpha 通道（network_alpha 会逐模块算）")

    # ---------- ⑫ ★ 产物转换真跑一遍（kohya 格式 → ComfyUI 格式）----------
    print("\n十二、产物转换端到端（造一个 kohya 格式 LoRA → 真转换 → 验键名）")
    if os.path.isfile(env_py) and os.path.isfile(os.path.join(sd, "networks", "convert_anima_lora_to_comfy.py")):
        fake_out = tempfile.mkdtemp(prefix="anima_out_")
        os.makedirs(os.path.join(fake_out, "probe"), exist_ok=True)
        src = os.path.join(fake_out, "probe", "fake.safetensors")
        mk = (
            "import sys; from safetensors.torch import save_file; import torch\n"
            "sd = {\n"
            "  'lora_unet_blocks_0_self_attn_q_proj.lora_down.weight': torch.randn(4, 8),\n"
            "  'lora_unet_blocks_0_self_attn_q_proj.lora_up.weight': torch.randn(8, 4),\n"
            "  'lora_unet_blocks_0_self_attn_q_proj.alpha': torch.tensor(16.0),\n"
            "  'lora_unet_llm_adapter_out_proj.lora_down.weight': torch.randn(4, 8),\n"
            "  'lora_unet_llm_adapter_out_proj.lora_up.weight': torch.randn(8, 4),\n"
            "  'lora_unet_llm_adapter_out_proj.alpha': torch.tensor(16.0),\n"
            "}\n"
            "save_file(sd, %r, metadata={'ss_network_dim': '16'})\n"
            "print('MADE')\n" % src)
        r0 = subprocess.run([env_py, "-c", mk], cwd=sd, capture_output=True, text=True,
                            encoding="utf-8", errors="replace", timeout=300)
        check("MADE" in (r0.stdout or ""), "造出 kohya 格式的假 LoRA（含 alpha 张量）")
        orig_out = S.OUT_ROOT
        try:
            S.OUT_ROOT = fake_out                     # convert_lora 只允许转 _out/ 下的文件
            res = S.convert_lora(src)
        finally:
            S.OUT_ROOT = orig_out
        check(res.get("ok") is True, "convert_lora 成功（错误：%s）" % (res.get("error") or res.get("log", ""))[:120])
        if res.get("ok"):
            rd = (
                "import json,sys; from safetensors import safe_open\n"
                "ks={}\n"
                "with safe_open(%r, framework='pt') as f:\n"
                "    for k in f.keys(): ks[k]=list(f.get_tensor(k).shape)\n"
                "print('###'+json.dumps(ks))\n" % res["dst"])
            r2 = subprocess.run([env_py, "-c", rd], cwd=sd, capture_output=True, text=True,
                                encoding="utf-8", errors="replace", timeout=300)
            mk2 = [l for l in (r2.stdout or "").splitlines() if l.startswith("###")]
            if not mk2:
                check(False, "读不回转换产物：%s" % (r2.stderr or "")[-200:])
            else:
                keys = json.loads(mk2[-1][3:])
                check(all(k.startswith("diffusion_model.") for k in keys),
                      "产物键名带 diffusion_model. 前缀（ComfyUI 认这个）：%s" % list(keys)[:2])
                check(any("lora_A" in k for k in keys) and any("lora_B" in k for k in keys),
                      "lora_down/up 已改名为 lora_A/lora_B")
                check(any(k.endswith(".alpha") for k in keys), "alpha 张量被原样保留（⇒ ComfyUI 里 strength 填 1.0）")
                check(any("llm_adapter" in k for k in keys),
                      "llm.adapter 的点号还原正确（llm_adapter，没有变成 llm.adapter.*）")
            check(os.path.isfile(res["dst"]) and res["dst"].endswith("_comfy.safetensors"),
                  "产物落在同目录的 *_comfy.safetensors")
        shutil.rmtree(fake_out, ignore_errors=True)

    # ---------- ⑬ ★ .cmd 前门端到端（真 cmd.exe，跑的就是用户双击的那个文件）----------
    print("\n十三、train_anima.cmd 前门端到端（真 cmd.exe + 真 HTTP）")
    launcher = os.path.join(PROJ, "train_anima.cmd")
    port = 8797
    lt = os.path.join(tempfile.gettempdir(), "anima_launcher_verify.log")
    if not os.path.isfile(launcher):
        check(False, "找不到 train_anima.cmd")
    else:
        import socket as _sock
        import urllib.request as _url

        def port_open(p):
            s = _sock.socket()
            try:
                return s.connect_ex(("127.0.0.1", p)) == 0
            finally:
                s.close()

        def kill_port(p):
            """把监听 p 的进程树杀掉 —— 本脚本自己崩过也会留孤儿，必须先清干净。"""
            try:
                r = subprocess.run(["netstat", "-ano"], capture_output=True, text=True, timeout=30)
                pids = {ln.split()[-1] for ln in (r.stdout or "").splitlines()
                        if (":%d " % p) in ln and "LISTENING" in ln}
                for pid in pids:
                    subprocess.run(["taskkill", "/F", "/T", "/PID", pid],
                                   capture_output=True, text=True, timeout=60)
            except Exception:
                pass

        kill_port(port)                       # ⛔ 上一轮崩在 time.sleep 处没走到 taskkill，孤儿就是这来的
        time.sleep(0.8)
        check(not port_open(port), "开跑前 %d 端口是空的（已清掉可能的孤儿）" % port)
        srv_proc = None
        try:                                   # ★ try/finally：本段任何异常都不许再留孤儿
            with open(lt, "wb") as lf:
                # ⚠️ 跑的是**原文件**（带 --open）⇒ 会真的弹一下浏览器 —— 这正是要验的行为，
                #    顺带能证明 --open 那条 Timer 线程没抛异常（抛了会落进这份日志）。
                srv_proc = subprocess.Popen(["cmd.exe", "/c", launcher, str(port)], cwd=PROJ,
                                            stdin=subprocess.DEVNULL, stdout=lf, stderr=subprocess.STDOUT,
                                            creationflags=0x08000000 | 0x00000200)
            up = False
            for _ in range(60):
                if port_open(port):
                    up = True
                    break
                time.sleep(0.5)
            check(up, "双击路径确实把服务起在了 127.0.0.1:%d" % port)
            if up:
                try:
                    with _url.urlopen("http://127.0.0.1:%d/" % port, timeout=15) as r:
                        body = r.read().decode("utf-8", "replace")
                        check(r.status == 200 and "Anima 训练台" in body, "首页可取（%d 字节）" % len(body))
                except Exception as e:
                    check(False, "取首页失败：%s" % e)
            time.sleep(1.5)
            try:
                log_txt = open(lt, encoding="utf-8", errors="replace").read()
            except OSError:
                log_txt = ""
            check(("打开" in log_txt) and (str(port) in log_txt), "控制台印出了要打开的网址")
            check("Traceback" not in log_txt, "启动过程没有异常（含 --open 那条线程）")
        finally:
            if srv_proc is not None:
                try:
                    subprocess.run(["taskkill", "/F", "/T", "/PID", str(srv_proc.pid)],
                                   capture_output=True, text=True, timeout=60)
                except Exception:
                    pass
            kill_port(port)
            time.sleep(1.5)
            check(not port_open(port), "关掉后 %d 端口已释放（没留孤儿进程）" % port)

        # 缺后端那条分支：用户若清掉后端就会走到它 —— 必须给提示 + 非零退出，不能静默失败
        env_missing = dict(os.environ)
        # 用「当前盘根下一个必然不存在的目录」而不是写死某个盘符 —— 换机也不会误指到真实目录
        env_missing["ANIMA_SD_DIR"] = os.path.join(os.path.sep, "__anima_no_such_sd__")
        try:
            r2 = subprocess.run(["cmd.exe", "/c", launcher, str(port + 1)], cwd=PROJ,
                                input="N\n", capture_output=True, text=True,
                                encoding="utf-8", errors="replace", env=env_missing, timeout=120)
            out2 = (r2.stdout or "") + (r2.stderr or "")
            check("not installed" in out2, "缺后端时会明确提示（不是静默失败）")
            check(r2.returncode != 0, "缺后端 + 选 N ⇒ 非零退出（实际 %r）" % r2.returncode)
        except Exception as e:
            check(False, "缺后端分支跑不动：%s" % e)

    # ---------- ⑭ ★ 参数基线对齐 + 两端默认值逐值一致 + 下拉里的优化器真能用 ----------
    print("\n十四、参数基线（.skills/70 §9.2）+ 两端默认值逐值一致 + 优化器真能用")
    # 9.2 表：33 张 ≈1MP 素材、从零重训那次的基线。本机 Anima 沿用同一套（batch/累积例外，见下）
    baseline = {
        "network_dim": 16, "network_alpha": 16, "steps": 600, "resolution": 1024,
        "buckets": True, "batch": 1, "grad_accum": 1, "learning_rate": 0.0001,
        "lr_scheduler": "constant", "lr_warmup_steps": 0, "caption_dropout": 0.1,
        "save_every": 200, "save_last": 2, "seed": 42, "optimizer": "AdamW8bit",
        "gradient_checkpointing": True, "cache_latents": True, "cache_te": True,
    }
    for k, want in baseline.items():
        got = S.DEFAULTS.get(k, "<无此键>")
        check(got == want, "基线 %-22s = %r（实际 %r）" % (k, want, got))
    check(S.DEFAULTS.get("batch") == 1 and S.DEFAULTS.get("grad_accum") == 1,
          "batch/累积 = 1/1（**有意**不同于 9.2 的 1/4 —— 那是 32G 卡配置；本机 8GB 优先时长，UI 里已注明）")
    check(abs(S.UPS_WARN - 1.2) < 1e-9, "素材放大告警阈值 = 1.2（9.1-3：≤1.2 最好）")

    # 两端默认值**逐值**比对：以前只比过"键集合"，改了后端忘了前端照样静默不一致
    def html_default(key):
        m = re.search(r"<input[^>]*\bid=\"%s\"[^>]*>" % re.escape(key), html)
        if m:
            tag = m.group(0)
            if 'type="checkbox"' in tag:
                return "checked" in tag
            v = re.search(r"\bvalue=\"([^\"]*)\"", tag)
            return v.group(1) if v else "<无 value>"
        m = re.search(r"<select[^>]*\bid=\"%s\"[^>]*>(.*?)</select>" % re.escape(key), html, re.S)
        if m:
            o = re.search(r"<option[^>]*value=\"([^\"]*)\"", m.group(1)) or \
                re.search(r"<option[^>]*>([^<]*)</option>", m.group(1))
            return o.group(1).strip() if o else "<无 option>"
        return "<没找到控件>"

    for k, want in sorted(S.DEFAULTS.items()):
        h = html_default(k)
        if isinstance(want, bool):
            check(h is want, "前端默认 %-20s 勾选 = %r（HTML：%r）" % (k, want, h))
        else:
            check(str(h) == str(want), "前端默认 %-20s = %r（HTML：%r）" % (k, want, h))

    # 下拉里的优化器必须**真的能 import** —— 放一个点了就报错的选项比不放更糟
    opts = re.findall(r"<option[^>]*>([^<]+)</option>", re.search(
        r"<select[^>]*id=\"optimizer\"[^>]*>(.*?)</select>", html, re.S).group(1))
    check(all(o not in ("Lion", "Prodigy", "DAdaptation") for o in opts),
          "下拉里没有装不上的优化器（当前 %r）" % opts)
    probe_map = {"AdamW8bit": "import bitsandbytes;print('ok')",
                 "AdamW": "import torch.optim;print('ok')",
                 "Adafactor": "from transformers.optimization import Adafactor;print('ok')"}
    if os.path.isfile(S.VENV_PY):
        for o in opts:
            o = o.strip()
            if o not in probe_map:
                check(o in ("AdamW8bit", "AdamW", "Adafactor"), "优化器 %r 有对应的可用性探针" % o)
                continue
            r = subprocess.run([S.VENV_PY, "-c", probe_map[o]], capture_output=True, text=True,
                               timeout=180)
            check("ok" in (r.stdout or ""), "优化器 %r 在训练 venv 里真的能用" % o)

    print("\n" + "=" * 74)
    print("全部通过 ✅" if not FAILS else "有问题 ❌（%d 项）" % len(FAILS))
    for f in FAILS:
        print("   ✗ " + f)
    print("=" * 74)
    return 0 if not FAILS else 1


if __name__ == "__main__":
    sys.exit(main())
