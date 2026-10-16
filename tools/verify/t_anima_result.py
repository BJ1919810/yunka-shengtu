# -*- coding: utf-8 -*-
"""验收一次训练产物（本机 Anima 训练台）。

查什么（都拿真文件/真日志，不猜）：
  ① LoRA 本体：体积 / 键数 / **alpha 张量数与取值**（决定 ComfyUI 里 strength 填多少）
  ② 作用面：每层训了哪些模块（确认含 **mlp_layer1/2** —— 管画风的那部分）
  ③ kohya 元数据：`ss_*`（rank/alpha/底模/图数/步数/lr/分辨率/优化器…）
  ④ **loss 曲线以 tensorboard 为准**（`.skills/78` 的教训：别拿 tqdm 抽样看）
  ⑤ 可选 `--convert`：另存一份 ComfyUI 键名版本，并读回校验前缀

用法：
    python tools/verify/t_anima_result.py                    # 验收 _out/ 里最新的那一版
    python tools/verify/t_anima_result.py <lora文件>          # 指定文件
    python tools/verify/t_anima_result.py --convert          # 顺手转出 *_comfy.safetensors

⚠️ 用**训练 venv** 的 python 跑（需要 safetensors / tensorboard）。
   venv 解释器位置不写死在本文件里 —— 用 `python paths.py --get ANIMA_VENV_PY` 取。
"""
from __future__ import annotations

import collections
import glob
import importlib.util
import os
import re
import sys

# ⛔ 本机控制台默认 GBK：自己切 UTF-8，免得调用者忘了设 PYTHONIOENCODING 就撞 UnicodeEncodeError
for _s in (sys.stdout, sys.stderr):
    try:
        _s.reconfigure(encoding="utf-8", errors="replace")
    except Exception:
        pass

HERE = os.path.dirname(os.path.abspath(__file__))
PROJ = os.path.dirname(os.path.dirname(HERE))
APP = os.path.join(PROJ, "train_anima")
OUT_ROOT = os.path.join(APP, "_out")

FAILS: list[str] = []


def check(cond, msg):
    print("  %s %s" % ("✓" if cond else "✗", msg))
    if not cond:
        FAILS.append(msg)


def newest_lora() -> str | None:
    files = [p for p in glob.glob(os.path.join(OUT_ROOT, "*", "*.safetensors"))
             if not os.path.basename(p).endswith("_comfy.safetensors")]
    return max(files, key=os.path.getmtime) if files else None


def load_server():
    spec = importlib.util.spec_from_file_location("anima_server_res", os.path.join(APP, "server.py"))
    m = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(m)
    return m


def show_loss() -> None:
    """tensorboard 曲线 —— 权威口径。"""
    tbs = sorted(glob.glob(os.path.join(APP, "_jobs", "*", "tb")), key=os.path.getmtime)
    if not tbs:
        print("  （没找到 tensorboard 目录，跳过）")
        return
    tb = tbs[-1]
    print("  tb 目录：%s" % tb)
    try:
        from tensorboard.backend.event_processing.event_accumulator import EventAccumulator
    except Exception as e:
        print("  （tensorboard 不可用：%s）" % e)
        return
    files = glob.glob(os.path.join(tb, "**", "events.out.*"), recursive=True)
    check(bool(files), "找到 %d 个事件文件" % len(files))
    for fp in files[:4]:
        ea = EventAccumulator(fp, size_guidance={"scalars": 0})
        ea.Reload()
        for tag in ea.Tags().get("scalars", []):
            vs = ea.Scalars(tag)
            vals = [v.value for v in vs]
            print("  %-26s 点数 %3d  首 %.4f → 末 %.4f（最小 %.4f 最大 %.4f）"
                  % (tag, len(vs), vals[0], vals[-1], min(vals), max(vals)))
            if "loss" in tag.lower() and len(vals) >= 10:
                # 看末段是否发散（末 10% 的均值 vs 最小值）
                tail = vals[max(0, int(len(vals) * 0.9)):]
                check(sum(tail) / len(tail) < min(vals) * 1.5 + 1e-6 or max(tail) < max(vals),
                      "%s 末段没有发散（末段均值 %.4f）" % (tag, sum(tail) / len(tail)))


def main() -> int:
    args = [a for a in sys.argv[1:] if not a.startswith("--")]
    do_convert = "--convert" in sys.argv
    path = args[0] if args else newest_lora()
    print("=" * 74)
    print("  训练产物验收")
    print("=" * 74)
    if not path or not os.path.isfile(path):
        print("[x] 找不到 LoRA 文件（%r）—— 先训一次，或用参数指定路径" % path)
        return 1
    path = os.path.abspath(path)
    print("  文件：%s" % path)
    print("  体积：%.1f MB" % (os.path.getsize(path) / 2 ** 20))

    from safetensors import safe_open
    with safe_open(path, framework="pt") as f:
        keys = list(f.keys())
        md = f.metadata() or {}
        alphas = [f.get_tensor(k).item() for k in keys if k.endswith(".alpha")]

    print("\n一、LoRA 本体")
    check(len(keys) > 100, "键总数 %d" % len(keys))
    check(bool(alphas), "带 %d 个 alpha 张量（⇒ ComfyUI 里 strength 填 1.0，不需要补 alpha）" % len(alphas))
    if alphas:
        vals = sorted(set(round(a, 3) for a in alphas))
        print("      alpha 取值：%s" % vals)

    print("\n二、作用面（每层训了哪些模块）")
    mods = collections.Counter(re.sub(r"^lora_unet_blocks_\d+_", "", k[:-6])
                               for k in keys if k.endswith(".alpha"))
    for m, n in sorted(mods.items()):
        print("      %-28s %d 层" % (m, n))
    check(any("mlp" in m for m in mods), "含 MLP（mlp_layer1/2）—— 画风/纹理就靠它")

    print("\n三、kohya 元数据")
    for k in ("ss_base_model_version", "ss_network_module", "ss_network_dim", "ss_network_alpha",
              "ss_num_train_images", "ss_max_train_steps", "ss_steps", "ss_num_epochs",
              "ss_learning_rate", "ss_lr_scheduler", "ss_optimizer", "ss_resolution",
              "ss_seed", "ss_caption_dropout_rate", "ss_mixed_precision", "ss_gradient_checkpointing"):
        if k in md:
            print("      %-28s = %s" % (k, md[k]))
    check(md.get("ss_base_model_version") == "anima", "底模标记 = anima")

    print("\n四、loss 曲线（tensorboard 权威口径）")
    show_loss()

    if do_convert:
        print("\n五、另存 ComfyUI 键名版本")
        S = load_server()
        r = S.convert_lora(path)
        check(r.get("ok") is True, "转换成功（%s）" % (r.get("error") or "")[:120])
        if r.get("ok"):
            with safe_open(r["dst"], framework="pt") as f:
                ks = list(f.keys())
            pref = sum(1 for k in ks if k.startswith("diffusion_model."))
            print("      产物：%s（%.1f MB，%d 键）" % (r["dst"], r.get("mb", 0), len(ks)))
            check(pref == len(ks), "全部 %d 个键都带 diffusion_model. 前缀" % len(ks))
            check(any(k.endswith(".alpha") for k in ks), "alpha 原样保留")

    print("\n" + "=" * 74)
    print("验收通过 ✅" if not FAILS else "有可疑项 ❌（%d）" % len(FAILS))
    for f in FAILS:
        print("   ✗ " + f)
    print("=" * 74)
    return 0 if not FAILS else 1


if __name__ == "__main__":
    sys.exit(main())
