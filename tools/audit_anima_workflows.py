# -*- coding: utf-8 -*-
"""分析 Civitai 下的 Anima 工作流合集：缺哪些节点 / 依赖哪些模型。

⛔ 别用「正则抓 NODE_CLASS_MAPPINGS」来列内置节点 —— 只有 125 个，
   而 ComfyUI 实际注册几百个（Note / SolidMask / GrowMask 这些都被漏掉，
   于是误报成"缺自定义节点"）。
⇒ 必须问**跑起来的** ComfyUI 要 `/object_info`（权威全量表）。

用法：python _anima_wf_audit.py <zip路径>
"""
import collections
import glob
import json
import os
import re
import subprocess
import sys
import time
import urllib.request
import zipfile

import pathlib as _pl
_here = _pl.Path(__file__).resolve()
for _r in (_here.parent, *_here.parents):          # 向上找仓库根（含 paths.py）
    if (_r / "paths.py").exists():
        sys.path.insert(0, str(_r))
        break
import paths  # noqa: E402  （路径唯一权威）

COMFY = "http://127.0.0.1:8188"
WEB = str(paths.COMFY_WEB)
OPENER = urllib.request.build_opener(urllib.request.ProxyHandler({}))  # 绕代理，必须


def ensure_comfy() -> bool:
    """起 ComfyUI 并等它就绪。"""
    r = subprocess.run([sys.executable, "local_comfy.py", "ensure"],
                       cwd=WEB, capture_output=True, text=True, timeout=300)
    print("      " + (r.stdout or r.stderr or "").strip()[-120:])
    for _ in range(20):
        try:
            with OPENER.open(COMFY + "/system_stats", timeout=6) as f:
                json.loads(f.read())
            return True
        except Exception:
            time.sleep(2)
    return False


def main() -> int:
    zp = sys.argv[1] if len(sys.argv) > 1 else \
        str(paths.DOWNLOADS / "animaWorkflows_v90_3212228.zip")

    print("=" * 78)
    print("Anima 工作流合集 审计")
    print("=" * 78)
    if not ensure_comfy():
        print("[x] ComfyUI 起不来"); return 1

    with OPENER.open(COMFY + "/object_info", timeout=120) as f:
        info = json.loads(f.read())
    builtin = set(info.keys())
    print("ComfyUI 注册节点总数: %d" % len(builtin))

    z = zipfile.ZipFile(zp)
    allmissing = collections.Counter()
    for name in sorted(z.namelist()):
        d = json.loads(z.read(name))
        nodes = d.get("nodes") or (d.get("workflow") or {}).get("nodes") or []
        types = collections.Counter()
        models = collections.Counter()

        def walk(ns):
            for n in ns:
                t = n.get("type", "")
                if t and not re.fullmatch(r"[0-9a-f-]{36}", t):
                    types[t] += 1
                wv = n.get("widgets_values")
                # 从节点类型猜模型引用
                for key in ("ckpt_name", "lora_name", "unet_name", "clip_name",
                            "vae_name", "model_name", "name", "control_net_name"):
                    pass
                if isinstance(wv, list):
                    for v in wv:
                        if isinstance(v, str) and v.endswith(
                                (".safetensors", ".ckpt", ".pt", ".pth", ".gguf", ".bin")):
                            models[v] += 1
                if n.get("nodes"):
                    walk(n["nodes"])

        walk(nodes)
        missing = sorted(t for t in types if t not in builtin)
        allmissing.update(missing)
        print()
        print("=== %s ===" % name)
        print("   节点 %d 个 / %d 种" % (sum(types.values()), len(types)))
        if missing:
            print("   ⛔ 缺 %d 种节点：" % len(missing))
            for t in missing:
                print("        %s ×%d" % (t, types[t]))
        else:
            print("   ✅ 全部内置，可直接跑")
        if models:
            print("   引用的模型文件：")
            for m, c in models.most_common():
                print("        %-56s ×%d" % (m, c))

    print()
    print("=" * 78)
    print("汇总：这些节点在所有工作流里都缺（需要装对应插件包）")
    print("=" * 78)
    for t, c in allmissing.most_common():
        print("   %-42s 出现在 %d/5 个工作流" % (t, c))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
