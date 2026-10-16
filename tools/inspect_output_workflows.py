# -*- coding: utf-8 -*-
"""读 ComfyUI 输出 PNG 里嵌的工作流元数据，验证"当时到底挂没挂 LoRA"。

ComfyUI 每张图都会把自己的 prompt/workflow 写进 PNG 的 tEXt/zTXt/iTXt 块 ⇒
**这是判断"某次出图挂了哪些节点"的铁证**（比事后猜可靠）。

用法：python tools/inspect_output_workflows.py；默认扫 paths.COMFY_OUTPUT 下最近 12 张。
"""
from __future__ import annotations

import json
import os
import struct
import sys
import zlib

for _s in (sys.stdout, sys.stderr):
    try:
        _s.reconfigure(encoding="utf-8", errors="replace")
    except Exception:
        pass

import pathlib as _pl
_here = _pl.Path(__file__).resolve()
for _r in (_here.parent, *_here.parents):          # 向上找仓库根（含 paths.py）
    if (_r / "paths.py").exists():
        sys.path.insert(0, str(_r))
        break
import paths  # noqa: E402  （路径唯一权威）

OUT = str(paths.COMFY_OUTPUT)


def png_text_chunks(path):
    """返回 {keyword: text}（支持 tEXt / zTXt / iTXt）。"""
    out = {}
    with open(path, "rb") as f:
        if f.read(8) != b"\x89PNG\r\n\x1a\n":
            return out
        while True:
            head = f.read(8)
            if len(head) < 8:
                break
            ln, typ = struct.unpack(">I4s", head)
            data = f.read(ln)
            f.read(4)  # crc
            if typ == b"tEXt":
                k, _, v = data.partition(b"\x00")
                out[k.decode("latin-1")] = v.decode("latin-1")
            elif typ == b"zTXt":
                k, rest = data.split(b"\x00", 1)
                try:
                    out[k.decode("latin-1")] = zlib.decompress(rest[1:]).decode("utf-8", "replace")
                except Exception:
                    pass
            elif typ == b"iTXt":
                try:
                    k, rest = data.split(b"\x00", 1)
                    comp_flag = rest[0]
                    rest = rest[1:]
                    _lang, rest = rest.split(b"\x00", 1)
                    _trans, rest = rest.split(b"\x00", 1)
                    txt = zlib.decompress(rest) if comp_flag else rest
                    out[k.decode("latin-1")] = txt.decode("utf-8", "replace")
                except Exception:
                    pass
            if typ == b"IEND":
                break
    return out


def main() -> int:
    if not os.path.isdir(OUT):
        print("[x] 没有输出目录：%s" % OUT)
        return 1
    files = []
    for root, _dirs, names in os.walk(OUT):
        for n in names:
            if n.lower().endswith(".png"):
                p = os.path.join(root, n)
                files.append((os.path.getmtime(p), p))
    files.sort(reverse=True)
    files = files[:12]
    print("=" * 78)
    print("ComfyUI 输出图里嵌的工作流 —— 看当时到底挂没挂 LoRA")
    print("=" * 78)
    print("  扫描目录：%s（共取最近 %d 张）" % (OUT, len(files)))
    for mt, p in files:
        import datetime
        ts = datetime.datetime.fromtimestamp(mt).strftime("%m-%d %H:%M:%S")
        t = png_text_chunks(p)
        pr = t.get("prompt")
        print("\n  ── %s  %s" % (ts, os.path.basename(p)))
        if not pr:
            print("      （图里没有 prompt 元数据 —— 可能不是 ComfyUI 存的，或被剥掉了）")
            continue
        try:
            wf = json.loads(pr)
        except Exception as e:
            print("      （prompt 解析失败：%s）" % e)
            continue
        types = {}
        for nid, node in wf.items():
            if isinstance(node, dict):
                types.setdefault(node.get("class_type") or "?", []).append(nid)
        lora_nodes = {k: v for k, v in types.items() if "Lora" in k or "lora" in k}
        cache = [k for k in types if "Cache" in k]
        print("      节点数 %d；LoRA 相关节点：%s" % (len(wf), lora_nodes or "⛔ 一个都没有"))
        for nid in (lora_nodes.get("LoraLoaderModelOnly") or []):
            ins = wf[nid].get("inputs", {})
            print("         node %s：lora_name=%r strength_model=%r" % (nid, ins.get("lora_name"), ins.get("strength_model")))
        # 提示词 & 采样参数
        for nid, node in wf.items():
            ct = (node or {}).get("class_type") or ""
            ins = (node or {}).get("inputs") or {}
            if ct in ("CLIPTextEncode", "TextEncodeQwenImage21") and isinstance(ins.get("text"), str):
                print("      提示词(%s)：%r" % (nid, ins["text"][:90]))
            if ct in ("KSampler", "SamplerCustom"):
                keep = {k: ins.get(k) for k in ("steps", "cfg", "sampler_name", "scheduler", "seed", "denoise") if k in ins}
                print("      采样(%s %s)：%s" % (nid, ct, keep))
        unet = [ (nid, wf[nid].get("inputs", {}).get("unet_name")) for nid, n in wf.items()
                 if (n or {}).get("class_type") == "UNETLoader" ]
        if unet:
            print("      底模：%s" % unet)
    print()
    return 0


if __name__ == "__main__":
    sys.exit(main())
