# -*- coding: utf-8 -*-
"""从云卡（ComfyUI）出图的 PNG 里回捞提示词 / 工作流。

ComfyUI 的 SaveImage 会把**整份工作流 JSON**写进 PNG 的 `prompt` 文本块，
所以「当时用的提示词是什么」永远能从图里挖出来，不必依赖日志或记忆。

用法：
    python png_prompt.py 图.png                 # 正/负提示词 + 关键参数（人读）
    python png_prompt.py 图.png --json# 完整工作流 JSON
    python png_prompt.py 图.png --out a.txt     # 另存为文本
    python png_prompt.py 某目录 --all           # 目录下所有 png 各出一份 .prompt.txt
"""
from __future__ import annotations

import argparse
import json
import os
import sys

from PIL import Image

PARAM_KEYS = ["seed", "steps", "cfg", "sampler_name", "scheduler", "denoise"]


def load_workflow(path: str) -> dict | None:
    with Image.open(path) as im:
        raw = im.info.get("prompt")
    if not raw:
        return None
    return json.loads(raw)


def _text_of(wf: dict, ref) -> str:
    """KSampler 的 positive/negative 引用 → 顺着链找上游编码节点的文本。

    ⚠️ 2.1 用的是 `TextEncodeQwenImage21`（不再是 CLIPTextEncode），
       两个都要认，否则回捞永远显示「未找到」。
    """
    if not isinstance(ref, list) or not ref:
        return "（未接）"
    queue, seen = [ref[0]], set()
    while queue:
        nid = queue.pop(0)
        if nid in seen or nid not in wf:
            continue
        seen.add(nid)
        node = wf[nid]
        ct = node.get("class_type")
        ins = node.get("inputs") or {}
        if ct == "CLIPTextEncode":
            return ins.get("text", "")
        if ct == "TextEncodeQwenImage21":
            # 同一节点有 positive(0) / negative(1) 两个输出，要按下标取值，否则负向会显示成正向
            neg = (nid == ref[0] and len(ref) > 1 and ref[1] == 1)
            return ins.get("negative_prompt" if neg else "prompt", "")
        for v in ins.values():   # 中间节点（条件/重绘）继续往上找
            if isinstance(v, list) and v and isinstance(v[0], str):
                queue.append(v[0])
    return "（未找到编码节点）"


def summarize(wf: dict) -> str:
    """把工作流压成人读的几行：正/负提示词 + 采样参数 + 尺寸 + 参考图接线。"""
    out: list[str] = []
    sampler = next((n for n in wf.values() if n.get("class_type") == "KSampler"), None)
    if sampler:
        ins = sampler["inputs"]
        out.append("【正提示词】" + (_text_of(wf, ins.get("positive")) or "（空）"))
        out.append("【负提示词】" + (_text_of(wf, ins.get("negative")) or "（空）"))
        ps = "  ".join("%s=%s" % (k, ins.get(k)) for k in PARAM_KEYS if k in ins)
        out.append("【采样】" + ps)
        lat = ins.get("latent_image")
        if isinstance(lat, list) and len(lat) == 2:
            src = wf.get(lat[0], {}).get("class_type", "?")
            out.append("【latent 来源】%s.%d（%s）" % (lat[0], lat[1], src))
    for n in wf.values():
        ct = n.get("class_type")
        if ct in ("EmptySD3LatentImage", "EmptyLatentImage"):
            i = n["inputs"]
            out.append("【尺寸】%dx%d（独立 latent）" % (i["width"], i["height"]))
        elif ct == "LoadImage":
            out.append("【输入图】%s（i2i / 重绘）" % n["inputs"].get("image"))
    # ⚠️ 2.1 的参考图走 Autogrow，在 API 格式里是「扁平点号键」images.image_N。
    #    写成嵌套的 "images": {...} 会被引擎**静默丢弃**，等于参考图一张都没传。
    #    这一节就是用来当场抓这个坑的（2026-10-03 的尺寸 bug 根因）。
    for nid, n in wf.items():
        if n.get("class_type") != "TextEncodeQwenImage21":
            continue
        i = n.get("inputs") or {}
        dotted = sorted(k for k in i if k.startswith("images."))
        out.append("【2.1 编码节点 %s】resolution=%s  参考图 %d 张" % (
            nid, i.get("resolution"), len(dotted)))
        if dotted:
            out.append("       " + "  ".join("%s←%s" % (k, i[k]) for k in dotted))
        if "images" in i:
            out.append('       ‼️ 存在嵌套的 "images" 键：引擎会静默忽略 ⇒ 参考图等于没传')
    models = []
    for n in wf.values():
        i = n.get("inputs") or {}
        for k in ("unet_name", "clip_name", "vae_name"):
            if k in i:
                models.append(i[k])
    if models:
        out.append("【模型】" + " / ".join(models))
    return "\n".join(out)


def main() -> None:
    ap = argparse.ArgumentParser(description="回捞 ComfyUI PNG 内嵌的提示词/工作流")
    ap.add_argument("path", help="PNG 文件或目录")
    ap.add_argument("--json", action="store_true", help="打印完整工作流 JSON")
    ap.add_argument("--out", default=None, help="另存路径（默认打印到屏幕）")
    ap.add_argument("--all", action="store_true", help="目录模式：每个 png 生成同名 .prompt.txt")
    a = ap.parse_args()

    targets: list[str] = []
    if os.path.isdir(a.path):
        targets = [os.path.join(a.path, f) for f in sorted(os.listdir(a.path))
                   if f.lower().endswith(".png")]
        if not a.all:
            sys.exit("目录模式需显式加 --all（会给每个 png 写一份 .prompt.txt）")
    else:
        targets = [a.path]

    for p in targets:
        wf = load_workflow(p)
        if wf is None:
            print("%s  →  无内嵌工作流（不是 ComfyUI 出的图，或元数据被剥）" % p)
            continue
        body = json.dumps(wf, ensure_ascii=False, indent=2) if a.json else summarize(wf)
        if a.all or a.out:
            dst = a.out or (os.path.splitext(p)[0] + ".prompt.txt")
            with open(dst, "w", encoding="utf-8") as f:
                f.write(body + "\n")
            print("%s  →  %s" % (p, dst))
        else:
            print("=" * 70)
            print(p)
            print("=" * 70)
            print(body)


if __name__ == "__main__":
    main()
