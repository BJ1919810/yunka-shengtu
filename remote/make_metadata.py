# -*- coding: utf-8 -*-
"""把「图片 + 同名 .txt」目录转成 datasets 能吃的 `metadata.jsonl`（JSONL，一行一个对象）。

云卡上运行：
    /root/autodl-tmp/train-env/bin/python make_metadata.py /root/autodl-tmp/dataset

规则：
    p01_1.png + p01_1.txt  →  {"file_name": "p01_1.png", "text": "<txt 内容>"}

⚠️ 两个坑（2026-10-12 实测）：
  1. **必须是 JSONL**（一行一个 JSON 对象）。写成 JSON 数组（`json.dump([...])`）会在
     pyarrow 那里报 `Column() changed from object to array in row 0`，很难看出原因。
  2. 官方脚本只会 `load_dataset(args.dataset_name, ...)`，**不传 data_dir**
     ⇒ 光有 metadata.jsonl 还不够，必须配合 `run_train_qwen21.py` 里的 datasets 补丁，
       它会把「本地目录」改写成 `load_dataset("imagefolder", data_dir=目录)`。
"""
from __future__ import annotations

import json
import os
import sys

IMG_EXT = {".png", ".jpg", ".jpeg", ".webp", ".bmp", ".tif", ".tiff"}


def main(data_dir: str) -> int:
    if not os.path.isdir(data_dir):
        sys.exit("[目录不存在] %s" % data_dir)

    names = sorted(os.listdir(data_dir))
    images = [n for n in names if os.path.splitext(n)[1].lower() in IMG_EXT]
    if not images:
        sys.exit("[没找到图片] %s" % data_dir)

    rows, no_caption = [], []
    for n in images:
        txt = os.path.splitext(n)[0] + ".txt"
        p = os.path.join(data_dir, txt)
        if os.path.isfile(p):
            with open(p, encoding="utf-8") as f:
                caption = " ".join(f.read().split())
        else:
            no_caption.append(n)
            caption = ""
        rows.append({"file_name": n, "text": caption})

    out = os.path.join(data_dir, "metadata.jsonl")
    with open(out, "w", encoding="utf-8") as f:      # ← JSONL：一行一个对象
        for r in rows:
            f.write(json.dumps(r, ensure_ascii=False) + "\n")

    print("[metadata.jsonl] %s" % out)
    print("  图片 %d 张，其中 %d 张有同名 .txt" % (len(rows), len(rows) - len(no_caption)))
    if no_caption:
        print("  ⚠️ 缺 caption 的（会变成空 caption → 等于无条件训练，风格会更牢但可控性差）：")
        for n in no_caption[:10]:
            print("     %s" % n)
        if len(no_caption) > 10:
            print("     … 共 %d 张" % len(no_caption))
    lens = sorted(len(r["text"]) for r in rows)
    if lens:
        print("  caption 长度：最短 %d / 中位 %d / 最长 %d 字"
              % (lens[0], lens[len(lens) // 2], lens[-1]))
    return 0


if __name__ == "__main__":
    sys.exit(main(sys.argv[1] if len(sys.argv) > 1 else "/root/autodl-tmp/dataset"))
