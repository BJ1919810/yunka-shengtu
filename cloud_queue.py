# -*- coding: utf-8 -*-
"""云卡 ComfyUI 队列查看 / 清理。

★ 开跑之前先看一眼 —— 你（或手机上的 App）可能正在出图，两边会**排在同一个队列**里。
  不先看就开跑，等于堵自己的队（2026-10-03 就是这么把对照实验跑慢的）。

用法：
    python cloud_queue.py                       # 只看
    python cloud_queue.py --clean               # 删掉「自己」排队中的（默认前缀 xy）
    python cloud_queue.py --clean --prefix app  # 前缀可改

ComfyUI 的队列 API（都是内置的，不需要额外装东西）：
    GET  /queue                     -> {queue_running, queue_pending}
                                       每项 = [num, prompt_id, nodes_dict, extra, outputs]
                                       从 nodes_dict 能读出 filename_prefix / steps /
                                       LoadImage.image，据此认出哪个任务是自己的
    POST /queue {delete:[pid,...]}  -> 删**排队中**的
    POST /interrupt {}              -> 中断**正在执行**的（⚠️ 全局，不认提交者）
"""
from __future__ import annotations

import sys

from cloudgen import Tunnel, api_json

CLEAN = "--clean" in sys.argv
PREFIX = "xy"
if "--prefix" in sys.argv:
    i = sys.argv.index("--prefix")
    if i + 1 < len(sys.argv):
        PREFIX = sys.argv[i + 1]


def peek(item: list) -> dict:
    """从队列项里抽出人看的字段。"""
    pid = item[1]
    nodes = item[2] if len(item) > 2 else {}
    d = {"pid": pid[:8], "prefix": "?", "steps": "?", "imgs": []}
    for _nid, nd in (nodes or {}).items():
        if not isinstance(nd, dict):
            continue
        ins = nd.get("inputs") or {}
        if "filename_prefix" in ins:
            d["prefix"] = ins["filename_prefix"]
        if "steps" in ins:
            d["steps"] = ins["steps"]
        if nd.get("class_type") == "LoadImage":
            d["imgs"].append(ins.get("image"))
    return d


def main() -> None:
    t = Tunnel()
    try:
        q = api_json(t, "/queue")
        run = q.get("queue_running") or []
        pend = q.get("queue_pending") or []
        print("queue_running = %d   queue_pending = %d" % (len(run), len(pend)))
        if not run and not pend:
            print("（队列是空的，可以放心开跑）")
            return

        mine: list[tuple[str, str]] = []
        for label, items in (("RUNNING", run), ("PENDING", pend)):
            for it in items:
                d = peek(it)
                is_mine = str(d["prefix"]).startswith(PREFIX)
                print("  %-7s %s  prefix=%-10s steps=%-4s imgs=%s  %s"
                      % (label, d["pid"], d["prefix"], d["steps"], d["imgs"],
                         "<- 自己的" if is_mine else ""))
                if is_mine:
                    mine.append((label, it[1]))

        if not CLEAN:
            print("\n（只看不删：加 --clean 才会清自己排队中的任务）")
            return

        pend_ids = [p for lab, p in mine if lab == "PENDING"]
        if pend_ids:
            api_json(t, "/queue", {"delete": pend_ids})
            print("\n已从队列删除 %d 个自己的任务" % len(pend_ids))
        elif any(lab == "RUNNING" for lab, _p in mine):
            api_json(t, "/interrupt", {})
            print("\n正在跑的是自己的任务，已发送 interrupt")
        else:
            print("\n队列里没有以 %r 开头的任务，什么都没动" % PREFIX)

        q2 = api_json(t, "/queue")
        print("清理后：running = %d   pending = %d"
              % (len(q2.get("queue_running") or []), len(q2.get("queue_pending") or [])))
    finally:
        t.close()


if __name__ == "__main__":
    main()
