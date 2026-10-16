"""验证 tools/pick_dataset.py 里的分桶复刻与云卡 diffusers **逐位一致**。

用法：  cd tools/verify && python t_pick_buckets.py

背景：  pick_dataset.py 需要在本机（没装 diffusers）预估「训练时会落哪个桶」，
        所以把 diffusers/training_utils.py 的 generate_aspect_ratio_buckets 和
        find_nearest_bucket 复刻了一份。**复刻必须可验证**，否则工具结论不可信。

参考数据：remote/_bucket_ref.json —— 2026-10-14 从云卡 train-env 的
          diffusers 直接导出（4 个 resolution 的完整桶列表 + 11 个尺寸的落桶结果）。
          复刻代码若被改动，这个脚本会立刻失败。
"""
import json
import os
import sys

ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.insert(0, os.path.join(ROOT, "tools"))
from pick_dataset import bucket_ladder, nearest_bucket  # noqa: E402

REF = os.path.join(ROOT, "remote", "_bucket_ref.json")


def main() -> int:
    if not os.path.isfile(REF):
        print("[x] 缺参考数据 %s" % REF)
        print("    重新生成：在云卡上跑")
        print("      python -c \"from diffusers.training_utils import *; ...\" 导出到该文件")
        return 2
    ref = json.load(open(REF, encoding="utf-8"))
    fails = 0

    print("=" * 68)
    print("① 桶列表逐位比对（本机复刻 vs 云卡 diffusers）")
    print("=" * 68)
    for res in sorted(k for k in ref if k.isdigit()):
        mine = [list(t) for t in bucket_ladder(int(res))]
        theirs = ref[res]
        same = mine == theirs
        if not same:
            fails += 1
        print("  %s res=%-5s 本机 %2d 个 / 云卡 %2d 个  →  %s"
              % ("✓" if same else "✗", res, len(mine), len(theirs),
                 "一致" if same else "不一致"))
        if not same:
            print("     本机:", mine)
            print("     云卡:", theirs)

    print()
    print("=" * 68)
    print("② 落桶比对（不同尺寸 → 哪个桶）")
    print("=" * 68)
    for k, v in ref["picks"].items():
        h, w = map(int, k.split("x"))
        res, expect = v
        got = list(nearest_bucket(h, w, bucket_ladder(res)))
        m = got == expect
        if not m:
            fails += 1
        print("  %s %-11s res=%-5d 期望 %-11s 得到 %-11s"
              % ("✓" if m else "✗", k, res, expect, got))

    print()
    print("=" * 68)
    print("全部逐位一致 ✅" if fails == 0 else "有 %d 处偏差 ❌" % fails)
    print("=" * 68)
    return 1 if fails else 0


if __name__ == "__main__":
    raise SystemExit(main())
