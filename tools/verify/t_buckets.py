"""离线验证 --use_aspect_ratio_buckets（分桶）参数。

用法：  cd tools/verify && python t_buckets.py
背景：  .skills/81-数据集质量与张数.md

为什么要这个开关（实测数据）：
  不开分桶 ⇒ 所有图被 BILINEAR 放大到 resolution² 再 RandomCrop。
  现有 38 张素材：放大中位 1.89 倍（最大 3.85）、裁掉中位 29.6%（最大 69.5%）。
  线稿被双线性放大 3.85 倍 = 线条糊成灰带 ⇒ LoRA 学到的就是"糊"。
  开了之后：bucket 上限 = 图片自身分辨率 ⇒ 不再造假细节。
"""
import os
import subprocess
import sys
import tempfile

ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.insert(0, os.path.join(ROOT, "train_web"))
import server  # noqa: E402

FLAG = "--use_aspect_ratio_buckets"


def main() -> int:
    fails = 0
    print("=" * 64)
    print("一、sanitize_opts：buckets 归一化为 bool")
    print("=" * 64)
    for raw, want in [(True, True), (False, False), ("", False), (None, False), (1, True)]:
        o, err = server.sanitize_opts({"name": "t", "buckets": raw})
        got = o["buckets"] if o else None
        mark = "✓" if (o and got is want) else "✗"
        if not (o and got is want):
            fails += 1
        print("  %s 输入 %-6r → %r" % (mark, raw, got))

    print()
    print("=" * 64)
    print("二、build_job_sh：只有开了才写开关")
    print("=" * 64)
    for val, want in [(True, True), (False, False)]:
        o, _ = server.sanitize_opts({"name": "t_b", "buckets": val})
        sh = server.build_job_sh(o, "T")
        has = FLAG in sh
        mark = "✓" if has == want else "✗"
        if has != want:
            fails += 1
        print("  %s buckets=%-6s ⇒ 命令行含 %s：%s" % (mark, val, FLAG, has))

    # 开了分桶时 resolution 仍要保留（实测官方脚本第 1496~1507 行：resolution
    # 只当 bucket 上限，两者共存不冲突）
    o, _ = server.sanitize_opts({"name": "t_b2", "buckets": True, "resolution": 1024})
    sh = server.build_job_sh(o, "T")
    keep_res = "--resolution 1024" in sh
    print("  %s 开分桶时仍保留 --resolution（官方只拿它当桶上限）" % ("✓" if keep_res else "✗"))
    fails += 0 if keep_res else 1

    print()
    print("=" * 64)
    print("三、bash -n 语法")
    print("=" * 64)
    for val in (True, False):
        o, _ = server.sanitize_opts({"name": "t_s", "buckets": val})
        sh = server.build_job_sh(o, "T")
        with tempfile.NamedTemporaryFile("w", suffix=".sh", delete=False,
                                         encoding="utf-8", newline="\n") as f:
            f.write(sh)
            tmp = f.name
        rc = subprocess.call(["bash", "-n", tmp])
        os.unlink(tmp)
        print("  %s buckets=%-6s rc=%d" % ("✓" if rc == 0 else "✗", val, rc))
        fails += 0 if rc == 0 else 1

    print()
    print("=" * 64)
    print("全部通过 ✅" if fails == 0 else "有 %d 处不符 ❌" % fails)
    print("=" * 64)
    return 1 if fails else 0


if __name__ == "__main__":
    raise SystemExit(main())
