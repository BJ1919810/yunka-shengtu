"""离线验证「训完自动关机」参数。

用法：  cd tools/verify && python t_shutdown.py
背景：  .skills/80（训练台）。AutoDL 上 `shutdown` 实测有效（2026-10-15 误触发验证）。
设计：  SHUTDOWN_BLOCK 用 `shutdown -h +1` 排程（1 分钟缓冲，可 `shutdown -c` 取消），
        比 `( sleep; shutdown ) &` 稳：排程交给 init，不依赖本脚本进程存活。
"""
import os
import subprocess
import sys
import tempfile

ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.insert(0, os.path.join(ROOT, "train_web"))
import server  # noqa: E402

MARK = "shutdown -h +1"


def main() -> int:
    fails = 0

    print("=" * 66)
    print("一、sanitize_opts：shutdown 归一化为 bool")
    print("=" * 66)
    for raw, want in [(True, True), (False, False), ("", False), (None, False), (1, True)]:
        o, _ = server.sanitize_opts({"name": "t", "shutdown": raw})
        got = o["shutdown"] if o else None
        ok = o and got is want
        if not ok:
            fails += 1
        print("  %s 输入 %-6r → %r" % ("✓" if ok else "✗", raw, got))

    print()
    print("=" * 66)
    print("二、build_job_sh：只有勾了（且非演练）才挂关机块")
    print("=" * 66)
    cases = [
        ("不勾", {"shutdown": False}, False),
        ("勾上", {"shutdown": True}, True),
        ("勾上+演练（应忽略）", {"shutdown": True, "dry": True}, False),
    ]
    for tag, extra, want in cases:
        o, err = server.sanitize_opts({"name": "t_s", **extra})
        if not o:
            print("  ✗ %s 被拒: %s" % (tag, err))
            fails += 1
            continue
        sh = server.build_job_sh(o, "T")
        has = MARK in sh
        ok = has == want
        if not ok:
            fails += 1
        print("  %s %-22s 脚本含关机块=%-6s（期望 %s）" % ("✓" if ok else "✗", tag, has, want))

    print()
    print("=" * 66)
    print("三、安全细节（顺序 + 可取消 + 兜底）")
    print("=" * 66)
    o, _ = server.sanitize_opts({"name": "t_o", "shutdown": True})
    sh = server.build_job_sh(o, "T")

    # ① done.rc 必须先写（否则网页看不到"已结束"就被断了）
    i_rc = sh.rindex("done.rc")
    i_sd = sh.index(MARK)
    ok1 = i_rc < i_sd
    print("  %s 先写 done.rc，再排程关机（网页能先看到结果）" % ("✓" if ok1 else "✗"))
    fails += 0 if ok1 else 1

    # ② 关机在 exit 之前
    i_exit = sh.rindex("exit \"$rc\"")
    ok2 = i_sd < i_exit
    print("  %s 关机块在 exit 之前" % ("✓" if ok2 else "✗"))
    fails += 0 if ok2 else 1

    # ③ 有 -h（明确 halt，不依赖裸 shutdown 的默认语义）
    ok3 = "-h +1" in sh
    print("  %s 用 -h +1（1 分钟缓冲，可 shutdown -c 取消）" % ("✓" if ok3 else "✗"))
    fails += 0 if ok3 else 1

    # ④ 找不到 shutdown 时不报错退出（|| true 兜底）
    ok4 = "command -v shutdown" in sh and "跳过自动关机" in sh
    print("  %s 找不到 shutdown 时优雅跳过（不污染 rc）" % ("✓" if ok4 else "✗"))
    fails += 0 if ok4 else 1

    # ⑤ 解锁提示：告诉怎么取消
    ok5 = "shutdown -c" in sh
    print("  %s 日志里提示了取消方法（shutdown -c）" % ("✓" if ok5 else "✗"))
    fails += 0 if ok5 else 1

    print()
    print("=" * 66)
    print("四、bash -n 语法")
    print("=" * 66)
    for val in (True, False):
        o, _ = server.sanitize_opts({"name": "t_b", "shutdown": val})
        sh = server.build_job_sh(o, "T")
        with tempfile.NamedTemporaryFile("w", suffix=".sh", delete=False,
                                         encoding="utf-8", newline="\n") as f:
            f.write(sh)
            tmp = f.name
        rc = subprocess.call(["bash", "-n", tmp])
        os.unlink(tmp)
        print("  %s shutdown=%-6s rc=%d" % ("✓" if rc == 0 else "✗", val, rc))
        fails += 0 if rc == 0 else 1

    print()
    print("=" * 66)
    print("全部通过 ✅" if fails == 0 else "有 %d 处不符 ❌" % fails)
    print("=" * 66)
    return 1 if fails else 0


if __name__ == "__main__":
    raise SystemExit(main())
