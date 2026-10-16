# -*- coding: utf-8 -*-
"""路径泛化回归判据：仓库里不得再渗入「本机 / 个人」路径。

为什么要有这个脚本
    2026-10-16 首次开源时发现：29 个文件里散着 62 处硬编码绝对路径
    （`C:\\Users\\<用户名>\\...`、`D:\\<开发根>\\...`），既泄露本机信息，
    也让别人 clone 下来根本跑不动。改成统一走 `paths.py` 之后，
    **必须有机器判据防止回潮** —— 否则下次随手一写又回来了。

五组断言（全部是「真读到东西」的断言，不是看代码觉得对）
    A. **实际入库文件**里不得出现敏感文件（凭据 / 权重 / 产物 / 内部笔记）
    B. `.gitignore` 必须仍然覆盖那些敏感项（防有人手滑删规则）
    C. 任何入库的**文本文件**里不得出现 Windows 用户目录路径（`X:\\Users\\...`）
    D. 任何入库的 **.py 文件**按 AST 取字符串常量后再查一遍
       —— `"C:\\\\Users\\\\x"` 这种转义写法，裸正则查不出来，AST 能查出来
    E. 【仅本机】不得出现 `Path.home()` / 本机开发根的字面量
       （CI 上 home 是 /home/runner，这条自动退化，不影响 CI）

用法：python tools/verify/t_paths.py
"""
from __future__ import annotations

import ast
import os
import pathlib
import re
import subprocess
import sys

for _s in (sys.stdout, sys.stderr):
    try:
        _s.reconfigure(encoding="utf-8", errors="replace")
    except Exception:
        pass

ROOT = pathlib.Path(__file__).resolve().parents[2]

FAILS: list[str] = []


def check(cond, msg: str) -> None:
    print("  %s %s" % ("✓" if cond else "✗", msg))
    if not cond:
        FAILS.append(msg)


# --------------------------------------------------------------------------- #
# A. 实际入库文件不得包含敏感项
# --------------------------------------------------------------------------- #
#: (人类可读说明, 判定函数：命中即失败)
FORBIDDEN_TRACKED = [
    ("SSH 连接信息（含明文口令）",
     lambda p: p in ("cloud-comfy.config.json",)
     or p.startswith("cloud-comfy.config.json.")),
    ("模型权重 / 训练存档",
     lambda p: p.lower().endswith((".safetensors", ".ckpt", ".pt", ".pth", ".bin"))),
    ("出图产物 out/",
     lambda p: p.startswith("out/") and p != "out/.gitkeep"),
    ("源图 / 老照片 input/",
     lambda p: p.startswith("input/") and p != "input/.gitkeep"),
    ("训练台运行时目录（权重/日志/数据集）",
     lambda p: p.startswith(("train_anima/_out/", "train_anima/_jobs/",
                             "train_anima/_staging/", "train_web/_staging/",
                             "train_web/_downloads/"))),
    ("内部工作记忆与内部技术笔记",
     lambda p: p.startswith((".workbuddy/", ".skills/"))),
    ("本机私有路径覆盖",
     lambda p: p in ("paths.local.cmd", "paths.local.py")),
    ("python 缓存",
     lambda p: "__pycache__/" in p or p.endswith(".pyc")),
    ("编辑期备份",
     lambda p: ".bak-" in p or p.endswith(".bak")),
]


def tracked_files() -> list:
    try:
        out = subprocess.run(["git", "ls-files", "-z"], cwd=ROOT,
                             capture_output=True, check=True).stdout
        return [p for p in out.decode("utf-8").split("\0") if p]
    except Exception:
        # 没有 git（比如导出成 zip）时退化为遍历目录
        skip = {".git", "__pycache__", "out", "input", "node_modules"}
        res = []
        for dp, dns, fns in os.walk(ROOT):
            dns[:] = [d for d in dns if d not in skip]
            for fn in fns:
                res.append(str(pathlib.Path(dp, fn).relative_to(ROOT)).replace("\\", "/"))
        return res


def section_a(files) -> None:
    print("\n[A] 入库文件里没有敏感项")
    for label, hit in FORBIDDEN_TRACKED:
        bad = [p for p in files if hit(p)]
        check(not bad, "%s：%s" % (label, ("无" if not bad else "❌ 发现 %s" % bad[:5])))


# --------------------------------------------------------------------------- #
# B. .gitignore 仍覆盖敏感项
# --------------------------------------------------------------------------- #
REQUIRED_IGNORES = [
    "cloud-comfy.config.json",
    "cloud-comfy.config.json.bak-*",
    "out/*",
    "input/*",
    "train_anima/_out/",
    "train_anima/_jobs/",
    "train_anima/_staging/",
    "train_web/_staging/",
    "*.safetensors",
    ".workbuddy/",
    ".skills/",
    "paths.local.cmd",
]


def section_b() -> None:
    print("\n[B] .gitignore 仍覆盖敏感项")
    gi = (ROOT / ".gitignore").read_text(encoding="utf-8")
    lines = {ln.strip() for ln in gi.splitlines()}
    missing = [r for r in REQUIRED_IGNORES if r not in lines]
    check(not missing, "12 条必需规则全在：%s" % ("是" if not missing else "❌ 缺 %s" % missing))


# --------------------------------------------------------------------------- #
# C / D / E. 文件内容里不得有本机 / 个人路径
# --------------------------------------------------------------------------- #
#: Windows 用户目录：`C:\Users\<某名字>\`（占位符 `<...>` 不算）
#: 名字只取「词字符」—— 免得把反引号/全角括号/行尾说明一起吃进来
RE_WIN_USER = re.compile(
    r"[A-Za-z]:[\\/]Users[\\/](?!<)([A-Za-z0-9_.\-\u4e00-\u9fff]+)", re.I)
#: 明确是占位符/说明性写法的，放过
ALLOW_NAMES = {"runner", "user", "username", "you", "你的用户名", "...", "name"}


def _scan_text(label: str, text: str) -> list:
    bad = []
    for m in RE_WIN_USER.finditer(text):
        if m.group(1) not in ALLOW_NAMES:
            bad.append((label, m.group(0)))
    return bad


def section_c(files) -> None:
    print("\n[C] 文本文件里没有 Windows 用户目录路径")
    bad = []
    for rel in files:
        p = ROOT / rel
        try:
            text = p.read_text(encoding="utf-8")
        except (UnicodeDecodeError, OSError):
            continue                      # 二进制，跳过
        bad += _scan_text(rel, text)
    check(not bad, "无 `X:\\Users\\<真名>\\`：%s" % ("是" if not bad else "❌ %s" % bad[:5]))


def section_d(files) -> None:
    print("\n[D] .py 的字符串常量（按 AST 取值）里没有用户目录路径")
    bad = []
    for rel in files:
        if not rel.endswith(".py"):
            continue
        p = ROOT / rel
        try:
            tree = ast.parse(p.read_text(encoding="utf-8"), filename=rel)
        except (SyntaxError, UnicodeDecodeError, OSError):
            continue
        for node in ast.walk(tree):
            if isinstance(node, ast.Constant) and isinstance(node.value, str):
                bad += _scan_text("%s:%s" % (rel, getattr(node, "lineno", "?")), node.value)
    check(not bad, "AST 扫描无命中：%s" % ("是" if not bad else "❌ %s" % bad[:5]))


def section_e(files) -> None:
    print("\n[E] 仅本机：没有 Path.home() / 本机开发根的字面量")
    needles = set()
    # 开发根只要探测到了就一定是「本机特征」，任何平台都查
    sys.path.insert(0, str(ROOT))
    dev_root = None
    try:
        import paths
        dev_root = paths.DEV_ROOT
        if dev_root is not None:
            needles.add(str(dev_root))
    except Exception as e:                                    # noqa: BLE001
        print("      （paths.py 导入失败，跳过 DEV_ROOT：%s）" % e)
    # Windows 用户名只在 Windows 上有意义；CI 上 home 是 /home/runner，查了只会误报
    if os.name == "nt":
        home = pathlib.Path.home()
        needles.add(str(home))
        if len(home.name) > 3:              # 太短的名字（<3）容易误伤
            needles.add(home.name)

    if not needles:
        print("      （无本机特征串：开发根未探测到且非 Windows ⇒ 跳过）")
        return
    bad = []
    for rel in files:
        p = ROOT / rel
        try:
            text = p.read_text(encoding="utf-8")
        except (UnicodeDecodeError, OSError):
            continue
        for n in needles:
            if n in text or n.replace("\\", "/") in text:
                bad.append((rel, n))
    check(not bad, "本机特征串(%s)未出现：%s"
          % (", ".join(sorted(needles)), "是" if not bad else "❌ %s" % bad[:5]))


# --------------------------------------------------------------------------- #
# F. paths.py 自检（仅本机有意义）
# --------------------------------------------------------------------------- #
def section_f() -> None:
    print("\n[F] paths.py 解析自检")
    sys.path.insert(0, str(ROOT))
    try:
        import paths
        has_dev_root = paths.DEV_ROOT is not None
    except Exception:                                         # noqa: BLE001
        has_dev_root = False
    if not has_dev_root:
        print("      （未探测到开发根 —— 典型情况是 CI 上只 clone 了本仓库 ⇒ 跳过）")
        return
    r = subprocess.run([sys.executable, str(ROOT / "paths.py"), "--selftest"],
                       capture_output=True, text=True, encoding="utf-8", errors="replace")
    check(r.returncode == 0, "全部路径可解析（rc=%d）" % r.returncode)
    if r.returncode != 0:
        print((r.stdout or "").strip()[-1200:])


def main() -> int:
    files = tracked_files()
    print("=" * 74)
    print("路径泛化回归判据 · 云卡生图")
    print("=" * 74)
    print("入库文件数 = %d" % len(files))
    section_a(files)
    section_b()
    section_c(files)
    section_d(files)
    section_e(files)
    section_f()
    print("\n" + "=" * 74)
    if FAILS:
        print("✗ 失败 %d 项：" % len(FAILS))
        for f in FAILS:
            print("   - %s" % f)
        return 1
    print("✓ 全部通过")
    return 0


if __name__ == "__main__":
    sys.exit(main())
