"""云卡生图 · 路径解析唯一权威（single source of truth）

⛔ 本项目任何脚本都不要再硬编码本机绝对路径 —— 一律从这里取。

解析顺序
--------
1. **环境变量**（换机 / CI 时用这个；见下表）
2. **自动探测**：沿仓库目录向上找「开发根」＝ 含 `ComfyUI` / `anima` /
   `qwen-image-2.1` / `Python312` 之一的那个祖先目录。
   本项目历史上的布局是 ``<开发根>/pythonworkspace/My_program/云卡生图``，
   同级还有 ``Android-tts``（网页控制台与 APK 源码在那里面）。
3. 探测不到时**不猜**：相关常量退化为不存在路径，`selftest()` 会报出来。

环境变量
--------
=========================  ==========================================
``YUNKA_DEV_ROOT``         开发根（默认见上）
``YUNKA_ANDROID_TTS``      Android-tts 项目目录
``COMFY_DIR``              ComfyUI 目录
``COMFY_PY``               ComfyUI venv 的解释器
``ANIMA_ROOT``             Anima 资产目录
``ANIMA_SD_DIR``           kohya sd-scripts 目录（兼容既有脚本）
``ANIMA_VENV_PY``          Anima 训练 venv 的解释器（兼容既有脚本）
``ANIMA_PW_PY``            同上，旧别名（兼容 t_anima_web.py）
``ANIMA_BASE_PY``          用来建 venv 的基础解释器（兼容 setup_backend.py）
``QWEN_DIR``               Qwen-Image-2.1 目录
``NODE_EXE``               node.exe
=========================  ==========================================

命令行
------
``python paths.py``              打印全部路径 + 是否存在
``python paths.py --get NAME``   只打印某个路径（给 .cmd 用）
``python paths.py --selftest``   断言所有路径存在，否则非零退出
"""

from __future__ import annotations

import os
import shutil
import sys
from pathlib import Path

__all__ = [
    "REPO", "DEV_ROOT", "SIBLING_DIR", "ANDROID_TTS", "COMFY_WEB", "COMFY_ANDROID",
    "APK_ASSETS", "COMFY_DIR", "COMFY_OUTPUT", "COMFY_INPUT", "COMFY_PY",
    "ANIMA_ROOT", "ANIMA_SD_DIR", "ANIMA_VENV_PY", "ANIMA_DIFFUSION_DIR",
    "QWEN_DIR", "QWEN_LORA_DIR", "PY312", "NODE_EXE", "DOWNLOADS", "WIN_FONTS",
    "require", "selftest",
]

# --------------------------------------------------------------------------- #
# 仓库自身
# --------------------------------------------------------------------------- #
REPO: Path = Path(__file__).resolve().parent

#: 「开发根」候选标志：任一存在即认为该目录是开发根
_DEV_MARKERS = ("ComfyUI", "anima", "qwen-image-2.1", "Python312")


def _probe_dev_root() -> "Path | None":
    for anc in [REPO, *REPO.parents]:
        try:
            if any((anc / m).is_dir() for m in _DEV_MARKERS):
                return anc
        except OSError:
            pass
        if anc == anc.parent:          # 到盘符根了
            break
    return None


DEV_ROOT: "Path | None" = (
    Path(os.environ["YUNKA_DEV_ROOT"]).expanduser()
    if os.environ.get("YUNKA_DEV_ROOT")
    else _probe_dev_root()
)

#: 探测失败时的占位根 —— 刻意做成必然不存在的路径，避免「静默用错目录」
_MISSING = REPO / "__yunka_path_not_found__"


def _root() -> Path:
    return DEV_ROOT if DEV_ROOT is not None else _MISSING


def _env(name: str, default: Path) -> Path:
    v = os.environ.get(name)
    return Path(v).expanduser() if v else default


# --------------------------------------------------------------------------- #
# 同级项目（Android-tts：网页控制台 server.py + APK 源码）
# --------------------------------------------------------------------------- #
SIBLING_DIR: Path = REPO.parent
ANDROID_TTS: Path = _env("YUNKA_ANDROID_TTS", SIBLING_DIR / "Android-tts")
if not ANDROID_TTS.is_dir():
    _alt = _root() / "Android-tts"
    if _alt.is_dir():
        ANDROID_TTS = _alt

COMFY_WEB: Path = ANDROID_TTS / "tools" / "comfy_web"
COMFY_ANDROID: Path = ANDROID_TTS / "tools" / "comfy_android"
APK_ASSETS: Path = COMFY_ANDROID / "app" / "src" / "main" / "assets"

# --------------------------------------------------------------------------- #
# ComfyUI
# --------------------------------------------------------------------------- #
COMFY_DIR: Path = _env("COMFY_DIR", _root() / "ComfyUI")
COMFY_OUTPUT: Path = COMFY_DIR / "output"
COMFY_INPUT: Path = COMFY_DIR / "input"
COMFY_LOG: Path = COMFY_DIR / "comfy_local.log"
COMFY_PY: Path = _env("COMFY_PY", COMFY_DIR / ".venv" / "Scripts" / "python.exe")

# --------------------------------------------------------------------------- #
# Anima（底模 + 训练环境 + kohya sd-scripts）
# --------------------------------------------------------------------------- #
ANIMA_ROOT: Path = _env("ANIMA_ROOT", _root() / "anima")
ANIMA_SD_DIR: Path = _env("ANIMA_SD_DIR", ANIMA_ROOT / "sd-scripts")
ANIMA_VENV_PY: Path = _env(
    "ANIMA_VENV_PY",
    Path(os.environ["ANIMA_PW_PY"]).expanduser()      # 旧别名兼容
    if os.environ.get("ANIMA_PW_PY")
    else ANIMA_ROOT / "train-env" / "Scripts" / "python.exe",
)
ANIMA_DIFFUSION_DIR: Path = ANIMA_ROOT / "diffusion_models"

# --------------------------------------------------------------------------- #
# Qwen-Image-2.1
# --------------------------------------------------------------------------- #
QWEN_DIR: Path = _env("QWEN_DIR", _root() / "qwen-image-2.1")
QWEN_LORA_DIR: Path = QWEN_DIR / "loras"

# --------------------------------------------------------------------------- #
# 其它工具
# --------------------------------------------------------------------------- #
PY312: Path = _env(
    "ANIMA_BASE_PY",
    _root() / "Python312" / "python.exe",
)

_node = os.environ.get("NODE_EXE") or shutil.which("node")
if not _node:
    for _c in (r"C:\Program Files\nodejs\node.exe",
               "/usr/bin/node", "/usr/local/bin/node"):
        if Path(_c).exists():
            _node = _c
            break
NODE_EXE: Path = Path(_node) if _node else _MISSING / "node"

DOWNLOADS: Path = Path(os.environ.get("YUNKA_DOWNLOADS", Path.home() / "Downloads"))
WIN_FONTS: Path = Path(os.environ.get("WINDIR", r"C:\Windows")) / "Fonts"


# --------------------------------------------------------------------------- #
# 辅助
# --------------------------------------------------------------------------- #
def require(*paths: Path) -> None:
    """断言路径存在，否则抛出带可操作提示的错误。"""
    for p in paths:
        if not Path(p).exists():
            raise FileNotFoundError(
                "路径不存在: %s\n"
                "  · 若本机布局不同，请设置环境变量 YUNKA_DEV_ROOT / COMFY_DIR / "
                "ANIMA_ROOT / QWEN_DIR 之一\n"
                "  · 跑 `python paths.py` 可查看当前解析结果" % p
            )


def _all() -> "dict[str, Path]":
    return {
        "REPO": REPO,
        "DEV_ROOT": DEV_ROOT or _MISSING,
        "ANDROID_TTS": ANDROID_TTS,
        "COMFY_WEB": COMFY_WEB,
        "COMFY_ANDROID": COMFY_ANDROID,
        "APK_ASSETS": APK_ASSETS,
        "COMFY_DIR": COMFY_DIR,
        "COMFY_OUTPUT": COMFY_OUTPUT,
        "COMFY_INPUT": COMFY_INPUT,
        "COMFY_PY": COMFY_PY,
        "ANIMA_ROOT": ANIMA_ROOT,
        "ANIMA_SD_DIR": ANIMA_SD_DIR,
        "ANIMA_VENV_PY": ANIMA_VENV_PY,
        "ANIMA_DIFFUSION_DIR": ANIMA_DIFFUSION_DIR,
        "QWEN_DIR": QWEN_DIR,
        "QWEN_LORA_DIR": QWEN_LORA_DIR,
        "PY312": PY312,
        "NODE_EXE": NODE_EXE,
        "DOWNLOADS": DOWNLOADS,
        "WIN_FONTS": WIN_FONTS,
    }


def selftest() -> int:
    """打印全部路径；返回不存在项的个数（0 = 全通过）。"""
    bad = 0
    print("云卡生图 · 路径解析（DEV_ROOT=%s）" % (DEV_ROOT or "探测失败"))
    print("-" * 74)
    for name, p in _all().items():
        ok = Path(p).exists()
        if not ok:
            bad += 1
        print("  %-22s %s  %s" % (name, "OK " if ok else "缺失", p))
    print("-" * 74)
    print("不存在 %d 项 / 共 %d 项" % (bad, len(_all())))
    return bad


if __name__ == "__main__":
    args = sys.argv[1:]
    if args and args[0] == "--get":
        if len(args) < 2:
            sys.exit("用法: python paths.py --get <NAME>")
        table = _all()
        if args[1] not in table:
            sys.exit("未知名称 %r；可用: %s" % (args[1], ", ".join(table)))
        print(table[args[1]])
    elif args and args[0] == "--selftest":
        # 只断言本仓库/开发根这类「没有就肯定跑不起来」的项
        sys.exit(1 if selftest() else 0)
    else:
        selftest()
