# -*- coding: utf-8 -*-
"""云端数据集准备：解包 zip → 铺进目标目录 → 体检 → 输出 JSON。

用法（在云卡上跑）：
    /root/autodl-tmp/train-env/bin/python prepare_dataset.py <zip路径> <目标目录> [触发词] [--keep]

它做四件事：
  1. 只从 zip 里挑出「图片 / .txt」，**只取 basename** ⇒ 天然扁平化 + 不可能路径穿越
     （所以本地拖整个文件夹、拖多层子目录，到这里都会摊平到同一层，符合训练要求）。
  2. 默认**清空**目标目录后铺入（`--keep` 则追加）。★ 清空不可逆，调用方要先确认。
  3. 体检：配对 / 尺寸 / 色彩模式 / caption 长度 / 触发词开头。
  4. stdout 最后一行输出 `###JSON###{...}`，供本机 web 端解析成结构化结果。

为什么体检放在云端而不是本机：图片字节已经在云端了，直接量真实文件最准；
本机那份只是上传前的暂存，可能和空中不一致。
"""
from __future__ import annotations

import collections
import json
import os
import shutil
import statistics
import sys
import tempfile
import zipfile

IMG_EXT = (".png", ".jpg", ".jpeg", ".webp", ".bmp")
TXT_EXT = (".txt",)
WANT_EXT = IMG_EXT + TXT_EXT


def _pick_tmpdir() -> str | None:
    """优先用数据盘（系统盘只有 ~10G）。"""
    for d in ("/root/autodl-tmp/tmp",):
        if os.path.isdir(d):
            return d
    return None


def _flatten_unzip(zip_path: str, tmp: str) -> tuple[int, int, list[str]]:
    """把 zip 里的图片/txt 摊平到 tmp。返回 (图片数, txt数, 跳过的条目)。"""
    n_img = n_txt = 0
    skipped: list[str] = []
    with zipfile.ZipFile(zip_path) as zf:
        for info in zf.infolist():
            name = info.filename.replace("\\", "/")
            if info.is_dir():
                continue
            # 跳过 macOS/Windows 的元数据垃圾
            if "__MACOSX" in name or "/._" in name:
                continue
            base = os.path.basename(name)
            if not base or base.startswith("."):
                continue
            ext = os.path.splitext(base)[1].lower()
            if ext not in WANT_EXT:
                skipped.append(name)
                continue
            # ★ 只保留 basename ⇒ 不可能写到 tmp 之外
            target = os.path.join(tmp, base)
            k = 1
            while os.path.exists(target):
                stem, e = os.path.splitext(base)
                target = os.path.join(tmp, "%s__dup%d%s" % (stem, k, e))
                k += 1
            with zf.open(info) as src, open(target, "wb") as dst:
                shutil.copyfileobj(src, dst)
            if ext in IMG_EXT:
                n_img += 1
            else:
                n_txt += 1
    return n_img, n_txt, skipped


def _check(dest: str, trigger: str) -> dict:
    """体检：全部走真实文件。"""
    try:
        from PIL import Image
    except Exception as e:  # pragma: no cover
        return {"error": "PIL 不可用: %s" % e}

    names = [f for f in os.listdir(dest) if os.path.isfile(os.path.join(dest, f))]
    imgs = sorted(f for f in names if os.path.splitext(f)[1].lower() in IMG_EXT)
    txts = sorted(f for f in names if os.path.splitext(f)[1].lower() in TXT_EXT)

    stems_i = {os.path.splitext(f)[0] for f in imgs}
    stems_t = {os.path.splitext(f)[0] for f in txts}

    sizes: list[tuple[int, int]] = []
    modes: collections.Counter = collections.Counter()
    broken: list[str] = []
    for f in imgs:
        try:
            with Image.open(os.path.join(dest, f)) as im:
                sizes.append(im.size)
                modes[im.mode] += 1
        except Exception:
            broken.append(f)

    cap_lens: list[int] = []
    not_trigger: list[str] = []
    empty_cap: list[str] = []
    for f in txts:
        with open(os.path.join(dest, f), encoding="utf-8-sig", errors="replace") as fh:
            body = fh.read().strip()
        cap_lens.append(len(body))
        if len(body) < 20:
            empty_cap.append(f)
        if trigger and not body.startswith(trigger):
            not_trigger.append(f)

    ws = [s[0] for s in sizes]
    hs = [s[1] for s in sizes]
    small = sorted({"%dx%d" % s for s in sizes if min(s) < 512})

    return {
        "dest": dest,
        "n_img": len(imgs),
        "n_txt": len(txts),
        "img_without_txt": sorted(stems_i - stems_t),
        "txt_without_img": sorted(stems_t - stems_i),
        "broken_img": broken,
        "modes": dict(modes),
        "w": [min(ws), statistics.median(ws), max(ws)] if ws else None,
        "h": [min(hs), statistics.median(hs), max(hs)] if hs else None,
        "small_sizes": small,
        "cap_len": [min(cap_lens), statistics.median(cap_lens), max(cap_lens)] if cap_lens else None,
        "cap_empty": empty_cap,
        "cap_not_leading_with_trigger": not_trigger,
        "total_mb": round(sum(os.path.getsize(os.path.join(dest, f)) for f in imgs) / 1048576, 1),
        "trigger": trigger,
        "warnings": _warnings(len(imgs), stems_i - stems_t, stems_t - stems_i, small,
                              len(not_trigger), len(txts), broken),
    }


def _warnings(n_img, miss_txt, orphan_txt, small, n_not_trig, n_txt, broken) -> list[str]:
    out = []
    if n_img == 0:
        out.append("一张图都没有 —— 目录里没有 png/jpg/jpeg/webp/bmp")
    if miss_txt:
        out.append("%d 张图没有同名 .txt（这些图会没有 caption）" % len(miss_txt))
    if orphan_txt:
        out.append("%d 个 .txt 没有对应图片（会被忽略）" % len(orphan_txt))
    if small:
        out.append("%d 种尺寸短边 <512（会被放大，细节是假的）" % len(small))
    if n_txt and n_not_trig:
        out.append("%d 个 caption 没有以触发词开头（触发词学不干净）" % n_not_trig)
    if broken:
        out.append("%d 张图片打不开（已从统计里剔除，训练时可能报错）" % len(broken))
    return out


def main() -> int:
    args = [a for a in sys.argv[1:] if not a.startswith("--")]
    keep = "--keep" in sys.argv
    if len(args) < 2:
        print(__doc__)
        return 2

    zip_path, dest = args[0], args[1]
    trigger = args[2] if len(args) > 2 else ""

    if not os.path.isfile(zip_path):
        print("[x] zip 不存在：%s" % zip_path)
        return 1

    tmp = tempfile.mkdtemp(prefix="ds_", dir=_pick_tmpdir())
    result: dict = {"ok": False, "zip": zip_path, "dest": dest, "keep": keep}
    try:
        n_img, n_txt, skipped = _flatten_unzip(zip_path, tmp)
        print("[1/3] 解包：图片 %d，txt %d，跳过 %d 个其它文件" % (n_img, n_txt, len(skipped)))

        if not keep and os.path.isdir(dest):
            old = len(os.listdir(dest))
            shutil.rmtree(dest)
            print("[2/3] 已清空旧目录（原有 %d 个文件，不可恢复）" % old)
        os.makedirs(dest, exist_ok=True)

        moved = 0
        for f in os.listdir(tmp):
            src, dst = os.path.join(tmp, f), os.path.join(dest, f)
            if os.path.exists(dst) and keep:
                stem, e = os.path.splitext(f)
                k = 1
                while os.path.exists(dst):
                    dst = os.path.join(dest, "%s__new%d%s" % (stem, k, e))
                    k += 1
            shutil.move(src, dst)
            moved += 1
        print("[3/3] 铺入 %d 个文件 → %s" % (moved, dest))

        rep = _check(dest, trigger)
        result.update({"ok": True, "check": rep})
        print()
        print("=== 数据集体检 ===")
        print("  图片 %d   txt %d   合计 %.1f MB" % (rep["n_img"], rep["n_txt"], rep["total_mb"]))
        if rep.get("w"):
            print("  宽 min/median/max: %g / %g / %g" % tuple(rep["w"]))
            print("  高 min/median/max: %g / %g / %g" % tuple(rep["h"]))
        print("  色彩模式: %s" % rep["modes"])
        if rep.get("cap_len"):
            print("  caption 字符数 min/median/max: %g / %g / %g" % tuple(rep["cap_len"]))
        if rep["img_without_txt"]:
            print("  图缺 txt: %s" % rep["img_without_txt"][:10])
        if rep["txt_without_img"]:
            print("  txt 缺图: %s" % rep["txt_without_img"][:10])
        if rep["small_sizes"]:
            print("  短边 <512: %s" % rep["small_sizes"])
        if rep["cap_not_leading_with_trigger"]:
            print("  非「%s」开头: %s" % (trigger, rep["cap_not_leading_with_trigger"][:10]))
        for w in rep["warnings"]:
            print("  [warn] " + w)
        if not rep["warnings"]:
            print("  [ok] 没发现问题")
    except Exception as e:  # noqa: BLE001
        import traceback
        traceback.print_exc()
        result["error"] = "%s: %s" % (type(e).__name__, e)
        return 1
    finally:
        shutil.rmtree(tmp, ignore_errors=True)

    print()
    print("###JSON###" + json.dumps(result, ensure_ascii=False))
    return 0


if __name__ == "__main__":
    sys.exit(main())
