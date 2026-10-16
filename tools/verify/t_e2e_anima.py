# -*- coding: utf-8 -*-
"""端到端实测：Anima 档真出一张图（走 server.py 构建的工作流 → 直投 ComfyUI）。

⚠️ 必须和 `local_comfy.py ensure` 放在**同一条 bash 命令**里跑：
   后台任务结束时子进程会被回收，ComfyUI 会被一起带走（2026-10-15 踩过）。
"""
import json
import os
import subprocess
import sys
import time
import urllib.request

import pathlib as _pl
_here = _pl.Path(__file__).resolve()
for _r in (_here.parent, *_here.parents):          # 向上找仓库根（含 paths.py）
    if (_r / "paths.py").exists():
        sys.path.insert(0, str(_r))
        break
import paths  # noqa: E402  （路径唯一权威）

WEB = str(paths.COMFY_WEB)
sys.path.insert(0, WEB)
os.chdir(WEB)
import server as S  # noqa: E402

COMFY = "http://127.0.0.1:8188"
OPENER = urllib.request.build_opener(urllib.request.ProxyHandler({}))  # 绕代理，必须


def post(path, obj, timeout=120):
    req = urllib.request.Request(COMFY + path, data=json.dumps(obj).encode(),
                                 headers={"Content-Type": "application/json"})
    with OPENER.open(req, timeout=timeout) as r:
        return json.loads(r.read())


def get(path, timeout=60):
    with OPENER.open(COMFY + path, timeout=timeout) as r:
        return json.loads(r.read())


def main() -> int:
    import argparse
    ap = argparse.ArgumentParser()
    ap.add_argument("--mode", default="t2i", choices=("t2i", "i2i"))
    ap.add_argument("--denoise", type=float, default=0.7)
    ap.add_argument("--src", default="", help="i2i 源图（ComfyUI output 里的文件名）")
    ap.add_argument("--steps", type=int, default=30)
    ap.add_argument("--model", default="", help="覆盖扩散模型文件名")
    ap.add_argument("--control", default="none")
    ap.add_argument("--control-model", default="anima-lllite-any-test-like-v2.safetensors")
    a = ap.parse_args()

    print("=" * 70)
    print("Anima 档 端到端实测  mode=%s" % a.mode)
    print("=" * 70)

    print("\n[1/4] 启动本机 ComfyUI（ensure）…", flush=True)
    r = subprocess.run([sys.executable, "local_comfy.py", "ensure"],
                       capture_output=True, text=True, timeout=300)
    print("      " + (r.stdout or r.stderr or "").strip()[-200:])
    if r.returncode != 0:
        print("      ✗ 启动失败"); return 1

    req = {
        "prompt": "masterpiece, best quality, score_7, safe, 1girl, solo, "
                  "silver long hair, purple eyes, white blouse, standing, simple background",
        "seed": 12345, "width": 1024, "height": 1024, "steps": a.steps,
    }
    if a.model:
        req["unet"] = a.model
    if a.mode == "i2i":
        # 把源源图拷进 ComfyUI 的 input 目录（LoadImage 只认那里）
        import shutil
        src_name = a.src or "web_00151_.png"
        src_path = os.path.join(str(paths.COMFY_OUTPUT), src_name)
        if not os.path.exists(src_path):
            print("      ✗ 源图不存在：%s" % src_path); return 1
        dst = os.path.join(str(paths.COMFY_INPUT), src_name)
        shutil.copy2(src_path, dst)
        print("      源图已就位：%s" % dst)
        req.update({"mode": "i2i", "image": src_name, "refs": [src_name],
                    "denoise": a.denoise,
                    "control": a.control, "controlModel": a.control_model})

    print("\n[2/4] 用 server.py 构建 Anima 工作流…", flush=True)
    S.apply_profile("anima")
    p = S.normalize(req)
    wf = S.build_workflow(p)
    print("      节点 %d 个：%s" % (len(wf), ", ".join(sorted(n["class_type"] for n in wf.values()))))
    print("      采样器 %s / 步数 %s / cfg %s / denoise %s"
          % (wf["3"]["inputs"]["sampler_name"], wf["3"]["inputs"]["steps"],
             wf["3"]["inputs"]["cfg"], wf["3"]["inputs"]["denoise"]))

    print("\n[3/4] 投给 ComfyUI 排队…", flush=True)
    t0 = time.time()
    pid = post("/prompt", {"prompt": wf, "client_id": "anima-e2e"})["prompt_id"]
    print("      prompt_id = %s" % pid)

    print("\n[4/4] 等出图（首次要加载 4GB 模型，耐心）…", flush=True)
    hist = None
    while time.time() - t0 < 900:
        time.sleep(5)
        h = get("/history/" + pid)
        if pid in h:
            hist = h[pid]
            break
        el = int(time.time() - t0)
        if el % 30 < 5:
            print("      …已等 %ds" % el, flush=True)

    if not hist:
        print("      ✗ 超时（900s）未出图"); return 1

    st = hist.get("status", {})
    print("\n      状态: %s" % st.get("status_str"))
    if st.get("status_str") != "success":
        print("      ✗ 失败详情：")
        print(json.dumps(st, ensure_ascii=False)[:900])
        return 1

    outs = hist.get("outputs", {})
    files = []
    for node_out in outs.values():
        for im in node_out.get("images", []):
            files.append(im)
    print("      ✓ 出图成功，耗时 %.0fs" % (time.time() - t0))
    for f in files:
        path = os.path.join(str(paths.COMFY_OUTPUT), f.get("subfolder", ""), f["filename"])
        size = os.path.getsize(path) if os.path.exists(path) else -1
        print("      → %s  (%.1f KB)  存在=%s"
              % (path, size / 1024, os.path.exists(path)))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
