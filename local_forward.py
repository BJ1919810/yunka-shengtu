# -*- coding: utf-8 -*-
"""SSH 本地端口转发：127.0.0.1:<本地> → 云卡 127.0.0.1:<远端 ComfyUI>。

用途：让只能吃 http 地址的工具（如 comfy_android/test_api.js、浏览器）直连云端
ComfyUI，而不必在云卡上开公网端口。

用法：
    python local_forward.py [本地端口] [远端端口]     # 默认 5263 → 6006
    Ctrl+C 退出。

口令来源与 cloudgen.py 一致：cloud-comfy.config.json 或 COMFY_SSH_* 环境变量。
"""
from __future__ import annotations

import json
import os
import select
import socket
import sys
import threading
import time

import paramiko

HERE = os.path.dirname(os.path.abspath(__file__))
LOCAL = int(sys.argv[1]) if len(sys.argv) > 1 else 5263
REMOTE_ARG = int(sys.argv[2]) if len(sys.argv) > 2 else 0


def load_cfg() -> dict:
    cfg: dict = {}
    p = os.path.join(HERE, "cloud-comfy.config.json")
    if os.path.exists(p):
        with open(p, encoding="utf-8") as f:
            cfg = json.load(f)
    out = {
        "host": os.environ.get("COMFY_SSH_HOST") or cfg.get("ssh_host", ""),
        "port": int(os.environ.get("COMFY_SSH_PORT") or cfg.get("ssh_port", 22)),
        "user": os.environ.get("COMFY_SSH_USER") or cfg.get("ssh_user", "root"),
        "pass": os.environ.get("COMFY_SSH_PASSWORD") or cfg.get("ssh_password", "")
                or os.environ.get("COMFY_SSH_PASS") or "",
        "comfy": int(os.environ.get("COMFY_REMOTE_PORT") or cfg.get("remote_comfy_port", 6006)),
    }
    if not out["host"] or not out["pass"]:
        sys.exit("[配置缺失] 需要 ssh_host / ssh_password（cloud-comfy.config.json 或 COMFY_SSH_*）")
    return out


def connect(cfg: dict) -> paramiko.SSHClient:
    """AutoDL 的 sshd 会回纯文本 `Exceeded MaxStartups`（不是 banner）⇒ 退避重连。"""
    last: Exception | None = None
    for i in range(6):
        ssh = paramiko.SSHClient()
        ssh.set_missing_host_key_policy(paramiko.AutoAddPolicy())
        try:
            ssh.connect(cfg["host"], cfg["port"], cfg["user"], cfg["pass"],
                        timeout=25, look_for_keys=False, allow_agent=False)
            return ssh
        except Exception as e:                     # noqa: BLE001
            last = e
            try:
                ssh.close()
            except Exception:                      # noqa: BLE001
                pass
            if i == 5:
                sys.exit("[SSH 连接失败，重试 6 次仍不通] %s: %s" % (type(e).__name__, e))
            time.sleep(0.4 * (i + 1))
    raise AssertionError(last)


def pump(src: socket.socket, dst) -> None:
    try:
        while True:
            r, _, _ = select.select([src], [], [], 5)
            if not r:
                continue
            data = src.recv(65536)
            if not data:
                break
            dst.sendall(data)
    except Exception:                              # noqa: BLE001
        pass
    finally:
        try:
            dst.close()
        except Exception:                          # noqa: BLE001
            pass


def main() -> None:
    cfg = load_cfg()
    remote = REMOTE_ARG or cfg["comfy"]
    ssh = connect(cfg)
    tr = ssh.get_transport()
    assert tr is not None

    srv = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
    srv.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
    srv.bind(("127.0.0.1", LOCAL))
    srv.listen(16)
    print("[转发中] http://127.0.0.1:%d  →  云卡 127.0.0.1:%d（Ctrl+C 退出）"
          % (LOCAL, remote), flush=True)

    try:
        while True:
            cli, addr = srv.accept()
            try:
                ch = tr.open_channel("direct-tcpip", ("127.0.0.1", remote), addr)
            except Exception as e:                 # noqa: BLE001
                print("[跳过] 打不开通道: %s" % e, flush=True)
                cli.close()
                continue
            threading.Thread(target=pump, args=(cli, ch), daemon=True).start()
            threading.Thread(target=pump, args=(ch, cli), daemon=True).start()
    except KeyboardInterrupt:
        print("\n[停止]")
    finally:
        srv.close()
        ssh.close()


if __name__ == "__main__":
    main()
