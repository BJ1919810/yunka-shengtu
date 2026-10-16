# -*- coding: utf-8 -*-
"""云卡 SSH 小工具 —— 避免每次重写 paramiko 样板。

用法：
    python cloud_sh.py "<远端 shell 命令>"
    python cloud_sh.py --file remote/start_comfy.sh
    python cloud_sh.py --put <本地文件> <远端路径>
    python cloud_sh.py --get <远端文件> <本地路径>

口令来源优先级：环境变量 COMFY_SSH_PASS → 同目录 cloud-comfy.config.json
"""
from __future__ import annotations

import json
import os
import sys
import time

import paramiko

HERE = os.path.dirname(os.path.abspath(__file__))
CFG_PATH = os.path.join(HERE, "cloud-comfy.config.json")


def load_cfg() -> dict:
    cfg: dict = {}
    if os.path.exists(CFG_PATH):
        with open(CFG_PATH, encoding="utf-8") as f:
            cfg = json.load(f)
    return {
        "host": os.environ.get("COMFY_SSH_HOST") or cfg.get("ssh_host", ""),
        "port": int(os.environ.get("COMFY_SSH_PORT") or cfg.get("ssh_port", 22)),
        "user": os.environ.get("COMFY_SSH_USER") or cfg.get("ssh_user", "root"),
        "pass": os.environ.get("COMFY_SSH_PASS") or cfg.get("ssh_password", ""),
        "comfy_port": int(os.environ.get("COMFY_REMOTE_PORT") or cfg.get("remote_comfy_port", 6006)),
    }


def connect(cfg: dict | None = None, retries: int = 6) -> paramiko.SSHClient:
    cfg = cfg or load_cfg()
    if not cfg["host"] or not cfg["pass"]:
        sys.exit("[配置缺失] 需要 ssh_host / ssh_password（或设 COMFY_SSH_* 环境变量）")

    last: Exception | None = None
    for attempt in range(retries):
        cli = paramiko.SSHClient()
        cli.set_missing_host_key_policy(paramiko.AutoAddPolicy())
        try:
            cli.connect(cfg["host"], cfg["port"], cfg["user"], cfg["pass"],
                        timeout=25, look_for_keys=False, allow_agent=False)
            return cli
        except Exception as e:
            last = e
            try:
                cli.close()
            except Exception:
                pass
            # ⚠️ AutoDL 的 sshd 会回一句纯文本 "Exceeded MaxStartups\r\n"（未认证并发连接
            #    达上限，通常是别的隧道/扫描占着），它不是 SSH banner ⇒ paramiko 抛
            #    "Error reading SSH protocol banner"。这不是云卡挂了，退避重连即可。
            #    2026-10-02 实测 6 次里有 2 次撞上。
            if attempt < retries - 1:
                time.sleep(0.4 * (attempt + 1))
    sys.exit(f"[SSH 连接失败，重试 {retries} 次仍不通] {type(last).__name__}: {last}")


def run(cli: paramiko.SSHClient, script: str, timeout: float = 600) -> tuple[str, str, int]:
    """执行一段脚本，返回 (stdout, stderr, exit_code)。

    ⚠️ 远端没有 syslog、无 sshpass；用绝对路径调 python。
    """
    stdin, stdout, stderr = cli.exec_command(script, timeout=timeout)
    out = stdout.read().decode("utf-8", "replace")
    err = stderr.read().decode("utf-8", "replace")
    rc = stdout.channel.recv_exit_status()
    return out, err, rc


def check_msys_path(remote: str, arg: str = "") -> None:
    """拦住 Git Bash（MSYS2）把远端路径悄悄改写掉。

    ★★ 2026-10-07 实测定位的真凶（`--put` 一直报 ENOENT 就是它）：
       Git Bash 会把命令行里的 `/root/xxx` 当成 Unix 路径，自动重写成
       `D:/Program Files (x86)/.../PortableGit/root/xxx`
       ⇒ paramiko 拿这个 Windows 路径去开**远端**文件 ⇒ `FileNotFoundError`。
       **不是 SFTP 的 bug。**（我为此误判过一轮，记在这里别再犯。）

    跑 `cloud_sh.py --put` 时如果撞上，看这个提示即可，不用去翻 paramiko 源码。
    """
    bad = [p for p in ("PortableGit", "/Git/root", "/usr/root") if p in remote]
    looks_windows = len(remote) > 2 and remote[1] == ":"
    if bad or (looks_windows and remote.startswith(tuple("CDEFGHcdefgh"))):
        # 从被改写的路径里把原始远端路径抠出来，提示更友好
        guess = remote
        for p in ("PortableGit",):
            if p in guess:
                guess = "/" + guess.split(p, 1)[1].lstrip("/")
        print("[x] 远端路径被 Git Bash 改写了：")
        print("      实际收到：%s" % remote)
        if guess != remote:
            print("      你想传的：%s" % guess)
        print()
        print("    这是 Git Bash 的 MSYS 路径转换，不是程序的错。三种解法：")
        print("      ① 命令前加  MSYS_NO_PATHCONV=1")
        print("      ② 远端路径写成双斜杠：//root/autodl-tmp/...")
        print("      ③ 在 cmd / PowerShell 里跑（不经 Git Bash）")
        sys.exit(2)


def put_file(cli: paramiko.SSHClient, local: str, remote: str) -> int:
    """上传文件，返回字节数。

    ⚠️ 教训（2026-10-07）：`sftp.put` 本身**没问题**，两次实测都逐字节一致。
       当时看到的"稳定 ENOENT"是 **Git Bash 把远端路径改写掉了**（见
       `check_msys_path`）。排查时**先把实参 print 出来**再看别的。
    """
    check_msys_path(remote)
    if not os.path.isfile(local):
        sys.exit(f"[本地文件不存在] {local}")
    size = os.path.getsize(local)
    sftp = cli.open_sftp()
    try:
        sftp.put(local, remote)
        got = sftp.stat(remote).st_size          # ← 这个 stat 查的是远端，别拿本地路径来查
        if got != size:
            sys.exit(f"[上传不完整] 本地 {size} B，远端 {got} B")
        return got
    finally:
        sftp.close()


def main() -> None:
    args = sys.argv[1:]
    if not args:
        sys.exit(__doc__)
    if args[0] == "--put":
        # python cloud_sh.py --put <本地文件> <远端路径>
        local, remote = args[1], args[2]
        cli = connect()
        try:
            n = put_file(cli, local, remote)
            print(f"[uploaded] {remote}  {n} bytes")
        finally:
            cli.close()
        return
    if args[0] == "--get":
        # python cloud_sh.py --get <远端文件> <本地路径>
        remote, local = args[1], args[2]
        check_msys_path(remote)                  # 远端路径同样会被 Git Bash 改写
        cli = connect()
        try:
            sftp = cli.open_sftp()
            sftp.get(remote, local)
            print(f"[downloaded] {local}  {os.path.getsize(local)} bytes")
            sftp.close()
        finally:
            cli.close()
        return
    if args[0] == "--file":
        with open(args[1], encoding="utf-8") as f:
            script = f.read()
    else:
        script = args[0]
    cli = connect()
    try:
        out, err, rc = run(cli, script)
        sys.stdout.write(out)
        if err:
            sys.stderr.write("\n[stderr]\n" + err)
        print(f"\n[exit={rc}]")
    finally:
        cli.close()


if __name__ == "__main__":
    main()
