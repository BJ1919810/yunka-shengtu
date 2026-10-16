# -*- coding: utf-8 -*-
"""云卡 ComfyUI 服务的「起 / 停 / 查」—— 一条命令搞定。

为什么要它：云卡（AutoDL）开机后 ComfyUI **不会自己跑**，每次都得手动 SSH 上去拉起。
这个脚本把 `remote/start_comfy.sh` 那套幂等启动逻辑包成子命令，配合同目录的
`cloud.cmd`（双击即用）就是「开卡 → 双击 → 出图」。

用法：
    python cloud_service.py            # 不给子命令 = menu（交互菜单）
    python cloud_service.py start      # 启动（幂等：已在跑就直接返回；等就绪）
    python cloud_service.py stop       # 停止
    python cloud_service.py restart    # 重启
    python cloud_service.py status     # 看状态（服务 / GPU / 队列 / 日志尾）
    python cloud_service.py queue      # 只看队列（含任务明细）
    python cloud_service.py logs [n]   # 看云端 comfy.log 尾部（默认 40 行）
    python cloud_service.py menu       # 交互菜单（边看状态边操作）
    python cloud_service.py where      # 看用的是哪份配置（不连 SSH，排查用）

配套的 cloud.cmd 是它的双击壳：
    cloud.cmd            双击 = 进交互菜单
    cloud.cmd status     等于 python cloud_service.py status
    cloud.cmd start      直接启动（跳过菜单）

⚠️ 连不上就立刻退出 —— 不做长时间重试。云卡没开机时去 AutoDL 控制台点开机，
   等 1~2 分钟再运行本脚本。
"""
from __future__ import annotations

import json
import os
import re
import sys

import cloud_sh

HERE = os.path.dirname(os.path.abspath(__file__))
START_SH = os.path.join(HERE, "remote", "start_comfy.sh")
LOG = "/root/autodl-tmp/comfy.log"

# 远端日志可能有非 GBK 字符，直接打印会 UnicodeEncodeError 把命令打断。
# ⚠️ 编码也一并**钉死成 UTF-8** —— 否则本机 Python 若开了 UTF-8 模式（输出 UTF-8），
#    而 cmd 控制台是 GBK，中文会变成「浜戝崱」这种乱码。
#    配套的 cloud.cmd 里有 `chcp 65001`，两边必须对齐。
for _s in (sys.stdout, sys.stderr):
    try:
        _s.reconfigure(encoding="utf-8", errors="replace")     # type: ignore[attr-defined]
    except Exception:
        pass


# ---------------------------------------------------------------- 连接
def connect():
    """连不上就打印人话提示并退出（用户明确要「立刻报错」，不做长等待）。"""
    cfg = cloud_sh.load_cfg()
    if not cfg["host"] or not cfg["pass"]:
        print("[x] 配置缺失：需要 ssh_host / ssh_password")
        print("    （或设 COMFY_SSH_HOST / COMFY_SSH_PASS 环境变量）")
        print("    诊断：python cloud_service.py where")
        sys.exit(2)
    try:
        return cfg, cloud_sh.connect(cfg, retries=2)
    except SystemExit as e:
        print("[x] 连不上云卡：%s" % (e.code if isinstance(e.code, str) else e))
        print()
        print("    可能的原因（按概率排）：")
        print("      1. 云卡还没开机 —— 去 AutoDL 控制台点「开机」，等 1~2 分钟再试")
        print("      2. SSH 并发占满（AutoDL 的 MaxStartups）—— 等几秒重试即可")
        print("      3. SSH 地址/端口变了 —— 开机后 AutoDL 可能分配新端口，")
        print("         跑 python cloud_service.py where 看看当前配置")
        sys.exit(2)


def run_remote(cli, script: str, timeout: float = 600) -> tuple[str, str, int]:
    return cloud_sh.run(cli, script, timeout=timeout)


# ---------------------------------------------------------------- 子命令
def cmd_start(cli, cfg) -> int:
    port = cfg["comfy_port"]
    print("[.] 正在启动云卡 ComfyUI（端口 %d）…" % port)
    try:
        with open(START_SH, encoding="utf-8") as f:
            script = f.read()
    except OSError as e:
        print("[x] 读不到 %s：%s" % (START_SH, e))
        return 1

    out, err, rc = run_remote(cli, script, timeout=300)
    out = (out or "").strip()
    if err.strip():
        print(err.strip())

    if "already_running" in out:
        print("[✓] 服务本来就在跑，没动它。")
        return 0
    m = re.search(r"started_in_(\d+)s", out)
    if m:
        print("[✓] 启动成功，耗时 %s 秒。" % m.group(1))
        print("    接下来用 cloudgen.py / App / 网页版都可以直接出图了。")
        return 0
    if "START_FAILED" in out:
        print("[x] 启动失败。远端日志尾部：")
        for line in out.splitlines()[-25:]:
            print("      " + line)
        return 1

    print("[?] 远端返回（没能识别状态，原样贴出）：")
    for line in out.splitlines():
        print("      " + line)
    return 0 if rc == 0 else 1


def cmd_stop(cli, cfg) -> int:
    """停服务。

    ⚠️ 不要用 pkill -f 'main.py --port 6006' —— 它会连**执行这条命令的 bash 自己**
       一起匹配上（命令行里就含这个字符串），把自己杀掉导致 SSH 直接断开。
       所以先扫 /proc 取 PID，再逐个 kill。
    """
    port = cfg["comfy_port"]
    script = """
PORT=%d
killed=0
for d in /proc/[0-9]*; do
  p=${d#/proc/}
  [ "$p" = "$$" ] && continue
  exe=$(readlink "$d/exe" 2>/dev/null || true)
  case "$exe" in
    *python*)
      cmd=$(tr '\\0' ' ' < "$d/cmdline" 2>/dev/null)
      case "$cmd" in
        *main.py*--port*$PORT*) echo "KILLED $p"; kill -9 "$p" 2>/dev/null; killed=1;;
      esac
      ;;
  esac
done
sleep 2
code=$(curl -s -o /dev/null -w '%%{http_code}' -m 3 http://127.0.0.1:$PORT/system_stats || true)
echo "AFTER_HTTP=$code"
[ "$killed" = "0" ] && echo "NOTHING_RUNNING"
""" % port

    print("[.] 正在停止云卡 ComfyUI…")
    out, err, rc = run_remote(cli, script, timeout=120)
    if err.strip():
        print(err.strip())
    killed = re.findall(r"KILLED (\d+)", out)
    after = re.search(r"AFTER_HTTP=(\S+)", out)
    if killed:
        print("[✓] 已停止（pid %s）" % ", ".join(killed))
    elif "NOTHING_RUNNING" in out:
        print("[i] 本来就没在跑。")
    if after and after.group(1) not in ("000", ""):
        print("[!] 端口 %d 仍有响应（HTTP %s）—— 可能还有别的实例" % (cfg["comfy_port"], after.group(1)))
    return 0


def cmd_restart(cli, cfg) -> int:
    rc = cmd_stop(cli, cfg)
    if rc:
        return rc
    print()
    return cmd_start(cli, cfg)


def _fetch_status(cli, cfg) -> dict:
    """一趟 SSH 拿齐所有状态，用 ### 分隔，避免多次往返。"""
    port = cfg["comfy_port"]
    script = """
PORT=%d
echo "### HTTP"
code=$(curl -s -o /dev/null -w '%%{http_code}' -m 4 http://127.0.0.1:$PORT/system_stats 2>/dev/null || true)
echo "${code:-000}"
echo "### PID"
for d in /proc/[0-9]*; do
  p=${d#/proc/}
  [ "$p" = "$$" ] && continue
  exe=$(readlink "$d/exe" 2>/dev/null || true)
  case "$exe" in
    *python*)
      cmd=$(tr '\\0' ' ' < "$d/cmdline" 2>/dev/null)
      case "$cmd" in
        *main.py*--port*$PORT*) echo "$p";;
      esac
      ;;
  esac
done
echo "### GPU"
nvidia-smi --query-gpu=name,memory.used,memory.total,utilization.gpu --format=csv,noheader 2>/dev/null || echo "n/a"
echo "### QUEUE"
curl -s -m 6 http://127.0.0.1:$PORT/queue 2>/dev/null || echo "{}"
echo
echo "### LOGTAIL"
tail -n 3 %s 2>/dev/null || echo "(没有日志)"
""" % (port, LOG)

    out, _err, _rc = run_remote(cli, script, timeout=90)
    parts: dict = {}
    cur = None
    for line in out.splitlines():
        if line.startswith("### "):
            cur = line[4:].strip()
            parts[cur] = []
        elif cur is not None:
            parts[cur].append(line)
    return {k: "\n".join(v).strip() for k, v in parts.items()}


def _status_lines(s: dict) -> list:
    """把 _fetch_status 的原始结果变成几行能直接打印的文本。"""
    http = (s.get("HTTP") or "000").strip()
    pids = [p for p in (s.get("PID") or "").splitlines() if p.strip()]
    gpu = (s.get("GPU") or "n/a").strip()
    running = http == "200"

    out = []
    if running:
        out.append("  服务    : 运行中（HTTP 200）" + ("   pid %s" % ", ".join(pids) if pids else ""))
    elif pids:
        out.append("  服务    : ✗ 进程在但不响应（HTTP %s）—— 僵住了，建议重启" % http)
    else:
        out.append("  服务    : ✗ 没在跑")

    out.append("  GPU     : %s" % gpu if gpu and gpu != "n/a"
               else "  GPU     : (没读到 nvidia-smi)")

    if running:
        head, detail = _fmt_queue(s.get("QUEUE") or "{}")
        out.append("  队列    : %s" % head)
        for d in detail:
            out.append("            %s" % d.strip())
    else:
        out.append("  队列    : （服务没起来，取不到）")

    tail = [l for l in (s.get("LOGTAIL") or "").splitlines() if l.strip()]
    if tail:
        out.append("  日志尾  : %s" % tail[-1][:100])
    return out


def _fmt_queue(raw: str) -> tuple[str, list]:
    """把 /queue 的 JSON 压成「运行 N / 排队 N」+ 任务明细。"""
    try:
        q = json.loads(raw or "{}")
    except Exception:
        return "(没拿到队列数据)", []
    run = q.get("queue_running") or []
    pend = q.get("queue_pending") or []
    lines = ["运行 %d 个 / 排队 %d 个" % (len(run), len(pend))]
    for label, items in (("跑", run), ("等", pend)):
        for it in items:
            pid = str(it[1])[:8] if len(it) > 1 else "?"
            nodes = it[2] if len(it) > 2 else {}
            pref, steps = "?", "?"
            for nd in (nodes or {}).values():
                if not isinstance(nd, dict):
                    continue
                ins = nd.get("inputs") or {}
                if "filename_prefix" in ins:
                    pref = ins["filename_prefix"]
                if "steps" in ins:
                    steps = ins["steps"]
            lines.append("    [%s] %s  %s  steps=%s" % (label, pid, pref, steps))
    return lines[0], lines[1:]


def cmd_status(cli, cfg) -> int:
    s = _fetch_status(cli, cfg)
    print("=" * 52)
    print("  云卡 ComfyUI · 状态")
    print("=" * 52)
    for line in _status_lines(s):
        print(line)
    print()
    return 0 if (s.get("HTTP") or "").strip() == "200" else 1


def cmd_menu(cli, cfg) -> int:
    """交互菜单 —— 双击 cloud.cmd 后想「边看边操作」时用。"""
    while True:
        s = _fetch_status(cli, cfg)
        print()
        print("=" * 52)
        print("  云卡 ComfyUI")
        print("=" * 52)
        for line in _status_lines(s):
            print(line)
        print("-" * 52)
        print("  1) 启动   2) 停止   3) 重启   4) 刷新状态")
        print("  5) 看日志  6) 看队列  0) 退出")
        try:
            ch = input("  选择 > ").strip()
        except (EOFError, KeyboardInterrupt):
            print()
            return 0

        if ch == "0" or ch.lower() in ("q", "quit", "exit"):
            return 0
        if ch in ("1", "start"):
            cmd_start(cli, cfg)
            input("\n  回车继续…")
        elif ch in ("2", "stop"):
            cmd_stop(cli, cfg)
            input("\n  回车继续…")
        elif ch in ("3", "restart"):
            cmd_restart(cli, cfg)
            input("\n  回车继续…")
        elif ch in ("4", "r", "status"):
            continue                                  # 回到循环顶部重新拉状态
        elif ch in ("5", "log", "logs"):
            try:
                n = input("  要最后几行？（回车=40）> ").strip()
            except (EOFError, KeyboardInterrupt):
                n = ""
            cmd_logs(cli, cfg, int(n) if n.isdigit() else 40)
            input("\n  回车继续…")
        elif ch in ("6", "queue"):
            cmd_queue(cli, cfg)
            input("\n  回车继续…")
        else:
            print("  [?] 不认识的选项：%s" % ch)


def cmd_queue(cli, cfg) -> int:
    port = cfg["comfy_port"]
    out, _e, _c = run_remote(
        cli,
        "curl -s -m 6 http://127.0.0.1:%d/queue 2>/dev/null || echo '{}'" % port,
        timeout=60)
    print("=" * 52)
    print("  云卡 ComfyUI · 队列")
    print("=" * 52)
    head, detail = _fmt_queue(out.strip() or "{}")
    print("  " + head)
    for d in detail:
        print("  " + d.strip())
    if not detail:
        print("  （空）")
    print()
    print("  提示：手机端/网页端提交的任务也会排在这里 —— 两边共用一个队列。")
    return 0


def cmd_logs(cli, cfg, n: int = 40) -> int:
    out, _e, _c = run_remote(
        cli, "tail -n %d %s 2>/dev/null || echo '(没有日志)'" % (n, LOG), timeout=60)
    print("=" * 52)
    print("  云卡 %s  (最后 %d 行)" % (LOG, n))
    print("=" * 52)
    print(out.rstrip())
    print()
    return 0


def cmd_where(_cli, cfg) -> int:
    print("  配置来源：COMFY_SSH_* 环境变量 > %s/cloud-comfy.config.json" % HERE)
    for k, label in (("host", "ssh_host"), ("port", "ssh_port"),
                     ("user", "ssh_user"), ("comfy_port", "远端 ComfyUI 端口")):
        print("    %-22s %s" % (label, cfg[k]))
    print("    %-22s %s" % ("ssh_pass", "已设置（%d 字符）" % len(cfg["pass"])
                            if cfg["pass"] else "**缺失**"))
    return 0


COMMANDS = {
    "start": cmd_start,
    "stop": cmd_stop,
    "restart": cmd_restart,
    "status": cmd_status,
    "queue": cmd_queue,
    "menu": cmd_menu,
    "where": cmd_where,
}


USAGE = __doc__


def main() -> None:
    args = sys.argv[1:]
    # 不给子命令 = 进交互菜单（双击 cloud.cmd 的场景：先看状态，再决定做什么）
    cmd = (args[0] if args else "menu").lower()
    if cmd in ("help", "-h", "--help", "/?"):
        print(USAGE)
        return

    # where 只看本地配置，**不连 SSH** —— 恰恰是连不上时最需要它
    if cmd == "where":
        sys.exit(cmd_where(None, cloud_sh.load_cfg()))

    if cmd == "logs":
        n = 40
        if len(args) > 1 and args[1].isdigit():
            n = max(1, min(2000, int(args[1])))
        cfg, cli = connect()
        try:
            sys.exit(cmd_logs(cli, cfg, n))
        finally:
            cli.close()

    if cmd not in COMMANDS:
        print("[x] 不认识的子命令：%s\n" % cmd)
        print(USAGE)
        sys.exit(2)

    cfg, cli = connect()
    try:
        sys.exit(COMMANDS[cmd](cli, cfg))
    finally:
        cli.close()


if __name__ == "__main__":
    main()
