#!/bin/bash
# 启动云卡 ComfyUI（幂等：已在跑就直接返回）
#
# ⚠️ 两个踩过的坑，改脚本时别改回去：
#  1) pkill -f 'main.py --port 6006' 会**连执行这条命令的 bash 自己也匹配上**，
#     pkill 杀掉自己所在的会话 → SSH 直接断开（表现为 exit=-1、无输出）。
#     所以改成先取 PID 再逐个 kill。
#  2) 端口探活要在起进程**之后**等一会儿；升级 ComfyUI 后旧进程会僵在
#     "进程在但端口 000" 的状态，必须按 PID 清掉再起。
set -u
PORT=6006

alive() { curl -s -o /dev/null -m 3 "http://127.0.0.1:$PORT/system_stats"; }

if alive; then
  echo "already_running"
  exit 0
fi

# ---- 清掉僵住的旧进程（只杀 python 本体，不碰 bash）----
for d in /proc/[0-9]*; do
  p=${d#/proc/}
  [ "$p" = "$$" ] && continue
  exe=$(readlink "$d/exe" 2>/dev/null || true)
  case "$exe" in
    *python*)
      cmd=$(tr '\0' ' ' < "$d/cmdline" 2>/dev/null)
      case "$cmd" in
        *main.py*--port*${PORT}*) echo "killing stale pid=$p"; kill -9 "$p" 2>/dev/null;;
      esac
      ;;
  esac
done
sleep 3

# python 解释器：新云卡（vGPU-32GB / 4080S，2026-10-09 换机）用镜像自带的 miniconda3 base；
# 旧实例是独立 venv（/root/autodl-tmp/envs/comfy/bin/python）。需要时用 COMFY_PY 覆盖。
PY="${COMFY_PY:-/root/miniconda3/bin/python}"

cd /root/autodl-tmp/ComfyUI || exit 1
# 清空日志，否则排查时看到的是上一次的输出，容易误判
: > /root/autodl-tmp/comfy.log
# ⚠️ `--enable-cors-header` **必须保留**：镜像自带的 /root/start.sh 一直带着它，
#    现在开机自启改走本脚本，去掉它就等于**改了行为**（App 的 WebView 直连 6006 时可能要 CORS）。
nohup setsid "$PY" main.py \
  --port $PORT --listen 0.0.0.0 --preview-method latent2rgb --enable-cors-header \
  >> /root/autodl-tmp/comfy.log 2>&1 < /dev/null &

for i in $(seq 1 60); do
  sleep 2
  code=$(curl -s -o /dev/null -w '%{http_code}' -m 3 "http://127.0.0.1:$PORT/system_stats" || true)
  if [ "$code" = "200" ]; then
    echo "started_in_$((i*2))s"
    curl -s "http://127.0.0.1:$PORT/system_stats" | head -c 400
    exit 0
  fi
done
echo "START_FAILED — 日志尾部："
tail -30 /root/autodl-tmp/comfy.log
exit 1