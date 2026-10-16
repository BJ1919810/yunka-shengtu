#!/bin/bash
# 测速：ModelScope 上 Qwen-Image-2.1 的 bf16 文件，单流 vs 8 线程分片
#
# 目的：下 30GB 之前先确认链路能不能到「当年 24 MB/s」，避免盲等几小时。
# 只下几十 MB 丢掉，不落盘。用法： bash bench_bf16.sh [每段MB]
set -u
BASE="https://www.modelscope.cn/api/v1/models/Comfy-Org/Qwen-Image-2.1/repo?Revision=master&FilePath="
REL="${REL:-diffusion_models/qwen_image_2.1_bf16.safetensors}"
URL="${BASE}${REL}"

SEGMB="${1:-32}"
THREADS="${2:-8}"
SEG=$(( SEGMB * 1024 * 1024 ))

echo "===== bf16 测速（文件: $REL）====="

# --- 0) 探可达 + 取精确大小 ---
H=$(curl -sI -m 20 -L "$URL" 2>/dev/null)
CODE=$(printf '%s' "$H" | grep -i '^HTTP/' | tail -1 | awk '{print $2}')
LEN=$(printf '%s' "$H" | grep -i '^content-length' | tail -1 | tr -d '\r' | awk '{print $2}')
echo "  可达性: http=${CODE:-000}  content-length=${LEN:-?}"
[ "${CODE:-000}" != "200" ] && { echo "  不可达，终止"; exit 1; }

# --- 1) 单流：老老实实从 0 开始下 SEGMB ---
echo
echo "--- 单流：下 ${SEGMB}MB ---"
T0=$(date +%s.%N)
GOT=$(curl -sS -L -m 180 -r "0-$((SEG-1))" -o /dev/null -w '%{size_download}' "$URL" 2>/dev/null)
T1=$(date +%s.%N)
awk -v b="${GOT:-0}" -v d="$(awk -v a="$T0" -v b="$T1" 'BEGIN{print b-a}')" \
  'BEGIN{ if(d>0&&b>0) printf "  结果: %.0f MB / %.1f s = %.2f MB/s\n", b/1048576, d, (b/1048576)/d; else printf "  失败（无数据）\n" }'
SPD1=$(awk -v b="${GOT:-0}" -v d="$(awk -v a="$T0" -v b="$T1" 'BEGIN{print b-a}')" \
  'BEGIN{ if(d>0&&b>0) printf "%.2f", (b/1048576)/d; else printf "0" }')

# --- 2) 8 线程分片：每线程下 SEGMB，取文件的不同区段 ---
echo
echo "--- $THREADS 线程分片：每线程 ${SEGMB}MB（合计 $((SEGMB*THREADS))MB）---"
TMPD=$(mktemp -d)
T0=$(date +%s.%N)
pids=""
for i in $(seq 0 $((THREADS-1))); do
  # 从文件中段开始切，避开开头（减少 CDN 缓存影响）
  BASE_OFF=$(( 2147483648 + i * SEG ))
  S=$BASE_OFF
  E=$(( S + SEG - 1 ))
  ( curl -sS -L -m 180 -r "${S}-${E}" -o "$TMPD/p$i" "$URL" 2>/dev/null ) &
  pids="$pids $!"
done
for p in $pids; do wait "$p"; done
T1=$(date +%s.%N)
TOT=0
for i in $(seq 0 $((THREADS-1))); do
  SZ=$(stat -c%s "$TMPD/p$i" 2>/dev/null || echo 0)
  TOT=$(( TOT + SZ ))
done
rm -rf "$TMPD"
awk -v b="$TOT" -v d="$(awk -v a="$T0" -v b="$T1" 'BEGIN{print b-a}')" \
  'BEGIN{ if(d>0&&b>0) printf "  结果: %.0f MB / %.1f s = %.2f MB/s（8 线程合计）\n", b/1048576, d, (b/1048576)/d; else printf "  失败（无数据）\n" }'
SPD8=$(awk -v b="$TOT" -v d="$(awk -v a="$T0" -v b="$T1" 'BEGIN{print b-a}')" \
  'BEGIN{ if(d>0&&b>0) printf "%.2f", (b/1048576)/d; else printf "0" }')

# --- 3) 推算 ---
echo
echo "===== 推算（30.21 GB 全量）====="
awk -v s="$SPD1" 'BEGIN{ if(s>0) printf "  按单流 %.2f MB/s  →  %.0f 分钟\n", s, (30.21*1024)/s/60 }'
awk -v s="$SPD8" -v n="$THREADS" 'BEGIN{ if(s>0) printf "  按 %d线程 %.2f MB/s →  %.0f 分钟\n", n, s, (30.21*1024)/s/60 }'
echo "  （当年参考值：24 MB/s ⇒ 约 21 分钟）"
