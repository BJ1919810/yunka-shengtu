#!/bin/bash
# 下载源测速：每个源抓 32MB 看实际吞吐，不落盘。
# 只测 DiT 那个最大最通用的文件（7.26GB），它最能代表链路速度。
set -u
REL="diffusion_models/qwen_image_2.1_int8_convrot.safetensors"
TESTMB="${1:-32}"
CHUNK=$((TESTMB * 1024 * 1024))

probe() {
  local name="$1" url="$2"
  # 先探可达性 + 真实大小 + 是否支持 Range
  local hdr
  hdr=$(curl -sI -m 20 -L "$url" 2>/dev/null)
  local code len ar
  code=$(printf '%s' "$hdr" | grep -i '^HTTP/' | tail -1 | awk '{print $2}')
  ar=$(printf '%s' "$hdr" | grep -i '^accept-ranges' | tail -1 | tr -d '\r' | awk '{print $2}')
  len=$(printf '%s' "$hdr" | grep -i '^content-length' | tail -1 | tr -d '\r' | awk '{print $2}')
  if [ "${code:-000}" != "200" ]; then
    echo "[$name] 不可达 http=$code"
    return
  fi
  local sz="?"
  [ -n "${len:-}" ] && sz=$(awk -v b="$len" 'BEGIN{printf "%.2f GB", b/1073741824}')
  # 实测：下 32MB 丢/dev/null
  local t0 t1 spd got
  t0=$(date +%s.%N)
  got=$(curl -s -m 90 -L -r "0-$((CHUNK - 1))" -o /dev/null -w '%{size_download}' "$url" 2>/dev/null)
  t1=$(date +%s.%N)
  spd=$(awk -v b="${got:-0}" -v d="$(awk -v a="$t0" -v b="$t1" 'BEGIN{print b-a}')" \
        'BEGIN{ if(d>0 && b>0) printf "%.1f", (b/1048576)/d; else printf "0" }')
  echo "[$name] http=200 accept-ranges=${ar:-none} size=$sz 实下$(( got/1048576 ))MB 用时$(awk -v a="$t0" -v b="$t1" 'BEGIN{printf "%.1f", b-a}')s → ${spd} MB/s"
  if [ "${spd:-0}" != "0" ]; then
    awk -v s="$spd" 'BEGIN{printf "         └ 14.3GB 全量预计 %.1f 分钟\n", (14.3*1024)/s/60}'
  fi
}

echo "===== 测速：每源下 ${TESTMB}MB，丢弃 ====="
echo "--- 源1: hf-mirror (HF 国内镜像) ---"
probe "hf-mirror" "https://hf-mirror.com/Comfy-Org/Qwen-Image-2.1/resolve/main/$REL"

echo "--- 源2: modelscope 网页域名 ---"
probe "modelscope" "https://www.modelscope.cn/models/Qwen/Qwen-Image-2.1/resolve/master/$REL"

echo "--- 源3: modelscope 短域名 ---"
probe "ms短" "https://modelscope.cn/models/Qwen/Qwen-Image-2.1/resolve/master/$REL"

echo "--- 源4: modelscope API endpoint ---"
probe "ms-api" "https://www.modelscope.cn/api/v1/models/Qwen/Qwen-Image-2.1/repo?Revision=master&FilePath=$REL"

echo "--- 源5: huggingface 直连 ---"
probe "hf官方" "https://huggingface.co/Comfy-Org/Qwen-Image-2.1/resolve/main/$REL"

echo "===== 结束 ====="