#!/bin/bash
# 第二轮测速：修好 ModelScope URL 后对比各源，重点看 ModelScope 能否压过 hf-mirror。
set -u
REL="diffusion_models/qwen_image_2.1_int8_convrot.safetensors"
TESTMB="${1:-32}"
CHUNK=$((TESTMB * 1024 * 1024))

probe() {
  local name="$1" url="$2"
  local t0 t1 spd got
  t0=$(date +%s.%N)
  got=$(curl -s -m 120 -L -r "0-$((CHUNK - 1))" -o /dev/null -w '%{size_download}' "$url" 2>/dev/null)
  t1=$(date +%s.%N)
  if [ "${got:-0}" -lt 1048576 ]; then
    echo "[$name] 失败/过慢（仅 ${got:-0} 字节）"
    return
  fi
  spd=$(awk -v b="$got" -v d="$(awk -v a="$t0" -v b="$t1" 'BEGIN{print b-a}')" \
        'BEGIN{ if(d>0) printf "%.1f", (b/1048576)/d; else printf "0" }')
  echo "[$name] 实下$(( got/1048576 ))MB 用时$(awk -v a="$t0" -v b="$t1" 'BEGIN{printf "%.1f", b-a}')s → ${spd} MB/s"
  awk -v s="$spd" 'BEGIN{printf "         └ 14.3GB 全量预计 %.1f 分钟\n", (14.3*1024)/s/60}'
}

echo "===== 第二轮测速：每源 ${TESTMB}MB ====="
echo "--- ModelScope api/v1 Comfy-Org 镜像 ---"
probe "modelscope" "https://www.modelscope.cn/api/v1/models/Comfy-Org/Qwen-Image-2.1/repo?Revision=master&FilePath=$REL"

echo "--- ModelScope api/v1 短域名 ---"
probe "ms短域" "https://modelscope.cn/api/v1/models/Comfy-Org/Qwen-Image-2.1/repo?Revision=master&FilePath=$REL"

echo "--- hf-mirror（对照，上轮0.3MB/s）---"
probe "hf-mirror" "https://hf-mirror.com/Comfy-Org/Qwen-Image-2.1/resolve/main/$REL"

echo "===== 结束 ====="