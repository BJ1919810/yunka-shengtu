#!/bin/bash
# Qwen-Image-2.1 三件套下载（经 ModelScope，8线程分片）
#
# 为什么不用 hf-mirror：实测 hf-mirror 在 0.3~1.5 MB/s 之间剧烈波动（13.5小时），
# ModelScope 稳定 5.2 MB/s。差距 3.5~17 倍。
#
# 为什么不用 ModelScope 官方 SDK：SDK 会把大文件拆缓存再拼，多一层不确定性；
# 直接 curl + Range 分片更可控，且能精确校验体积。
set -u

MS="https://www.modelscope.cn/api/v1/models/Comfy-Org/Qwen-Image-2.1/repo?Revision=master&FilePath="
DEST="/root/autodl-tmp/ComfyUI/models"
LOG="/root/autodl-tmp/dl_qwen21.log"
THREADS=16

# 期望体积（curl -I 实测拿到的 content-length，别猜）
declare -A EXPECT=(
  ["diffusion_models/qwen_image_2.1_bf16.safetensors"]=14230280616
  ["text_encoders/qwen3vl_8b_bf16.safetensors"]=17534334616
  ["vae/qwen_image_2.1_vae_bf16.safetensors"]=675509688
)

log() { echo "[$(date +%H:%M:%S)] $*" | tee -a "$LOG"; }

dl_one() {
  local rel="$1" want="${EXPECT[$1]}"
  local out="$DEST/$rel"
  local tmp="$out.part"
  mkdir -p "$(dirname "$out")"

  if [ -f "$out" ]; then
    local have; have=$(stat -c%s "$out")
    if [ "$have" = "$want" ]; then
      log "跳过（已完整）：$rel"
      return 0
    fi
    log "已存在但体积不符（$have ≠ $want），重下：$rel"
    rm -f "$out"
  fi

  log "开始 $rel  期望 $(( want / 1073741824 )) GB  ${THREADS}线程"

  # 分片下载：每线程负责 [i*chunk, min((i+1)*chunk-1, want-1)]
  local chunk=$(( (want + THREADS - 1) / THREADS ))
  : > "$tmp"
  local pids=""
  for ((i = 0; i < THREADS; i++)); do
    local start=$(( i * chunk ))
    local end=$(( start + chunk - 1 ))
    [ "$end" -ge "$want" ] && end=$(( want - 1 ))
    [ "$start" -gt "$end" ] && continue
    local part="$out.p$i"
    (
      # 单线程最多重试 4 次。
      # ⚠️ 勿加 --retry-all-errors：云卡curl 7.68 不认，会让整条命令直接报错退出。
      # ⚠️ -L 必须带：ModelScope 返回 302 跳 modelscope CDN。
      curl -sS -L --retry 4 --retry-delay 3 \
           -r "${start}-${end}" -o "$part" "${MS}${rel}" \
        && echo "PART_OK $i $rel"
      if [ "$(stat -c%s "$part" 2>/dev/null || echo 0)" != "$(( end - start + 1 ))" ]; then
        echo "PART_BAD $i $rel got=$(stat -c%s "$part" 2>/dev/null || echo 0) want=$(( end - start + 1 ))"
      fi
    ) &
    pids="$pids $!"
  done
  for p in $pids; do wait "$p"; done

  # 拼接
  : > "$tmp"
  for ((i = 0; i < THREADS; i++)); do
    [ -f "$out.p$i" ] && cat "$out.p$i" >> "$tmp"
    rm -f "$out.p$i"
  done

  local got; got=$(stat -c%s "$tmp" 2>/dev/null || echo 0)
  if [ "$got" != "$want" ]; then
    log "✗ 失败 $rel  实得 $got / 期望 $want （保留 .part 可续传）"
    return 1
  fi
  mv "$tmp" "$out"
  log "✓ 完成 $rel  $(( got / 1073741824 )) GB"
  return 0
}

: > "$LOG"
log "===== 开始下载 Qwen-Image-2.1 三件套（ModelScope / ${THREADS}线程）====="
df -h /root/autodl-tmp | tail -1 | tee -a "$LOG"

# ⛔ 预检：先用单线程拉 1MB，确认 curl 参数与 Range 支持都没问题，再起并发。
#    （2026-10-02 踩过两次坑：
#      1) --retry-all-errors 不被支持，8 分片一秒全废；
#      2) ModelScope 是 302 跳 CDN，**必须 -L**，漏了只拿到 366 字节的重定向 HTML）
PRE_REL="vae/qwen_image_2.1_vae_bf16.safetensors"
PRE=$(curl -sS -L -r 0-1048575 -o /dev/null -w '%{size_download}' "${MS}${PRE_REL}" 2>/dev/null)
if [ "${PRE:-0}" -lt 1048576 ]; then
  log "✗ 预检失败（只拉到 ${PRE:-0} 字节），终止，未动任何文件"
  log "  排查：curl 版本 / Range 支持 / 网络 / 是否漏了 -L"
  curl --version | head -1 | tee -a "$LOG"
  exit 2
fi
log "✓ 预检通过（1MB Range 正常），开始正式下载"

fail=0
# 大件先下
for rel in "diffusion_models/qwen_image_2.1_bf16.safetensors" \
           "text_encoders/qwen3vl_8b_bf16.safetensors" \
           "vae/qwen_image_2.1_vae_bf16.safetensors"; do
  dl_one "$rel" || fail=1
done

echo "===== 结果 =====" | tee -a "$LOG"
ls -lh "$DEST/diffusion_models/"*.safetensors \
       "$DEST/text_encoders/"*.safetensors \
       "$DEST/vae/"*.safetensors 2>/dev/null | tee -a "$LOG"
df -h /root/autodl-tmp | tail -1 | tee -a "$LOG"
[ "$fail" = "0" ] && log "全部成功" || log "有失败项，见上"
exit $fail