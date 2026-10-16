#!/bin/bash
# 把 ComfyUI models 里的【断链软链】移出扫描路径（移动，不是删除 —— 可一键恢复）
#
# 为什么：
#   社区镜像在 models/ 下塞了几百个模型名（Wan2.2 / LTX-2.3 / minimax_h3 / krea2 …），
#   绝大多数是指向 `/.autodl/...`（平台公共模型库、本实例未挂载）的**断链**。
#   ComfyUI 用 glob 扫目录、**不检查文件是否存在** ⇒ 这些名字全被列进下拉
#   ⇒ App / web 的「扩散模型 / 文本编码器 / VAE」下拉变成几百项，
#      而且**选中任何一个都会加载失败**。
#
# 做法：只搬**断链**；真文件与**有效**软链一律不动。
#   ⚠️ 我们自己的那两条同义软链（int8_convrot → bf16 / w4a8 → bf16）是**有效**的，
#      所以不会被搬走 —— 这正是想要的。
#
# 用法：bash disable_dead_links.sh [models目录] [存放目录]
# 恢复：bash restore_dead_links.sh
set -u

SRC="${1:-/root/autodl-tmp/ComfyUI/models}"
DST="${2:-/root/autodl-tmp/_disabled_dead_links}"
MAN="$(dirname "$DST")/_disabled_dead_links.txt"

[ -d "$SRC" ] || { echo "models 目录不存在：$SRC"; exit 1; }
mkdir -p "$DST"
: > "$MAN"

n=0
keep=0
while IFS= read -r -d '' f; do
    if [ -e "$f" ]; then
        keep=$((keep + 1))          # 有效软链 / 真文件：原地不动
        continue
    fi
    rel="${f#"$SRC"/}"
    mkdir -p "$DST/$(dirname "$rel")"
    if mv "$f" "$DST/$rel"; then
        printf '%s\n' "$rel" >> "$MAN"
        n=$((n + 1))
    fi
done < <(find "$SRC" -type l -print0)

echo "断链搬走     : $n 个"
echo "保留不动     : $keep 个（有效软链 / 真文件）"
echo "存放处       : $DST"
echo "清单         : $MAN"
echo
echo "⚠️ 记得重启 ComfyUI 才会重扫模型列表（模型列表有缓存）"
echo "   恢复命令   : bash /root/autodl-tmp/restore_dead_links.sh"
