#!/bin/bash
# 把 disable_dead_links.sh 搬走的断链软链**搬回原位**（按清单逐个恢复）
#
# 什么情况下要跑它：
#   · 平台哪天真的挂上了 `/.autodl` 公共模型库 ⇒ 那些链就"活"了，想用就直接恢复
#   · 想看全量模型列表 / 排查别的问题
#
# 用法：bash restore_dead_links.sh [models目录] [存放目录]
set -u

SRC="${1:-/root/autodl-tmp/ComfyUI/models}"
DST="${2:-/root/autodl-tmp/_disabled_dead_links}"
MAN="$(dirname "$DST")/_disabled_dead_links.txt"

[ -f "$MAN" ] || { echo "找不到清单：$MAN（没跑过 disable_dead_links.sh？）"; exit 1; }

n=0
miss=0
while IFS= read -r rel; do
    [ -n "$rel" ] || continue
    s="$DST/$rel"
    t="$SRC/$rel"
    if [ -L "$s" ] || [ -e "$s" ]; then
        mkdir -p "$(dirname "$t")"
        mv "$s" "$t" && n=$((n + 1))
    else
        miss=$((miss + 1))
    fi
done < "$MAN"

echo "已恢复 $n 个软链回 $SRC"
[ "$miss" -gt 0 ] && echo "（清单里有 $miss 条在存放处已找不到，跳过）"
echo
echo "⚠️ 记得重启 ComfyUI 才会重扫模型列表"
