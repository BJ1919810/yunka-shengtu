# -*- coding: utf-8 -*-
"""按人工确认的坐标切掉翻拍边框 / 桌面 / 手写字。

坐标来自`_peek/grid_*.png` 网格图（每 10% 一格），逐张目视确定。
⚠️ 自动亮度/纹理检测对「锯齿花边」无效（相邻像素一亮一暗），必须人工定框。
"""
import os
import sys
from PIL import Image

sys.stdout.reconfigure(encoding="utf-8", errors="replace")
HERE = os.path.dirname(os.path.abspath(__file__))
SRC = os.path.join(HERE, "input", "old")
OUT = os.path.join(HERE, "input", "crop")
os.makedirs(OUT, exist_ok=True)

# name -> (left%, top%, right%, bottom%)，按网格图目视
CROP = {
    # 顶部 0~9% 是桌面背景；底部 86~100% 是桌面 + 手写字「张xx 拍摄」
    # ⚠️ 第一轮只切到桌面/字，**上下锯齿花边还在**（一格亮一格暗，自动法无效）
    #    ⇒ 再往里收，把花边整条切掉（花边带高度约 4~5%）
    "old1_handheld.jpg":      (0.020, 0.135, 0.985, 0.815),

    # 拍立得：照片几乎占满画幅，四边白边很窄；左上角略有暗角
    # 右上角有明显反光亮块（靠右上角 ~x>55%, y<25%）→ 交给提示词修，不在此切
    "old2_handheld_glare.jpg": (0.045, 0.030, 0.965, 0.955),

    # 白边在下方约 8%，左右各约 3%；右下角还有黄色时间戳 → 提示词修
    "old3_handheld_stamp.jpg": (0.045, 0.020, 0.980, 0.870),
}

for n, (l, t, r, b) in CROP.items():
    sp = os.path.join(SRC, n)
    im = Image.open(sp).convert("RGB")
    W, H = im.size
    box = (int(W * l), int(H * t), int(W * r), int(H * b))
    c = im.crop(box)
    dp = os.path.join(OUT, n)
    c.save(dp, quality=97)
    print("%-26s %4dx%-4d → %4dx%-4d  (%.2f MP)  box=%s"
          % (n, W, H, c.size[0], c.size[1],
             c.size[0] * c.size[1] / 1048576, box))
    pv = c.copy()
    pv.thumbnail((420, 420))
    pv.save(os.path.join(HERE, "_peek", "crop2_" + n), quality=90)