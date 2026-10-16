import os
from PIL import Image, ImageDraw

import sys as _sys
import pathlib as _pl
_here = _pl.Path(__file__).resolve()
for _r in (_here.parent, *_here.parents):          # 向上找仓库根（含 paths.py）
    if (_r / "paths.py").exists():
        _sys.path.insert(0, str(_r))
        break
import paths  # noqa: E402  （路径唯一权威）

SRC = str(paths.DOWNLOADS)
OUT = str(paths.REPO / "_peek")
os.makedirs(OUT, exist_ok=True)

names = ["手持1.jpg", "手持2（带有反光）.jpg", "手持3.jpg",
         "电子版1.jpg", "电子版2.jpg"]

for f in names:
    p = os.path.join(SRC, f)
    if not os.path.exists(p):
        print("missing", f); continue
    im = Image.open(p).convert("RGB")
    W, H = im.size
    # 在原图上画相对坐标网格（每 10% 一条线并标注）
    im2 = im.copy()
    d = ImageDraw.Draw(im2)
    for i in range(1, 10):
        x = int(W * i / 10); y = int(H * i / 10)
        d.line([(x, 0), (x, H)], fill=(255, 0, 0), width=1)
        d.line([(0, y), (W, y)], fill=(255, 0, 0), width=1)
        d.text((x + 2, 2), "%d%%" % (i * 10), fill=(255, 255, 0))
        d.text((2, y + 2), "%d%%" % (i * 10), fill=(0, 255, 255))
    im2.save(os.path.join(OUT, "grid_" + f), quality=92)
    print("grid", f, W, H)