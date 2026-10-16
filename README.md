# 云卡生图

云端 GPU（AutoDL 3080）出图工具。**本目录是项目总入口**，包含两条链路：

| 链路 | 位置 | 说明 |
|---|---|---|
| **桌面 CLI** | 本目录 | SSH 隧道 + 命令行出图（`cloudgen.py`） |
| **Android App** | ⬇️ 见下方「Android App」 | 手机出图，**源码与构建产物不在本目录** |

两条链路连的是**同一台云卡、同一个 ComfyUI**，共用 `cloud-comfy.config.json` 里的连接信息。

> ### 📁 关于路径
>
> 本文里 `<ComfyUI 目录>`、`<Anima 目录>`、`<Qwen-Image-2.1 目录>`、`<Android-tts 目录>`
> 这类尖括号都是**占位符**，不是一个真存在的路径。
> 它们由仓库根目录的 **`paths.py` 统一解析**（环境变量 > 自动探测 > 明确报错），
> 代码里不再硬编码任何本机绝对路径。想知道本机实际解析成什么：
>
> ```bash
> python paths.py              # 打印全部路径 + 是否存在
> python paths.py --get COMFY_DIR
> ```
>
> 换机 / 目录布局不同时，设 `YUNKA_DEV_ROOT`（开发根）即可整体改基；
> 单个目录另可用 `COMFY_DIR` / `ANIMA_ROOT` / `QWEN_DIR` / `YUNKA_ANDROID_TTS` 覆盖。
> 三个 `.cmd` 入口还会额外读一份本机私有配置 `paths.local.cmd`
> （从 `paths.local.cmd.example` 复制而来，已被 `.gitignore` 忽略）。

---

## 一、Android App（手机端）

### 📍 源码与构建目录

```
<Android-tts 目录>\tools\comfy_android\
```

> ⚠️ **它物理上在 `Android-tts` 项目里**（那是历史原因 —— 和 TTS 一起起的）。
> 本目录只做**索引**，不复制代码。改 App 请去上面那个路径。
> 默认取**本仓库的同级目录** `Android-tts`，可用环境变量 `YUNKA_ANDROID_TTS` 覆盖
> （见 `paths.py`）。

已出包产物：

```
<Android-tts 目录>\outputs\云卡生图_v1.3.7.apk
```

（包名 `com.audio8.comfy` · **v1.3.7** · minSdk 26 / targetSdk 34 · 纯 Java WebView 壳，零第三方依赖）

> **v1.3.7 = 复制按钮 + 全屏左右滑翻页 + 两个真 bug**（2026-10-04）：用户气泡下加「复制」；
> 全屏看图左右滑翻页（相册式跟手）；**修掉「文生图带上图生图残留源图」（等效图生图）**、
> **「多图合成提交即报需要至少一张参考图」**。详见 App 的 README。
>
> **v1.3.3 = 看图放大 + 生成中可切模式**（2026-10-03）：全屏看图支持双击/捏合缩放
> 与拖动；生成过程中不再禁止切换文生图/图生图/多图合成。
>
> **v1.3.2 = @ 补全修复**（2026-10-03）：弹框不再被软键盘遮住（先显示再量高度）、
> `<imageN>` 插入后有一排缩略图回显。
>
> **v1.3.1 = 真机反馈修复**（2026-10-03）：参数面板可滚动、`.hidden` 全局规则补齐
> （修「点清除没反应」）、源图切多图时自动并入队列、**新增「取消」**、默认步数 25→10。
>
> **v1.3.0 = 参考图修复 + 画幅可控**（2026-10-03）。相对 v1.2.0：
> ① ⛔ **修掉参考图静默失效** —— `TextEncodeQwenImage21.images` 是 Autogrow，
>    API 格式必须是**扁平点号键** `"images.image_1"`；App 一直在发嵌套的
>    `images: {image_1: …}` ⇒ 引擎**静默丢弃**：图生图/多图退化成纯文生图、
>    出图恒定 1024×1024 且不报错。（同日 `cloudgen.py` 也从源码定位到同一问题。）
> ② ⛔ **修掉死参数** —— 「尺寸」里的宽高从来没接进工作流。
> ③ ★ **新增画幅控制** —— 带参考图时多一个「跟随第一张参考图」开关，
>    取消勾选即可用宽高指定画幅（`EmptyLatentImage`，与官方 edit 模板同款接法）。
>
> **v1.2.0 = Qwen-Image-2.1 适配版**（2026-10-02）。相对 v1.1.0 的改动：
> 新模型三件套、**原生透明背景开关**、**多图合成（最多 10 张参考图）**、
> KV 缓存设备选择、移除重绘强度滑块。

### App 是什么形态

**不是** Tauri，**不用 Rust，不用 NDK**。就是一个 Java WebView 壳 + 业务全在 JS：

```
comfy_android/
├── app/src/main/
│   ├── java/com/audio8/comfy/MainActivity.java   原生桥：网络 / 文件 / 图片 / Keystore
│   ├── assets/api.js        桥与网络封装 + 参数规范化（纯逻辑，可在 Node 测）
│   ├── assets/store.js      会话数据模型（零 DOM，Node 可跑单测）
│   ├── assets/app.js        界面逻辑：渲染 / 交互 / 生成编排
│   ├── assets/index.html    结构与样式
│   └── res/mipmap-*/        图标
├── comfy-release.jks       签名密钥（口令见本机私存，别弄丢）
├── test_store.js           130 项纯逻辑单测
├── test_api.js             41 项连真实 ComfyUI（含多参考图端到端）
├── ui_test/run_test.js     217 项真浏览器 UI 端到端（无头 Edge + CDP，零依赖）
│                           ⚠️ 需要一个 http 地址 ⇒ 用桌面端的 local_forward.py 把云卡 6006 转过来
└── app/build/outputs/apk/  {debug,release}/app-{debug,release}.apk
```

**分层原则**：Java 只做页面做不到的事（网络/文件/图片解码/Keystore），业务逻辑一律放 JS。
`store.js` 因此是零 DOM 的纯模型 ⇒ 能直接在 Node 里跑完整单测。

### 重新编译 APK

工具链**本机已就绪，无需下载**（路径已写进 `gradle.properties` / `local.properties`）：

| 件 | 路径 |
|---|---|
| JDK 17 | `<JDK17 目录>` |
| Gradle 8.14.3 | `%USERPROFILE%\.gradle\wrapper\dists\...\gradle-8.14.3\bin\gradle.bat` |
| Android SDK | `<Android SDK 目录>` |

```bash
cd "<Android-tts 目录>/tools/comfy_android"

# 调试版
# （wrapper jar 没随仓库携带 ⇒ 用解压出来的 gradle.bat，别用 ./gradlew）
"%USERPROFILE%\.gradle\wrapper\dists\gradle-8.14.3-bin\<hash>\gradle-8.14.3\bin\gradle.bat" assembleDebug   --offline
# 正式版（用 comfy-release.jks 签名）
"%USERPROFILE%\.gradle\wrapper\dists\gradle-8.14.3-bin\<hash>\gradle-8.14.3\bin\gradle.bat" assembleRelease --offline
```

- ⚠️ **`--offline` 必须加**：AGP 8.4.2 等依赖已在本地Gradle 缓存里，能过；去掉会试图联网。
- 产物：`app/build/outputs/apk/release/app-release.apk`
- 装完包后要手动拷到 `Android-tts/outputs/` 并改版本名。
- **wrapper jar 没随仓库携带** ⇒ 用上面那个绝对路径的 `gradle.bat`，别用 `./gradlew`。

签名（丢了就重新生成，同签名才能覆盖安装）：

```bash
keytool -genkeypair -keystore comfy-release.jks -alias comfy \
  -keyalg RSA -keysize 2048 -validity 10000 \
  -storepass 你的storepass -keypass 你的keypass \
  -dname "CN=YunKa ShengTu, O=Personal, L=Guangzhou, C=CN"
```

### ⚠️ UI 改了必须重编 APK

UI 打在 `assets/` 里，**改界面 ≠ 热更新**，不重编 APK 看不到变化。
（这跟笑樱那个 Tauri 项目的 `dist` 机制不一样，别搞混。）

改完 UI **一定要重跑两套测试**，别靠眼睛看：

```bash
node test_store.js      # 130 项纯逻辑
node ui_test/run_test.js# 217 项真浏览器（起静态服务 + 无头 Edge + CDP）
```

### App 关键设计（都是踩过坑才这么做的）

1. **网络走 Java 不走页面 fetch** —— 页面从 `file:///android_asset/` 加载，
   **Chromium 禁止 `file://` 源发跨域 fetch** ⇒ 全走 `Native` 桥的 `HttpURLConnection`。
   副作用：完全绕开 CORS，ComfyUI 不需要开 `--enable-cors-header`。
2. **图片分块传（180 KB）** —— 大图 base64 约 2 MB，单次 `evaluateJavascript` 塞不进去
   （WebView IPC 有大小限制）⇒ `__imgBegin/__imgPart/__imgEnd`。
3. **选图不用 `<input type=file>`** —— 那个要实现 `WebChromeClient.onShowFileChooser`，
   且 `content://` 在 JS 里读不到字节 ⇒ 改成 `Native.pickImage()`，Java 直接读直接传。
4. **原图落盘，页面只渲染 384px 缩略图** —— 20 张原图 base64 ≈ 40 MB 堆在 WebView 里必被回收。
5. **只加密服务器地址** —— 提示词和图片最终必须明文送到云卡，加密只挡"物理接触 + root"，
   代价却是文件管理器里看不到。地址用 Keystore AES-256-GCM，格式 `[1B IV长度][IV][密文+tag]`。
6. ⛔ **WebView 致命坑**：没设 `WebChromeClient` 时 `window.confirm()`/`alert()`
   **既不弹框又直接返回 false**，表现是"点了没反应"。
   ⇒ 页面里一律用自绘弹层（`ask()` + `#confirm`），Java 侧保留 WebChromeClient 兜底。
7. ⛔ **`run_test.js` 会自我维持地故障** —— `assemble()` 原本只复制 `api.js`/`store.js`/`app.js`
   三个文件到临时目录，**没复制 `uitest.js` 和 `native_stub.js`**，注入的 `<script>` 全 404，
   报错「测试脚本没加载进来：undefined|undefined」。以前能跑纯粹是临时目录残留旧副本。
   现在 `assemble()` 会先 `rmSync` 清空、复制全部 5 个脚本、组装后立即自检 —— **改这个文件别退回去**。

---

## 二、桌面 CLI（本目录）

### 架构

```
本机 cloudgen.py
  │  paramiko，SSH direct-tcpip 隧道
  │  ComfyUI 只监听云卡 127.0.0.1:6006，不占公网端口
  ▼
云卡 /root/autodl-tmp/ComfyUI
  │  /prompt 提交 → /history/{id} 轮询 → /view 取回
  ▼
本机 out/
```

模型三件套（**Qwen-Image-2.1**，2026-10-02 换上）：

| 角色 | 文件 | 大小 |
|---|---|---|
| DiT（int8 ConvRot） | `qwen_image_2.1_int8_convrot.safetensors` | 7.26 GB |
| 文本编码器（w4a8） | `qwen3vl_8b_w4a8.safetensors` | 6.31 GB |
| VAE（RGBA） | `qwen_image_2.1_vae_bf16.safetensors` | 676 MB |

默认采样：**1024×1024 / steps 25 / cfg 1 / shift 3.5 / euler + simple**（官方模板值）

⚠️ **cfg 必须 ≥ 1**。旧版的 2.5 那套直接照搬会出一片噪声 —— 2.1 官方模板就是 1。

### 2.1 带来的新能力

| 能力 | 说明 |
|---|---|
| **原生 RGBA** | VAE 保留 4 通道，alpha 是**连续值不是二值**。实测 1024×1024 图alpha 分布 76.6% 全透明 + 22.6% 全不透明 + 仅 0.5% 过渡像素，二值化后边界干净无毛边 |
| **多参考图** | 最多 **10 张**（`TextEncodeQwenImage21` 的 `images` 字段是Autogrow 动态键 `image_1`..`image_16`） |
| **编辑语义变了** | 不再是 `VAEEncodeForInpaint` + `KSampler.denoise`，改成「**原图 + mask 当双参考图 + 自然语言指令**」，`denoise` 参数**彻底废弃** |
| **KV 缓存** | `QwenImage21Cache` 节点控制 prefix cache 的设备/精度 —— **10GB 卡的救命项**，默认 `device=cpu` |

### 专用节点

- `TextEncodeQwenImage21` —— 一次输出 positive / negative / latent 三元组，**自带 64 通道 latent**，
  不需要 `EmptySD3LatentImage`。
- `QwenImage21Cache` —— prefix KV cache。int8 DiT(6.76 GiB) + VAE(0.63 GiB) ≈ 7.4GB 常驻显存，
  文本编码器 w4a8(5.88 GiB) 靠 DynamicVRAM 换出到系统内存（375GB 充足）。

### ComfyUI 版本门槛

**必须 ≥ 0.37.0**（2.1 于 2026-09-20 合入）。云卡现跑 **0.38.0**。

### 内存与性能（RTX 3080 / 10GB）

| 场景 | 耗时 |
|---|---|
| 1024×1024 steps=25 | 约 53–57 秒 |
| 1024×1024 steps=8 | 约 21 秒 |
| 3 张参考图 → 3 人合影 | 约 53 秒 |
| 2 张参考图融合 → 4 人同框 | 约 57 秒 |

### 开云卡服务：`cloud.cmd`（双击即用）

AutoDL 开机后 ComfyUI **不会自己跑**。双击 `cloud.cmd` 就行，也可以在终端里给它下指令：

```
cloud.cmd            双击 = 进交互菜单（先看状态，再选要做什么）
cloud.cmd status     看状态：服务 / GPU / 队列 / 日志尾
cloud.cmd start      直接启动（跳过菜单；幂等，已在跑直接返回）
cloud.cmd stop       停掉云卡 ComfyUI
cloud.cmd restart    停掉再起
cloud.cmd queue      只看队列（含每个任务的 prefix / steps）
cloud.cmd logs 80    看云端 comfy.log 最后 80 行
cloud.cmd where      看用的是哪份 SSH 配置（不连网，连不上时最有用）
```

菜单长这样：

```
====================================================
  云卡 ComfyUI
====================================================
  服务    : 运行中（HTTP 200）   pid 1352
  GPU     : NVIDIA GeForce RTX 3080, 224 MiB, 10240 MiB, 0 %
  队列    : 运行 0 个 / 排队 0 个
  日志尾  : [INFO] To see the GUI go to: http://0.0.0.0:6006
----------------------------------------------------
  1) 启动   2) 停止   3) 重启   4) 刷新状态
  5) 看日志  6) 看队列  0) 退出
  选择 >
```

底层是 `cloud_service.py`（同一套子命令，给脚本调用）：

```bash
python cloud_service.py menu       # 等价于双击 cloud.cmd
python cloud_service.py start      # 直接启动
python cloud_service.py status
```

**连不上就立刻退出**，不做长时间重试。云卡没开机时先去 AutoDL 控制台点开机，等 1~2 分钟再运行。
`cloud.cmd where` 不需要 SSH，专门用来排查"口令/端口对不对"。

> ⚠️ 改 `cloud.cmd` 必须保持 **CRLF 换行 + 纯 ASCII**：`cmd.exe` 用 LF 解析 `.bat`
> 会把 `REM` 前缀丢掉、把注释逐词当命令执行（报一串"XX 不是内部或外部命令"，2026-10-03 踩过）。

### 用法

```bash
# 云卡关机后先起 ComfyUI —— 双击 cloud.cmd，或：
python cloud_service.py start
python cloud_service.py status

# 文生图（省略 --out 落out/xy_时间戳.png）
python cloudgen.py t2i --prompt "黄昏的商店街，一个少女回头" --out a.png
# 透明背景 PNG（生成式alpha，追加透明描述词到 prompt）
python cloudgen.py t2i --transparent --prompt "一只恐龙的图标" --out d.png
# 图生图/编辑：源图作为参考图 + 自然语言指令（2.1 无 denoise）
python cloudgen.py i2i --src a.png --prompt "把她改成油画风格" --out b.png
# 多图合成：最多 10 张参考图。两张写法都行：重复传 or 逗号连写
# 不指定画幅时，输出跟随 image_1 的尺寸（要 9:16 就把 9:16 的那张放第一个）
python cloudgen.py multi --refs a.png --refs b.png --refs c.png \
  --prompt "<image1> 与 <image2> 合成一张合影" --out c.png
python cloudgen.py multi --refs "a.png,b.png,c.png" --prompt "..." --out c.png

# ★ 自由指定画幅（t2i / i2i / multi / inpaint 都支持，三种写法任选）
#   按比例：--aspect 9:16 --megapixels 4.0   （1MP≈1024²，官方 2K 档给 4.0）
#   按像素：--width 1536 --height 2752
python cloudgen.py multi --refs a.png --refs b.png \
  --aspect 9:16 --megapixels 4.0 --prompt "..." --out c.png
python cloudgen.py multi --refs a.png --width 2048 --height 2048 --prompt "..." --out c.png
# 局部重绘（蒙版法，负值＝从右/下边缘回数）
python cloudgen.py inpaint --src a.png --mask-rect "-240,-130,-1,-1" --out b.png
# 透明 PNG（蒙版法，确定性；与 2.1 的生成式透明互补）
python cloudgen.py alpha --color c.png --mask m.png --out t.png
# 查云卡节点/模型
python cloudgen.py info

# 出图后回捞提示词（ComfyUI 把整份工作流JSON 写进了 PNG 文本块）
python png_prompt.py a.png
```

### 坑（实测踩过）

- ⛔ **中文文件名会坏**：`i2i`/`inpaint`/`multi` 上传走 HTTP→JSON 多层转码，
  非ASCII 一律换`xy_<uuid>.<ext>`。云卡 `input/` 里看到乱码名是正常的。
- ⚠️ **inpaint 的 mask 语义是「白 = 要重绘」**。`--mask-rect` 支持负值从右/下回数。
- ★ inpaint 云端返回**整图**，mask 外经过 VAE 往返有损 ⇒ 默认自动 `paste_back` 贴回原像素
  （`--feather` 羽化，**alpha 原样保留**）。只想看云端原始整图用 `--no-paste`。
- ⚠️ `paste_back` 的 `keep_alpha` 一定要开：把 alpha 一起贴回会把透明区弄成不透明。
- ⚠️ 云卡**没有 `ImageThreshold` 节点**，二值化用 extra 节点的 `ThresholdMask(mask, value)`。
- ⚠️ 上传走 ComfyUI 的 `/upload/image`，**不是 SFTP**（SFTP 只在 `cloud_sh.py --put/--get`）。
- 隧道 timeout 给到 1800s，出图排队时连接不能断。
- ⚠️ 出图有随机性，`--seed` 不指定是时间戳取模。判「稳定复现」必须钉 seed 重跑对照。
- ⛔ **下载源别用 hf-mirror** —— 实测只有 0.3~1.5 MB/s（14.3GB 要十几个小时）。
  **ModelScope 稳定 5.2 MB/s**，14.3GB 只用 9 分 30 秒。
  仓库路径是 `Comfy-Org/Qwen-Image-2.1`（**不是** `Qwen/Qwen-Image-2.1`，后者 500）。
- ⚠️ 云卡 curl 7.68 **不支持 `--retry-all-errors`**，加了会 8 个分片一秒全废。
- ⚠️ 所有 curl **必须加 `-L`**：ModelScope 会 302 跳 CDN，不跟就只拿到 366 字节的 HTML。
- ⚠️ 加载器**还是 `UNETLoader`**（别换成 `LoadDiffusionModel`—— 云卡没这节点）。
  它的 `display_name` 就是 "Load Diffusion Model"，只是内部 class 名沿用旧叫法。
- ⚠️ seed 回显要去 `node["6"]["inputs"]["seed"]` 找，不是节点顶层；0.38 后
  `entry["prompt"]` 也从 dict 变成了 5 元素列表。
- ⛔ **`pkill -f 'main.py --port 6006` 会杀掉执行它的 bash 自己** → SSH 直接断连（exit=-1、无输出）。
  正确做法是遍历 `/proc/*/exe` 只杀 python 本体。已固化进 `remote/start_comfy.sh`。
- ⛔ **`Error reading SSH protocol banner` ≠ 云卡挂了** —— AutoDL 的 sshd 在未认证连接达上限时
  会回一句**纯文本** `Exceeded MaxStartups\r\n`，它不是 SSH banner，paramiko 于是抛这个
  误导性的异常。实测 6 次连接里有 2 次撞上。
  判别方法：raw socket 能拿到 `SSH-2.0-OpenSSH_8.2p1...` 就说明云卡好得很。
  `cloud_sh.py` 与 `cloudgen.py` 的 `connect()` 都已加退避重试（2026-10-02）。
- ⛔⛔ **Autogrow 输入在 API 格式里必须用「扁平点号键」**（2026-10-03 定位，代价＝三轮实验白跑）：
  `TextEncodeQwenImage21.images` 是 `io.Autogrow`（`COMPY_AUTOGROW_V3`），`/prompt` 里
  子槽位必须写成 **`"images.image_1": ["r0", 0]`**；写成嵌套的
  `"images": {"image_1": [...]}` 会被引擎**静默丢弃** —— 既不生效、也**不报错**。

  ```python
  # comfy_api/latest/_io.py（v0.38.0 与 master 行为一致，已逐字比对）
  # ① Autogrow 容器名「故意」不进已知输入表
  #    _expand_schema_for_dynamic(): "purposely do not include self in out_dict"
  #    ⇒ validate_prompt 只遍历 required∪optional，多出来的嵌套键既不校验也不使用
  # ② 展开时用「点号全名」判断槽位到底有没有接线
  expected_id = finalize_prefix(curr_prefix, name)   # → "images.image_1"
  if expected_id in live_inputs:                     # ← 嵌套写法永远命中不了
  # ③ 命中不了 ⇒ dynamic_paths["images"] 落成 EMPTY_DICT
  #    build_nested_inputs(): default == EMPTY_DICT → value = {} → 节点收到 images={}
  ```

  后果有两个，此前被当成两个独立 bug 追：
  1. **参考图一张都进不去**（模型退化成纯文生图，所谓"身份保持"无从谈起）；
  2. `latent_w = latent_h = resolution or 1024`，`resolution=0` 时 `0 or 1024`
     ⇒ **出图恒定 1024×1024**（这就是那个"要 9:16 死活做不到"的 P0）。

  现在 `run_workflow` 提交前会跑 `_selfcheck_autogrow()`，同类写法**必定炸出来**；
  `test_workflow.py` 里有对应回归测试。诊断命令：`python png_prompt.py 图.png`
  会打印「参考图 N 张」——**0 张就是踩了这个坑**。

### ★ 画幅怎么指定（2026-10-03 打通）

**默认行为**：`TextEncodeQwenImage21` 的 latent 输出被**硬编码**成「第一张参考图的尺寸」，
节点 tooltip 原话 —— *Empty latent on the first reference image's size, to match with
sampling as any other size shifts the edit.*
⇒ 不指定时，输出只能跟随 `image_1`。
**注意 `resolution` 不是画幅**：它是「参考图统一缩放到 resolution×resolution 的面积」。

**指定画幅**：显式给 `--width/--height`，或给 `--aspect` + `--megapixels`。
此时 cloudgen 会插一个 `EmptyLatentImage(width, height)` **顶掉节点自带的 latent**。
这正是**官方 image_edit 模板的接法** —— 模板里就是
`EmptyLatentImage` + `ComfySwitchNode` 二选一（我们这里用客户端判断代替那个 switch 节点，
工作流更干净）。

⚠️ 单位是**像素**，不要自己除 2。`EmptyLatentImage.generate` 返回的 latent 字典里带了
`downscale_ratio_spacial = 8`（`nodes.py:1274`），`common_ksampler` 会把它转交给
`fix_empty_latent_channels`；后者发现 `8 ≠ 16`（Qwen 的 `spacial_downscale_ratio = 16`），
就用 nearest-exact 把 latent 空间尺寸**折半** ⇒ 最终 `H/16 × W/16`，解码正好是 `W×H`。
通道数则由 `repeat_to_batch_size` 从 4 补齐到 64。

⚠️ tooltip 的后半句仍然有效：**latent 尺寸与参考图差太多时，编辑内容会偏移**。
指定画幅后建议让参考图也接近目标尺寸（必要时把 `--resolution` 调大，别留 0）。

官方 README 的 2K 档参考值：`1:1` 2048²、`4:3` 2400×1792、`3:4` 1792×2400、
`3:2` 2528×1696、`2:3` 1696×2528、`16:9` 2752×1536、`9:16` 1536×2752。
`calc_size()` 按公式算并取整到 32 的倍数（4MP 的 9:16 会得到 1536×2720），
想要官方那几个精确值就直接用 `--width/--height`。

### ⏳ 多参考图的指代规范（官方硬性要求，非可选）

> ⚠️ **本节原有的「实测」依据已作废（2026-10-03）**。
> exp3 / exp5 / exp6 三轮实验其实**一张参考图都没送进模型**（见上方 Autogrow 的坑），
> 所以当时"换了指代写法结果就变好"是出图随机性，**不是因果**。
> 下面引用的**官方 System Prompt 原文仍然有效**（那是官方文档的硬性规定，不依赖我们的实验），
> 但标着 ✅ 的实测结论要等修好之后重跑才能确认。**在此之前别拿它当结论。**

多图角色丢失的**头号根因已经找到**：参考图根本没送进模型（见上方 Autogrow 的坑）。
修好接线之后，官方 PE System Prompt 的 "Image Reference Rules" 依然是硬性要求：

> **For Multi-Image Input (N >= 2)**, the rewritten instruction MUST use
> `<image1>`, `<image2>`, ... to refer to each input image.
> **Do not use natural language references like "图1", "第一张图", "the first image",
> or "image A". This tagging format is mandatory and non-negotiable.**
> For single-image input (N = 1), **do NOT use tags**.

同一份文档还有两条直接命中「人物特征没保留」的：

> **Identity is the hardest invariant.** ... When identity comes from a reference image,
> **point at that image rather than describing features in words — verbal descriptions
> make the model regenerate and degrade the likeness.**
>
> **Say what stays, without repainting it.** ... prefer one blanket preservation clause
> over walking the frame. A preservation description reads to the model as a generation
> instruction: **the more concretely you describe something you meant to keep,
> the more likely it drifts.**

⇒ **写"保持同样的五官、脸型、发型"反而会让模型重新生成、相似度下降。**
   正确做法是「`<image1>` 中的人脸保持不变」这种**指着图说**。

### 源码依据：参考图是原封不动喂进去的

`comfy_extras/nodes_qwen.py` 的 `TextEncodeQwenImage21.execute`：

```python
if (width, height) == (samples.shape[3], samples.shape[2]):
    s = image[:1]# 尺寸一致 → 直接用原图，一个像素都没动
else:
    s = comfy.utils.common_upscale(samples, width, height, "lanczos", "disabled")
if s.shape[-1] > 3:
    rgb = rgb * s[:, :, :, 3:] + (1.0 - s[:, :, :, 3:])   # 视觉塔看 alpha over 白底
images_vl.append(rgb)          # ← 给 Qwen3-VL 看
ref_latents.append(vae.encode(s))   # ← 给 DiT，身份靠它
```

两个结论：
- `resolution=0` 走else 分支，按原图尺寸取整到 32 的倍数 ⇒ **保真度最高**，
  改成 `resolution=1024` 只会更糟或持平。（此前怀疑方向搞反了，已纠正。）
- **2.1 删掉了旧版 `TextEncodeQwenImageEditPlus` 里的 `image_prompt`
  （`"Picture N: <|vision_start|>..."`）和 `llama_template`**
  ⇒ 多张图在视觉序列里是**并列无编号**的，模型没有任何锚点把文字和具体某张图绑定。
  这就是必须由用户在prompt 里显式写 `<imageN>` 的原因。

### App 里的 @ 标签补全

`multi` 模式提示词框输入 `@` 会弹出缩略图选择框，选中即插入 `<imageN>`。
之所以做成补全而不是让用户手打：`imageN` 是**按下标绑定**的，
删掉中间某张图后面所有 N 都会平移，手打极易对错图。参考图一变，浮层自动撤掉。

`api.js` 还有两道校验：
- 参考图 ≥2 张却没写 `<imageN>` ⇒ 拦下并提示
- `<imageN>` 的 N 超出参考图张数 ⇒ 拦下并提示正确范围
- 单图给标签不拦（官方说单图别用，但无害）

### 官方完整方案（未启用，可选升级）

官方 Image Edit 模板比我们的工作流多一整层 **PE 提示词增强器**：
`TextGenerate` 节点 + `qwen3.5_9b_qwen_image_2.1_pe_i2i.int8_convrot.safetensors`
（放在 `models/text_encoders/`），由一份专门的 System Prompt 把用户口语指令
改写成专业编辑指令。我们目前是「用户文本直接进编码器」。
磁盘只剩约 8GB，装不下这个 5GB 模型，故暂缓。

---

## 三、手机怎么连上云卡（Termux 隧道）

### ⚠️ 为什么不能直连公网

**AutoDL 部分地区实例的 http/https 公网映射只对企业认证用户开放**（2026-09-15 确认）。
个人用户拿到 `region-xx.seetacloud.com:xxxxx` 会看到：

> 为配合监管要求该地区 http/https 服务仅对企业认证后的用户开放

**6006 / 6008 同理** —— 受限的是"公网映射"这件事，不是某个端口。
⇒ App 设置页现在**只提供「一键填入 `http://127.0.0.1:6006`」**，不再引导填公网地址。

### 唯一通路：SSH 隧道（公钥免密已配好，直接可用）

云卡 sshd 实测：`AllowTcpForwarding yes`·`PubkeyAuthentication yes`·`PasswordAuthentication yes`·
`PermitRootLogin yes`·`ClientAliveInterval 0`（不主动踢隧道）。

**手机已部署 SSH 公钥** ⇒ 下面第3 步的免密配置**已跳过**，直接起隧道即可。

```bash
# Termux（必须 F-Droid 版，商店版已停维护）
pkg update -y && pkg install -y openssh

# 起隧道：手机 6006 → 云卡 127.0.0.1:6006
termux-wake-lock
ssh -CN -L 6006:127.0.0.1:6006 -p 你的SSH端口 \
  -o ServerAliveInterval=30 -o ServerAliveCountMax=3 \
  -o ExitOnForwardFailure=yes \
  root@connect.你的实例地址.seetacloud.com
```

- `-N` 只转发不开 shell；`-C` **不要加**（传 PNG 压不动还白耗 CPU）；`-g` **不要加**（扩大暴露面）
- 跑起来**什么都不输出、光标停着= 正常**，别以为卡住
- **这个 Termux 会话不能关**。按 Home 切后台可以。

### 让隧道活下来（持久化）

1. **系统设置 → 应用 → Termux → 电池 → 「不优化」/「无限制」**
2. `termux-wake-lock`（上面命令已带）
3. 想要**开机自动起**：装 `Termux:Boot`（同F-Droid），然后

```bash
mkdir -p ~/.termux/boot
cat > ~/.termux/boot/tunnel.sh <<'EOF'
#!/data/data/com.termux/files/usr/bin/sh
termux-wake-lock
exec ssh -CN -L 6006:127.0.0.1:6006 -p 你的SSH端口 \
  -o ServerAliveInterval=30 -o ServerAliveCountMax=3 \
  -o ExitOnForwardFailure=yes \
  root@connect.你的实例地址.seetacloud.com
EOF
chmod +x ~/.termux/boot/tunnel.sh
```

装完 Termux:Boot **要先手动打开一次它的 App**，脚本才会被开机拉起。

### 然后在 App 里

右上角 **⚙** → 「填入 `http://127.0.0.1:6006`」→ **测试连接**（✅）→ 保存

### 隧道故障排查

这套方案已在真机上跑通，**照上面两步走即可，不再有坑**。万一出问题：

| 现象 | 原因 & 处理 |
|---|---|
| 测试连接一直转圈 | Termux 隧道掉了，回那个会话重跑命令 |
| 隧道命令秒退 | `bind: Address already in use` = 手机 6006 被占（多半是代理类 App）→ 换 `-L 6007:127.0.0.1:6006`，App 地址也改成 `http://127.0.0.1:6007`。**左边是手机端口，右边 6006 是云卡的、不能改** |
| 生成到一半失败 | 隧道中途断。**已生成的图和对话都在本机，不会丢**，重连后继续 |
| 长时间不用后连不上 | Android 杀了 Termux 后台 → 电池优化改成「不优化」，并确认 `termux-wake-lock` 生效 |

**隧道开着时那些 `WARNING: ... post-quantum key exchange ...` 是正常提示，不是错误**，忽略即可。

**数据不会因隧道断而丢**：对话记录、提示词、图片全在 App 私有目录 `/data/data/com.audio8.comfy/files/`。
断的只是"够不着云卡"这一件事。

---

## 四、目录结构（本目录）

```
云卡生图\
├── cloudgen.py                 桌面 CLI 主入口（t2i/i2i/inpaint/alpha/info）
├── cloud_sh.py                 SSH 小工具（put/get/执行远端脚本）
├── png_prompt.py               从 PNG 回捞提示词
├── cloud-comfy.config.json     云卡连接信息（含口令，gitignore）
├── remote/start_comfy.sh       云卡侧启动 ComfyUI（幂等）
├── requirements.txt            paramiko / Pillow / numpy
├── out/    input/              产物与源图暂存（gitignore）
└── README.md                   本文件
```

**Android App 不在这里** → 见第一节的⬆️ 路径索引。

## 五、已知遗留

- ⚠️ **云卡口令有两份副本**（本目录 + `笑樱-galgame\tools\`），
  因为笑樱的语音下载链路也用同一台云卡。**AutoDL 端口/口令变更时要改两处。**
- App **不能改 UI 后免重编**（UI 打在 assets 里，改界面必须重编 APK），
  也不支持后台多任务（单人单任务，一次一张图）。
- 云卡**关机或 ComfyUI 没起来**时隧道再通也没用 —— 出图前确认云卡在运行。


---

## 显存预算闸（2026-10-03 实测加入）

> Qwen-Image-2.1 的 `TextEncodeQwenImage21.resolution` 就是模型自带的「云卡侧降质开关」——
> 节点 tooltip 原话：*Reference images are resized to about resolution x resolution pixels,
> at multiples of 32, preserving aspect ratio. 0 keeps each reference at its own size.*
> **它的 default 是 1024**，我们之前为了保真把它置成 0，于是 4K 参考图原样进模型。

### ⛔ 超限的后果不是"任务报错"，而是整个 ComfyUI 进程被 Abort

实测日志（`/root/autodl-tmp/comfy.log`）：

```
Ran out of memory when regular VAE decoding, retrying with tiled VAE decoding.
terminate called after throwing an instance of 'c10::AcceleratorError'
  what():  CUDA error: invalid argument
Fatal Python error: Aborted
```

崩溃点是 **VAEDecode**，不是采样。**一次越界请求会把云卡服务整个打死**，别的任务一起陪葬。

### 实测边界（RTX 3080 10GB，steps=1；OOM 在编码/解码，与步数无关）

| 参考图合计 | 输出 | 结果 |
|---|---|---|
| 2.00MP | 1.00MP | OK（26s） |
| 4.50MP | 1.00MP | OK（48s） |
| 3.57MP | 4.18MP | OK —— 真实跑通过的那一档 |
| 4.50MP | 3.98MP | **OOM** |
| 8.00MP | 1.00MP | **OOM** |
| — | 9.00MP | OK 但 66s/步（实际不可用） |
| — | 12.25MP | **进程 Abort** |

⇒ **参考图的每 MP 比输出的每 MP 贵**（多走 VLM 视觉编码 + VAE 编码 + 拼进 prefix KV）。
⇒ 两条闸：`REF_MAX_MP = 4.5`、`TOTAL_MAX_MP = 8.0`（参考图 + 输出）。

### 行为

- 编辑场景（i2i / multi / inpaint）的参考图档位**默认 1024**（≈1MP/张），不再置 0。
- 参考图张数多时自动下调（10 张 → 672），保证合计 ≤ 4.5MP。
- 输出画幅超预算时**等比缩小**（保持比例），并把原因打印出来。
- 想让参考图保持原尺寸：显式传 `--resolution 0`，CLI 会读本机图片尺寸过闸 ——
  安全就放行（保真），超了才降档。
- 新增探针工具 `probe_vram.py`：阶梯测显存上限，**服务被拖崩会自动重启**并抓日志标记。

> ⚠️ 这两个阈值是**从 8 个实测点拟合**的，不是理论推导。要改请先跑 `probe_vram.py` 复测。


---

## 四、网页控制台（comfy_web）：云卡 / 本机 双模式

`Android-tts/tools/comfy_web/start_web.bat` 双击后会问一句：

```
  ComfyUI Web Console  (Qwen-Image-2.1)
  ----------------------------------------------------
   [1] 云卡    RTX 3080 10GB   走 SSH 隧道（云卡要先开机）
   [2] 本机    RTX 3070  8GB   用 <Qwen-Image-2.1 目录>

      本机更省事但慢很多；云卡快，但额度有限。

  输入 1 或 2 后回车：
```

也可以跳过菜单：`start_web.bat 1` / `start_web.bat 2`（第二个参数是端口）。
选 2 时会**自动把本机 ComfyUI 拉起来**，不用手动开。

### 本机那份 ComfyUI（2026-10-04 装好）

| 项 | 值 |
|---|---|
| 位置 | `<ComfyUI 目录>`（ComfyUI **0.38.0**，与云卡同版本） |
| Python | `<ComfyUI 目录>\.venv`（基于 `<基础 Python 目录>`） |
| torch | **2.8.0+cu128** |
| 模型 | `extra_model_paths.yaml` 指向 `<Qwen-Image-2.1 目录>`（不复制文件） |
| 起停 | `python comfy_web\local_comfy.py ensure / stop / status / logs` |
| 日志 | `<ComfyUI 目录>\comfy_local.log` |

⚠️ **torch 必须 ≥ 2.7**：ComfyUI 0.38 的 `comfy_kitchen` 用了 `list[int]` 类型注解，
torch 2.6 的 `infer_schema` 不认，启动直接 `ValueError`。
⚠️ **PyPI（含清华/阿里镜像）上的 Windows torch 是 CPU 版**（`2.8.0+cpu`，CUDA 不可用）。
CUDA 版只能从 `https://download.pytorch.org/whl/cu128` 取 —— 国内镜像都没有 Windows 的 cu128 wheel。

### 实测性能（3070 8GB vs 云卡 3080 10GB）

| 场景 | 本机 3070 8GB | 云卡 3080 10GB |
|---|---|---|
| 启动 | 13~14 秒 | ~8 秒 |
| t2i 1024² / 10 步 | 首次 58s，**稳态 74s** | 20~40s |
| i2i（1080×1920 参考图）/ 10 步 | **82s** | 20~40s |

⇒ **本机大约是云卡的 1/2 ~ 1/3 速度**。原因很实在：DiT 6.8GB + 文本编码器 5.9GB
一共 12.7GB，8GB 显存装不下，ComfyUI 只能反复把权重换进换出。
好处是**不吃额度**；着急或要大批量出图还是走云卡。

参考图场景**没有 OOM**（生成中显存空闲 2.5~3.8GB），但已经贴边。

### ⚠️ 两套显存预算闸（按模式自动切换）

超限的后果很重：**一次越界请求会把 ComfyUI 进程整个打死**（不只是那个任务失败），
所以只能**提交前拦**。两个模式的闸不一样：

| 模式 | 参考图合计 | 参考图 + 输出 |
|---|---|---|
| 云卡 3080 10GB | 4.5 MP | 8.0 MP |
| **本机 3070 8GB** | **3.0 MP** | **6.0 MP** |

本机上限是实测标定的（steps=1；OOM 发生在编码/解码阶段，与步数无关）：

| 输出画幅 | 结果 |
|---|---|
| 1.00 MP（1024²） | ✅ 45s（含首次加载 12.7GB 模型） |
| **4.00 MP（2048² = 2K）** | ✅ **30s，很轻松** |
| 6.25 MP（2560²） | ✅ 但 **144s**（已开始换页，不可用） |
| 9.00 MP（3072²） | ❌ 超时 420s + 回退 tiled VAE |

⇒ 本机的定位就是"**2K 输出畅通**"：`约 2MP 参考图 + 2K 输出 = 6.0MP` 正好放行。
超限**自动等比下调 + 明确提示**，优先级是**先保参考图档位、让输出让位**
（反过来会把参考图挤到 320 档、等于白给）。

页头副标题按实际模式渲染（`本机 RTX 3070 8.0GB` / `云卡 RTX 3080 10GB`），
鼠标悬停可看到当前生效的预算值。`python server.py --where` 也会打印。
本机这套值来源：`probe_local_vram.py`（可重跑重标定）。
