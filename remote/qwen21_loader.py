# -*- coding: utf-8 -*-
"""从 ComfyUI 单文件权重加载 Qwen-Image-2.1 的 DiT / 文本编码器（供 diffusers 官方训练脚本用）。

【为什么需要它】
官方 HF 仓库是「分片 + 未融合」布局；云卡上已有的却是 Comfy-Org 单文件。
实测（2026-10-12，比对 key 与形状）——**直接软链不可行**：

    模块       comfy 张量   hf 张量   差异
    DiT          265        297     仅 img_mlp：comfy `gate_up`(24576,4096)
                                    = hf `gate_layer`(12288,4096) + `proj`(12288,4096)（沿 dim0 拼接）
    文本编码器     750        750     一一对应，仅前缀 `model.X` → `model.language_model.X`
    VAE          238        238     0 个同名（ComfyUI 原生命名）⇒ 不走本模块，直接下 HF 版（1.35G）

融合顺序是**按字节验证**的（取 HF 张量首行 8KB 与 comfy 两半比对）：
    gate_up[:12288] == gate_layer ，gate_up[12288:] == proj
⇒ `chunk(2, dim=0)` 依次得到 (gate_layer, proj)。
⚠️ 顺序错了不会报错，只会静默训出废 LoRA —— 所以这条不是靠惯例猜的。

【省内存设计（重要）】
AutoDL 容器内存被 cgroup 限死，**`free` 看到的是宿主机、会骗人**（2026-10-12 实测：
无卡模式 `memory.max` 只有 2 GiB，直接 OOM Killed）。所以这里：
  1. **按目标 dtype 构建**，绝不让它默认 fp32（fp32 构建＝权重占 2 倍）；
  2. **惰性映射 + 直接替换 `.data`**，不做 `dict()` 整体物化、不做二次拷贝
     ⇒ CPU 峰值 ≈ 权重本身一份。
`Q21_LOAD_DEVICE=cuda` 可切到「meta 构建 + 直接铺到显存」，CPU 几乎不占。

【用法】见 `run_train_qwen21.py`：先 `patch()`，再 runpy 跑官方脚本。
"""
from __future__ import annotations

import os
from collections.abc import Mapping

import torch
from safetensors import safe_open

MODEL_DIR = os.environ.get("Q21_MODEL_DIR", "/root/autodl-tmp/Qwen-Image-2.1")
DIT_SRC = os.environ.get(
    "Q21_COMFY_DIT",
    "/root/autodl-tmp/ComfyUI/models/diffusion_models/qwen_image_2.1_bf16.safetensors",
)
TE_SRC = os.environ.get(
    "Q21_COMFY_TE",
    "/root/autodl-tmp/ComfyUI/models/text_encoders/qwen3vl_8b_bf16.safetensors",
)
LOAD_DEVICE = os.environ.get("Q21_LOAD_DEVICE", "cpu")


def _as_dtype(d):
    if isinstance(d, str):
        return getattr(torch, d)
    return d or torch.bfloat16


def _pick_device():
    if str(LOAD_DEVICE).startswith("cuda") and torch.cuda.is_available():
        return LOAD_DEVICE
    return "cpu"


class ComfyStateDict(Mapping):
    """惰性 Mapping：`__getitem__` 时才从单文件读那个张量。

    这样配合「直接替换 param.data」就能做到峰值 ≈ 权重一份（不做 dict 物化、不拷贝）。
    """

    def __init__(self, src: str, mode: str, target_keys=None, device: str = "cpu"):
        self.src, self.mode, self.device = src, mode, device
        with safe_open(src, framework="pt", device="cpu") as f:
            raw = list(f.keys())
        plan: dict[str, tuple[str, int | None]] = {}
        for k in raw:
            if mode == "dit" and k.endswith(".img_mlp.gate_up.weight"):
                base = k[: -len(".gate_up.weight")]
                plan[base + ".gate_layer.weight"] = (k, 0)
                plan[base + ".proj.weight"] = (k, 1)
            else:
                nk = k
                if (
                    mode == "te"
                    and target_keys is not None
                    and k not in target_keys
                    and k.startswith("model.")
                ):
                    alt = "model.language_model." + k[len("model."):]
                    if alt in target_keys:
                        nk = alt
                plan[nk] = (k, None)
        self._plan = plan

    def __len__(self) -> int:
        return len(self._plan)

    def __iter__(self):
        return iter(self._plan)

    def __contains__(self, k) -> bool:
        return k in self._plan

    def keys(self):
        return self._plan.keys()

    def __getitem__(self, k: str) -> torch.Tensor:
        raw_key, part = self._plan[k]
        with safe_open(self.src, framework="pt", device=self.device) as f:
            t = f.get_tensor(raw_key)
        if part is not None:
            t = t.chunk(2, dim=0)[part]      # 视图，不额外占内存
        return t


def _persistent_state(model) -> dict:
    """{完整名: tensor}，口径与 `model.state_dict()` 一致（跳过非持久 buffer）。

    ⚠️⛔ 为什么不用 `model.named_buffers()`：它会把**非持久 buffer** 也算进来，
    而 RoPE 的 `rotary_pos_emb.inv_freq` / `original_inv_freq` 正是 transformers 在
    `_from_config` 时按 config **算出来的非持久 buffer** —— `state_dict()` 里没有它、
    权重文件里也没有。算进来就会报「映射不完整：missing=4」的**误报**，训练直接起不来。

    ⚠️ 也不能直接用 `model._non_persistent_buffers_set` 过滤 `named_buffers()` 的结果：
    那个集合存的是**每个子模块自己的相对名字**（`inv_freq`），而 `named_buffers()`
    给的是**完整路径**（`model.visual.rotary_pos_emb.inv_freq`），两者对不上 ⇒ 过滤不掉。
    所以这里逐模块看各自的集合（2026-10-14 连着踩了这两下）。
    """
    out: dict = {}
    for mod_name, mod in model.named_modules():
        skip = getattr(mod, "_non_persistent_buffers_set", set())
        for n, t in mod._parameters.items():
            if t is not None:
                out[("%s.%s" % (mod_name, n)) if mod_name else n] = t
        for n, t in mod._buffers.items():
            if t is not None and n not in skip:
                out[("%s.%s" % (mod_name, n)) if mod_name else n] = t
    return out


def _apply_weights(model, sd: ComfyStateDict):
    """把 sd 里的张量直接铺到模型上（替换而非拷贝），并回报缺失/多余。

    用 `_persistent_state()` 取键 ⇒ 口径与 `state_dict()` /
    `validate_qwen21_loader.py` 完全一致，不会有非持久 buffer 的误报。
    """
    named = _persistent_state(model)
    missing = [n for n in named if n not in sd]
    unexpected = [k for k in sd.keys() if k not in named]
    done = 0
    with torch.no_grad():
        for name, tensor in named.items():
            if name in sd:
                tensor.data = sd[name]
                done += 1
    return missing, unexpected, done


def _build(cls, cfg, dtype):
    """按目标 dtype 构建 —— 绝不让它落到 fp32。"""
    for kw in ({"torch_dtype": dtype}, {"dtype": dtype}, {}):
        try:
            return cls.from_config(cfg, **kw)
        except TypeError:
            continue
    return cls.from_config(cfg)


def load_transformer(dtype=torch.bfloat16, device: str | None = None):
    from diffusers import QwenImage21Transformer2DModel

    dtype = _as_dtype(dtype)
    device = device or _pick_device()
    cfg = QwenImage21Transformer2DModel.load_config(os.path.join(MODEL_DIR, "transformer"))
    if device == "cpu":
        model = _build(QwenImage21Transformer2DModel, cfg, dtype)
    else:
        with torch.device("meta"):
            model = QwenImage21Transformer2DModel.from_config(cfg)
    sd = ComfyStateDict(DIT_SRC, "dit", device=device)
    missing, unexpected, done = _apply_weights(model, sd)
    if missing or unexpected:
        raise RuntimeError(
            "DiT 权重映射不完整：missing=%d unexpected=%d\n  missing[:6]=%s\n  unexpected[:6]=%s"
            % (len(missing), len(unexpected), missing[:6], unexpected[:6])
        )
    print("[qwen21_loader] DiT 就绪：%d 个张量 ← %s" % (done, DIT_SRC), flush=True)
    if device == "cpu":
        model = model.to(dtype)
    return model


def load_text_encoder(dtype=torch.bfloat16, device: str | None = None):
    from transformers import Qwen3VLConfig, Qwen3VLForConditionalGeneration

    dtype = _as_dtype(dtype)
    device = device or _pick_device()
    cfg = Qwen3VLConfig.from_pretrained(os.path.join(MODEL_DIR, "text_encoder"))
    if device == "cpu":
        try:
            model = Qwen3VLForConditionalGeneration._from_config(cfg, dtype=dtype)
        except TypeError:
            model = Qwen3VLForConditionalGeneration._from_config(cfg)
    else:
        with torch.device("meta"):
            model = Qwen3VLForConditionalGeneration._from_config(cfg)
    target = set(model.state_dict().keys())
    sd = ComfyStateDict(TE_SRC, "te", target_keys=target, device=device)
    missing, unexpected, done = _apply_weights(model, sd)
    if missing or unexpected:
        raise RuntimeError(
            "文本编码器权重映射不完整：missing=%d unexpected=%d\n  missing[:6]=%s\n  unexpected[:6]=%s"
            % (len(missing), len(unexpected), missing[:6], unexpected[:6])
        )
    print("[qwen21_loader] 文本编码器就绪：%d 个张量 ← %s" % (done, TE_SRC), flush=True)
    if device == "cpu":
        model = model.to(dtype)
    return model


def _effective(path, subfolder) -> str:
    return os.path.abspath(os.path.join(str(path), subfolder) if subfolder else str(path))


def patch() -> None:
    """把两个类的 from_pretrained 换成「命中我们目录 → 改从 Comfy 单文件加载」。

    ⚠️ 只拦**我们那个目录**；其它路径（例如真的 HF 仓库）原样透传。
    diffusers 的 pipeline 内部加载组件时用的就是这两个类对象，所以这里 patch 一下，
    `QwenImage21Pipeline.from_pretrained()` 也会跟着走我们的加载器。
    """
    from diffusers import QwenImage21Transformer2DModel
    from transformers import Qwen3VLForConditionalGeneration

    orig_t = QwenImage21Transformer2DModel.from_pretrained.__func__
    orig_e = Qwen3VLForConditionalGeneration.from_pretrained.__func__
    dit_dir = os.path.abspath(os.path.join(MODEL_DIR, "transformer"))
    te_dir = os.path.abspath(os.path.join(MODEL_DIR, "text_encoder"))

    @classmethod
    def t_from_pretrained(cls, path, *a, **kw):
        if _effective(path, kw.get("subfolder")) == dit_dir:
            return load_transformer(kw.get("torch_dtype") or kw.get("dtype") or torch.bfloat16)
        return orig_t(cls, path, *a, **kw)

    @classmethod
    def e_from_pretrained(cls, path, *a, **kw):
        if _effective(path, kw.get("subfolder")) == te_dir:
            return load_text_encoder(kw.get("torch_dtype") or kw.get("dtype") or torch.bfloat16)
        return orig_e(cls, path, *a, **kw)

    QwenImage21Transformer2DModel.from_pretrained = t_from_pretrained
    Qwen3VLForConditionalGeneration.from_pretrained = e_from_pretrained
