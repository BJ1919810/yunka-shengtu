# -*- coding: utf-8 -*-
"""Qwen-Image-2.1 LoRA 训练启动器：打两个补丁，然后原样跑官方脚本。

== 补丁 1：模型从哪来 ==
官方脚本里模型是这么取的：
    QwenImage21Transformer2DModel.from_pretrained(path, subfolder="transformer", ...)
    Qwen3VLForConditionalGeneration.from_pretrained(path, subfolder="text_encoder", ...)
HF 仓库是 30G 分片版，云卡上只有 ComfyUI 单文件（且 DiT 的 img_mlp 是融合过的）。
`qwen21_loader.patch()` 把这两个类的 from_pretrained 拦下来改成读 Comfy 单文件
—— **不用下载第二份 30G 模型**。

== 补丁 2：per-image caption 怎么读 ==
官方脚本只会 `load_dataset(args.dataset_name, args.dataset_config_name, cache_dir=...)`，
**不传 data_dir**；直接给一个本地目录路径，datasets 会当成 json/parquet 去解析
⇒ 结果是**没有 image 列**、caption 也读不到（实测 2026-10-12）。
这里把「本地目录」改写成 `load_dataset("imagefolder", data_dir=目录)`，
之后 `--image_column image --caption_column text` 才生效。

== 用法 ==
    cd /root/autodl-tmp/lora-train
    /root/autodl-tmp/train-env/bin/python run_train_qwen21.py \
        --pretrained_model_name_or_path /root/autodl-tmp/Qwen-Image-2.1 \
        --dataset_name /root/autodl-tmp/dataset \
        --image_column image --caption_column text \
        ... （完整参数见 train_qwen21.sh）

用 accelerate 包一层也行：
    /root/autodl-tmp/train-env/bin/accelerate launch run_train_qwen21.py ...
"""
import os
import runpy
import sys

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)

import qwen21_loader  # noqa: E402

SCRIPT = os.path.join(HERE, "train_dreambooth_lora_qwenimage21.py")
if not os.path.isfile(SCRIPT):
    sys.exit("[缺少官方脚本] 没找到 %s\n（应与 train_dreambooth_lora_qwenimage21.py 同目录）" % SCRIPT)


def patch_datasets() -> None:
    """让 `--dataset_name <本地目录>` 走 imagefolder，从而支持 per-image caption。"""
    import datasets

    orig = datasets.load_dataset

    def patched(path, *a, **kw):
        if isinstance(path, str) and os.path.isdir(path):
            print("[run_train] datasets 补丁：%s → load_dataset('imagefolder', data_dir=…)" % path,
                  flush=True)
            return orig("imagefolder", data_dir=path, **kw)
        return orig(path, *a, **kw)

    datasets.load_dataset = patched


def patch_model_card() -> None:
    """不让收尾阶段去连 HuggingFace（2026-10-14 用户要求：反正就是不让连 hf）。

    链路：官方脚本收尾 → 它自己的 `save_model_card()` → diffusers 的
    `load_or_create_model_card()` → 去 huggingface.co 探/拉模型卡。
    海外不通 ⇒ 指数退避重试，**训练明明跑完了，任务却吊着不结束**
    （实测：20 步早训完、LoRA 也存了，任务还挂在重试上）。

    这里直接换成本地空卡 —— 模板本来就在 diffusers 包里
    （`diffusers/utils/model_card_template.md`），根本不需要联网。
    和 job.sh 里的 `HF_HUB_OFFLINE=1` 互为双保险（那条管住所有别的 HF 调用）。

    ⚠️ 必须在 `runpy.run_path(SCRIPT)` **之前** 打好：脚本里是
    `from diffusers.utils.hub_utils import load_or_create_model_card`，
    这个 from-import 发生在脚本执行时，晚于本函数 ⇒ 它拿到的是替换后的版本。
    """
    try:
        import diffusers.utils.hub_utils as hub_utils
    except ImportError:
        return

    def _local_only(*_a, **_kw):
        try:
            from huggingface_hub.repocard import ModelCard
        except ImportError:                      # huggingface_hub 旧版
            from huggingface_hub import ModelCard
        return ModelCard(
            "# DreamBooth LoRA (local training)\n\n"
            "模型卡未生成：训练离线完成，已禁用联网。\n"
        )

    hub_utils.load_or_create_model_card = _local_only
    print("[run_train] 已禁用 HF 模型卡（收尾不再联网）", flush=True)


qwen21_loader.patch()
patch_datasets()
patch_model_card()

sys.argv = [SCRIPT] + sys.argv[1:]
runpy.run_path(SCRIPT, run_name="__main__")
