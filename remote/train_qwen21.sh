#!/usr/bin/env bash
# =============================================================================
# Qwen-Image-2.1 画风 LoRA 训练 —— 云卡启动模板（32G 4080S）
#
# 用法：
#   DATA=/root/autodl-tmp/dataset OUT=/root/autodl-tmp/lora-out/bj_style \
#       bash /root/autodl-tmp/lora-train/train_qwen21.sh
#
# 前置（都已就绪）：
#   /root/autodl-tmp/train-env                隔离 venv（复用 base 的 torch 2.9.1+cu130）
#   /root/autodl-tmp/lora-train/              官方脚本 + run_train_qwen21.py + qwen21_loader.py
#   /root/autodl-tmp/Qwen-Image-2.1/          伪 HF 目录（config/processor/scheduler + HF 版 VAE）
#   /root/autodl-tmp/ComfyUI/models/…         DiT 与文本编码器（由 qwen21_loader 直接读，不复制）
#
# ⚠️ 必须用**有卡模式**跑；无卡模式没有 CUDA。
# ⚠️ 跑之前先 `python cloud_queue.py`（云卡与本机共用同一队列）。
# =============================================================================
set -euo pipefail

ENV=/root/autodl-tmp/train-env
TRAIN=/root/autodl-tmp/lora-train
MODEL=/root/autodl-tmp/Qwen-Image-2.1

DATA=${DATA:-/root/autodl-tmp/dataset}                 # 数据集目录（图片 + 同名 .txt）
OUT=${OUT:-/root/autodl-tmp/lora-out/bj_style}
TRIG=${TRIG:-bj_style}                                 # 触发词
STEPS=${STEPS:-800}

# 系统盘只剩 ~10G ⇒ 所有缓存强制落到数据盘，否则装到一半磁盘满
export PIP_CACHE_DIR=/root/autodl-tmp/.pipcache
export HF_HOME=/root/autodl-tmp/hf
export HF_DATASETS_CACHE=/root/autodl-tmp/hf/datasets
export TMPDIR=/root/autodl-tmp/tmp
export PYTORCH_ALLOC_CONF=expandable_segments:True

mkdir -p "$OUT" "$HF_HOME" "$TMPDIR"

# ── 1) 图片 + 同名 .txt → metadata.jsonl（JSONL 格式，见 make_metadata.py 里的两个坑）──
"$ENV/bin/python" "$TRAIN/make_metadata.py" "$DATA"

# ── 2) 训练 ──────────────────────────────────────────────────────────────────
cd "$TRAIN"
"$ENV/bin/python" run_train_qwen21.py \
  --pretrained_model_name_or_path "$MODEL" \
  --dataset_name "$DATA" \
  --image_column image \
  --caption_column text \
  --instance_prompt "$TRIG" \
  --output_dir "$OUT" \
  --resolution 1024 \
  --train_batch_size 1 \
  --gradient_accumulation_steps 4 \
  --rank 16 \
  --lora_alpha 16 \
  --lora_dropout 0.05 \
  --learning_rate 1e-4 \
  --lr_scheduler constant \
  --lr_warmup_steps 0 \
  --max_train_steps "$STEPS" \
  --checkpointing_steps 200 \
  --checkpoints_total_limit 2 \
  --caption_dropout 0.1 \
  --gradient_checkpointing \
  --use_8bit_adam \
  --cache_latents \
  --offload \
  --mixed_precision bf16 \
  --seed 42 \
  --report_to tensorboard \
  "$@"

# ── 显存怎么省（先后顺序很重要）──────────────────────────────────────────────
#   ① `--offload`（已开）★ 官方参数，含义＝「VAE 和文本编码器不用时挪到 CPU」
#        ⇒ 峰值 ≈ max(文本编码器 17.5G, DiT 13.25G)，32G 卡**不用量化底模**就够。
#        ⚠️ 量化会稍微伤 LoRA 质量，能不用就不用 —— 所以这是首选。
#   ② 还 OOM 再加：--bnb_quantization_config_path "$TRAIN/bnb_nf4.json"（DiT 13.25G → 约 7G）
#   ③ 再不行：--resolution 768 / --gradient_accumulation_steps 8（保持等价 batch、降峰值）
#
# ── 想每 N 步看图（会额外吃显存，先别开）────────────────────────────────────
#   --validation_prompt "$TRIG, 1名少女站在教室里看向镜头" \
#   --num_validation_images 2 \
#   --validation_epochs 50 \
