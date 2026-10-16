#!/usr/bin/env bash
# 云卡训练环境体检 —— 一趟 SSH 拿齐。
# 用法： python cloud_sh.py --file remote/probe_train_state.sh
# ⚠️ 只读，不改任何东西。
echo "### HOST"
hostname
echo "### MEMCGROUP"
cat /sys/fs/cgroup/memory.max 2>/dev/null || echo "n/a"
echo "### DISKFREE"
df -h /root/autodl-tmp 2>/dev/null | tail -1
echo "### DATADISK"
ls -A /root/autodl-tmp/ 2>/dev/null | grep -v '^autodl' || echo "(空)"
echo "### GPU"
nvidia-smi --query-gpu=name,memory.used,memory.total,driver_version --format=csv,noheader 2>/dev/null || echo "no-gpu"
echo "### TRAINENV"
ls -d /root/autodl-tmp/train-env /root/autodl-tmp/lora-train /root/autodl-tmp/Qwen-Image-2.1 2>/dev/null || echo "(缺失)"
echo "### LORATRAIN_FILES"
ls /root/autodl-tmp/lora-train/ 2>/dev/null || echo "(缺失)"
echo "### DATASET_DIR"
if [ -d /root/autodl-tmp/dataset ]; then
  echo "count=$(ls -A /root/autodl-tmp/dataset | wc -l)"
  ls -A /root/autodl-tmp/dataset | head -20
else
  echo "(目录不存在)"
fi
echo "### LORA_OUT"
ls -la /root/autodl-tmp/lora-out/ 2>/dev/null || echo "(无)"
echo "### MODELS"
ls /root/autodl-tmp/ComfyUI/models/diffusion_models/ 2>/dev/null | head -10
echo "-- text_encoders --"
ls /root/autodl-tmp/ComfyUI/models/text_encoders/ 2>/dev/null | head -10
echo "### PROCS"
ps aux | grep -E 'run_train_qwen21|train_dreambooth|main.py' | grep -v grep || echo "(无训练/ComfyUI 进程)"
echo "### PYVER"
/root/autodl-tmp/train-env/bin/python -c "import torch;print('torch',torch.__version__,'cuda_avail',torch.cuda.is_available())" 2>&1 | tail -2
/root/autodl-tmp/train-env/bin/python -c "import diffusers,transformers;print('diffusers',diffusers.__version__,'transformers',transformers.__version__)" 2>&1 | tail -2
echo "### DONE"
