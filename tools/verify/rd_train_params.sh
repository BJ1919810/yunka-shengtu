D=/root/autodl-tmp/lora-train/runs/1014-021438
if [ -f "$D/job.sh" ]; then
  echo "### job.sh 训练参数"
  grep -nE -- "^\s+--(rank|lora_alpha|lr_scheduler|learning_rate|lr_warmup_steps|max_train_steps|resolution|optimizer|train_batch_size|gradient_accumulation_steps|mixed_precision|seed|instance_prompt)" "$D/job.sh"
else
  echo "(job.sh 已清理)"
fi
echo "### 日志里的 scheduler / warmup 佐证"
grep -oE "lr_scheduler[a-z_]*|cosine[a-z_]*|warmup[_a-z]*" "$D/train.log" 2>/dev/null | sort | uniq -c | head
