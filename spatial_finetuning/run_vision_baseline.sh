#!/usr/bin/env bash

# Model argument (default: qwen2-2b)
MODEL=${1:-"qwen2-2b"}

python3 train.py \
  --model_name="$MODEL" \
  --train_dir=data/train \
  --val_json=data/coco_qa_two_obj.json \
  --coco_val_dir=data/coco \
  --param_budget_frac=0.3 \
  --batch_size=15 \
  --selector="vision_mlp_lastk" \
  --eval_max_samples=500 \
  --bf16 \
  --log_every=5 \
  --eval_save_steps=200