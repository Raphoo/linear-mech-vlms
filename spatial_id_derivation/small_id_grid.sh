#!/usr/bin/env bash

set -e

IMAGE_BASE="${1:-data/grid_data}"
OUTPUT_BASE="${2:-embeds/id_grid}"

MODELS=("llava-7b" "llama-11b")

mkdir -p "$OUTPUT_BASE/camel_frog"
mkdir -p "$OUTPUT_BASE/camel_frog_text"

for model in "${MODELS[@]}"; do
  python compute_id.py \
    "$IMAGE_BASE/camel_frog" \
    "$OUTPUT_BASE/camel_frog" \
    "Is the frog to the left or right of the camel?" \
    "frog,left,right" \
    --model="$model"

  python compute_axes.py \
      "$OUTPUT_BASE/camel_frog/$model.pt" \
      "$OUTPUT_BASE/camel_frog/$model"

  python ../utils/extract_embeds.py \
    "Is the frog to the left or right of the camel?" \
    "left,right" \
    "$OUTPUT_BASE/camel_frog_text/$model.pt" \
    --model="$model"
done