#!/usr/bin/env bash

set -e

IMAGE_PATH="${1:-data/val1.jpg}"
UNIVERSAL_ID_PATH="${2:-embeds/universal_llava.pt}"
OUTPUT_DIR="${3:-embeds/single_lr}"

MULT_FACTOR=5

mkdir -p "$OUTPUT_DIR"

python single_lr.py \
  --image_path "$IMAGE_PATH" \
  --query "QUESTION: Is the potted plant to the left or right of the bottle? Answer left or right. ANSWER: " \
  --patch_type "<text>" \
  --mult_factor "$MULT_FACTOR" \
  --universal_id_path "$UNIVERSAL_ID_PATH" \
  --save_output_path "${OUTPUT_DIR}/image_intervention_appendbind_to_noise_loc_0-1_3-1_thisthat_universal_50x_BOTTLE_double_fixed_moved.csv"

python single_lr.py \
  --image_path "$IMAGE_PATH" \
  --query "QUESTION: Is the potted plant to the left or right of the bottle? Answer left or right. ANSWER: " \
  --patch_type "<text>" \
  --mult_factor "$MULT_FACTOR" \
  --universal_id_path "$UNIVERSAL_ID_PATH" \
  --save_output_path "${OUTPUT_DIR}/image_intervention_appendbind_to_noise_loc_3-1_0-1_thisthat_universal_50x_BOTTLE_double_fixed_moved.csv"