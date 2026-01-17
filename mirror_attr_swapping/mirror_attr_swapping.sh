#!/usr/bin/env bash

set -e

DATA_DIR="${1:-data/coco}"
OUTPUT_DIR="${2:-embeds/mirror_swapping}"

mkdir -p "$OUTPUT_DIR"

python mirror_swapping.py \
    --data_dir "$DATA_DIR" \
    --save_output_path "$OUTPUT_DIR/mirror_swap_objwords"

python attribute_swapping.py \
    --data_dir "$DATA_DIR" \
    --save_output_path "$OUTPUT_DIR/attribute_swap_objwords"
