#!/usr/bin/env bash

set -e

DATA_DIR="${1:-data}"
UNIVERSAL_ID_PATH="${2:-embeds/universal_llava.pt}"
OUTPUT_DIR="${3:-embeds}"

MULT_FACTOR=5

mkdir -p "${OUTPUT_DIR}/batch_lr"
python batch_lr.py \
    --data_dir "$DATA_DIR/coco" \
    --save_output_path "${OUTPUT_DIR}/batch_lr" \
    --mult_factor "$MULT_FACTOR" \
    --universal_id_path "$UNIVERSAL_ID_PATH"

mkdir -p "${OUTPUT_DIR}/batch_nearfar"
python batch_nearfar.py \
    --data_dir "$DATA_DIR/objaverse_nearfar" \
    --save_output_path "${OUTPUT_DIR}/batch_nearfar" \
    --mult_factor "$MULT_FACTOR" \
    --universal_id_path "$UNIVERSAL_ID_PATH"

mkdir -p "${OUTPUT_DIR}/batch_inbetween"
python batch_inbetween.py \
    --data_dir "$DATA_DIR/objaverse_inbetween" \
    --save_output_path "${OUTPUT_DIR}/batch_inbetween" \
    --mult_factor "$MULT_FACTOR" \
    --universal_id_path "$UNIVERSAL_ID_PATH"