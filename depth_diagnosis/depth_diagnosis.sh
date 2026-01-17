#!/usr/bin/env bash

set -e

DATA_DIR="${1:-data/coco}"
SPATIAL_IDS="${2:-embeds/universal_id/llava-7b.pt}"
OUTPUT_DIR="${3:-embeds/depth_diagnosis}"

MULT_FACTOR=5
LAYERS="12,13,14"

mkdir -p "$OUTPUT_DIR"

python depth_diagnosis.py \
    --data_dir="$DATA_DIR" \
    --output_path="$OUTPUT_DIR/depth_diagnosis" \
    --spatial_ids_path="$SPATIAL_IDS" \
    --direction=height \
    --mult_factor="$MULT_FACTOR" \
    --layers="$LAYERS"

python depth_diagnosis.py \
    --data_dir="$DATA_DIR" \
    --output_path="$OUTPUT_DIR/depth_diagnosis" \
    --spatial_ids_path="$SPATIAL_IDS" \
    --direction=depth \
    --mult_factor="$MULT_FACTOR" \
    --layers="$LAYERS"
