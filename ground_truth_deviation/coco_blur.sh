#!/usr/bin/env bash

set -e

COCO_IMG_ROOT="${1:-data/coco}"
COCO_ANN="${2:-data/coco/annotations/instances_val2017.json}"
CAPTIONS_JSON="${3:-data/coco/coco_qa_two_obj.json}"
OUTPUT_DIR="${4:-embeds/blur}"

BLUR_RADIUS=20
MODELS=("llava" "llama")

mkdir -p "$OUTPUT_DIR"

for model in "${MODELS[@]}"; do
    python coco_blur.py \
        --captions_json "$CAPTIONS_JSON" \
        --coco_ann "$COCO_ANN" \
        --coco_img_root "$COCO_IMG_ROOT" \
        --output "${OUTPUT_DIR}/${model}_blur.pt" \
        --model_key "$model" \
        --blur_radius "$BLUR_RADIUS"
done