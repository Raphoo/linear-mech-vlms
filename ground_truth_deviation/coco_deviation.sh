#!/usr/bin/env bash

set -e

COCO_IMG_DIR="${1:-data/coco}"
COCO_QA_JSON="${2:-data/coco/coco_qa_two_obj.json}"
OUTPUT_BASE="${3:-embeds/coco_deviation}"

MODELS=("llama-11b" "llava-7b")

for model in "${MODELS[@]}"; do
    out_dir="${OUTPUT_BASE}/${model}"
    python coco_deviation.py \
        "$COCO_IMG_DIR" \
        "$out_dir" \
        "$COCO_QA_JSON" \
        --model="$model"
done
