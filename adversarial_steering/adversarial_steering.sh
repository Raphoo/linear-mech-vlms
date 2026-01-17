#!/usr/bin/env bash

set -e

COCO_IMG_DIR="${1:-data/coco}"
COCO_QA_JSON="${2:-data/coco/coco_qa_two_obj.json}"
UNIVERSAL_ID_DIR="${3:-embeds/universal_id}"
OUTPUT_BASE="${4:-embeds/adversarial_steering}"
METADATA_DIR="${5:-metadata}"

CONFIG=(
  "qwen2-2b:6:16"
  "qwen-3b:8:18"
  "qwen-7b:6:16"
  "llava-7b:8:20"
  "llava-13b:10:20"
  "llama-11b:10:20"
  "internvl-1b:6:16"
  "internvl-2b:6:16"
  "internvl-8b:6:16"
  "internvl-14b:12:24"
  "gemma-4b:10:22"
  "gemma-12b:10:22"
)

FACTOR=5

for item in "${CONFIG[@]}"; do
    IFS=: read -r model start end <<< "$item"
    out_dir="${OUTPUT_BASE}/${model}_${FACTOR}"
    mkdir -p "$out_dir"

    python adversarial_steering.py \
        "$COCO_IMG_DIR" \
        "$out_dir" \
        "$COCO_QA_JSON" \
        --model="$model"

    for i in $(seq "$start" "$end"); do
        python adversarial_steering.py \
            "$COCO_IMG_DIR" \
            "$out_dir" \
            "$COCO_QA_JSON" \
            --model="$model" \
            --intervention_layer="$i" \
            --intervention_factor="$FACTOR" \
            --intervention_embed="${UNIVERSAL_ID_DIR}/${model}.pt" \
            --control_dir="$out_dir"

        python adversarial_steering.py \
            "$COCO_IMG_DIR" \
            "$out_dir" \
            "$COCO_QA_JSON" \
            --model="$model" \
            --intervention_layer="$i" \
            --intervention_factor="$FACTOR" \
            --intervention_embed="${UNIVERSAL_ID_DIR}/${model}.pt" \
            --control_dir="$out_dir" \
            --use_noise
    done

    mkdir -p "$METADATA_DIR"
    python compute_coco_stats.py "$out_dir" "$model" "${METADATA_DIR}/${model}_${FACTOR}.json"
done
