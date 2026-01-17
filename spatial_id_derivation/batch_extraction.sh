#!/usr/bin/env bash

set -e

IMAGE_BASE="${1:-data/extraction_data}"
OUTPUT_BASE="${2:-embeds/universal_id}"

for model in "llava-7b" "llava-13b" "llama-11b" "qwen-7b" "qwen-3b" "qwen2-2b" "internvl-1b" "internvl-2b" "internvl-8b" "internvl-14b" "llava-cot" "gemma-4b" "gemma-12b"; do
  for subdir in "${IMAGE_BASE}"/*; do
    if [[ -d "${subdir}" ]]; then
      base="$(basename "${subdir}")"
      IFS='_' read -r object_a object_b <<< "${base}"
      outdir="${OUTPUT_BASE}/${object_a}_${object_b}"
      mkdir -p "${outdir}"

      query="Is ${object_b} to the left or right of ${object_a}?"
      target_words="${object_b}"

      python compute_id.py \
        "${subdir}" \
        "${outdir}" \
        "${query}" \
        "${target_words}" \
        --model="${model}"
    fi
  done
  python average_id.py \
    "${OUTPUT_BASE}" \
    "${OUTPUT_BASE}/${model}.pt" \
    --model="${model}"
  python compute_axes.py \
    "${OUTPUT_BASE}/${model}.pt" \
    "${OUTPUT_BASE}/${model}"
done