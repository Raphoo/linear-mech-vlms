#!/usr/bin/env bash

set -e

OUTPUT_BASE="${1:-data/extraction_data}"
objects=( backpack fan camel frog dog cake log robot person teapot)

for a in "${objects[@]}"; do
  for b in "${objects[@]}"; do
    for size in 74 124 174 224; do
      if [[ "$a" == "$b" ]]; then
        continue
      fi

      A_PATH="data/objects/${a}.png"
      B_PATH="data/objects/${b}.png"
      OUT_DIR="${OUTPUT_BASE}/${a}_${b}"

      python dataengine/tile_two_objects.py \
        "$A_PATH" \
        "$B_PATH" \
        "$OUT_DIR" \
        --object_wh="$size"
    done
  done
done