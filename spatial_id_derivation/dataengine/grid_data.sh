#!/usr/bin/env bash

set -e

OUTPUT_BASE="${1:-data/grid_data}"

A_PATH="data/objects/camel.png"
B_PATH="data/objects/frog.png"

mkdir -p "$OUTPUT_BASE/camel_frog"

python dataengine/tile_two_objects.py \
    "$A_PATH" \
    "$B_PATH" \
    "$OUTPUT_BASE/camel_frog"