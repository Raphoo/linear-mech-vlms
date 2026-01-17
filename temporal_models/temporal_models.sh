#!/usr/bin/env bash

set -e

SYNTHETIC_VIDEO_DIR="${1:-data/objaverse_video}"
MVBENCH_VIDEO_DIR="${2:-data/mvbench}"
EMBEDS_DIR="${3:-embeds}"

mkdir -p "$EMBEDS_DIR"
mkdir -p "$EMBEDS_DIR/mirror_swapping"
mkdir -p "$EMBEDS_DIR/temporal_steering"

# # Experiment 10b: Temporal ID Extraction
python extraction_pca.py \
    --video_dir "$SYNTHETIC_VIDEO_DIR" \
    --embeds_dir "$EMBEDS_DIR"

# Experiment 10a: Mirror Swapping
python mirror_swapping.py \
    --manifest_path "$MVBENCH_VIDEO_DIR/mvbench_scene_queries.json" \
    --video_dir "$MVBENCH_VIDEO_DIR/video" \
    --output_path "$EMBEDS_DIR/mirror_swapping/mirror_swapping_results.pt" \
    --max_samples 200

# Experiment 10c: Temporal Steering
python temporal_steering.py \
    --manifest_path "$MVBENCH_VIDEO_DIR/mvbench_scene_queries.json" \
    --video_dir "$MVBENCH_VIDEO_DIR/video" \
    --temporal_ids_path "$EMBEDS_DIR/temporal_ids.pt" \
    --output_dir "$EMBEDS_DIR/temporal_steering"
