# Temporal Models

Extends spatial ID analysis to video VLMs, demonstrating that temporal IDs mediate temporal reasoning analogously to spatial IDs. Contains three experiments: mirror swapping (Figure 10A), temporal ID extraction (Figure 10B), and temporal steering (Figure 10C).

**Note:** This experiment requires a different environment. Use `temporal_models/environment.yml` instead of the root environment.

## Environment Setup

```bash
cd temporal_models
conda env create -f environment.yml
conda activate temporal-models
```

**Key differences from root environment:**
- `transformers==4.41.2` (specific version for video models)
- `decord` and `av` for video processing
- LLaVA-NeXT for video VLM support

## Data Requirements

**Prerequisites:**
- Synthetic Objaverse videos with temporal object appearances [here](https://drive.google.com/file/d/1p7LHGL243ARK-ACVrl63TFuJIFkm_ab0/view?usp=drive_link)
- MVBench Scene_QA [dataset](https://drive.google.com/file/d/1MNhGp7TFoO_D2g6NBEiP2TMWuIDniCXf/view?usp=drive_link)

**Input Structure:**
```
data/
├── objaverse_video/
│   ├── appear_circle3_triangle5.mp4
│   ├── appear_square2_circle6.mp4
│   └── ...                              # 8-frame videos with objects appearing at different times
└── mvbench/
    ├── mvbench_scene_queries.json      # MVBench Scene_QA annotations
    └── video/
        ├── video001.mp4
        ├── video002.mp4
        └── ...                          # MVBench Scene_QA videos
```

## Steps to Reproduce Figure 10

1. Setup temporal environment: `conda env create -f environment.yml && conda activate temporal-models`
2. Download synthetic videos to `data/objaverse_video/`
3. Download MVBench Scene_QA to `data/mvbench/`
4. Run all experiments: `bash temporal_models.sh`
5. Open `fig10.ipynb` in parent directory
6. Run all cells to generate Figure 10A, 10B, and 10C

## Running the Experiments

```bash
bash temporal_models.sh [SYNTHETIC_VIDEO_DIR] [MVBENCH_VIDEO_DIR] [EMBEDS_DIR]
```

**Arguments and Default Values:**
- `SYNTHETIC_VIDEO_DIR`: `data/objaverse_video` (synthetic temporal videos)
- `MVBENCH_VIDEO_DIR`: `data/mvbench` (MVBench Scene_QA dataset)
- `EMBEDS_DIR`: `embeds` (output directory)

The script runs three experiments in sequence:

### Experiment 1: Temporal ID Extraction (Figure 10B)

Extracts temporal IDs from synthetic videos where objects appear at different frames.

```bash
python extraction_pca.py \
    --video_dir data/objaverse_video \
    --embeds_dir embeds
```

**Hyperparameters (set in script):**
- Model: LLaVA-Video-7B-Qwen2
- Number of frames: 8 per video
- Query format: "In this video there are two scenes that occur in different order. Does the scene '{object1}' occur before or after the scene '{object2}'?"

### Experiment 2: Mirror Swapping (Figure 10A)

Validates temporal information flow by swapping activations between temporally reversed videos.

```bash
python mirror_swapping.py \
    --manifest_path data/mvbench/mvbench_scene_queries.json \
    --video_dir data/mvbench/video \
    --output_path embeds/mirror_swapping/mirror_swapping_results.pt \
    --max_samples 200
```

**Hyperparameters (set in script):**
- Layer step: 2 (process every 2nd layer)
- Swap modes: all text tokens, all video patches, object words only

### Experiment 3: Temporal Steering (Figure 10C)

Tests controllability by steering model beliefs with extracted temporal IDs.

```bash
python temporal_steering.py \
    --manifest_path data/mvbench/mvbench_scene_queries.json \
    --video_dir data/mvbench/video \
    --temporal_ids_path embeds/temporal_ids.pt \
    --output_dir embeds/temporal_steering
```

**Hyperparameters (set in script):**
- Steering layers: 13, 14, 15 (middle layers)
- Multiplication factor: 5.0 (steering magnitude)

## Output Structure

```
embeds/
├── temporal_ids.pt                     # Extracted temporal IDs (Experiment 1)
├── text_embeddings.pt                  # "before"/"after" text embeddings
├── mirror_swapping/
│   └── mirror_swapping_results.pt      # Mirror swap results by layer (Experiment 2)
└── temporal_steering/
    ├── temporal_steering_results.pt    # Per-video steering results (Experiment 3)
    └── temporal_steering_summary.pt    # Aggregated statistics
```

**Format:**
- Temporal IDs: `{layer: {frame_idx: vector}}`
- Mirror swapping: `{layer: {swap_mode: belief_shift}}`
- Steering results: `{video_id: {layer: {temporal_id_frame: Δlog P}}}`
