# Spatial ID Derivation

Extracts universal spatial IDs from VLMs by averaging object-specific spatial representations across multiple object pairs. Contains two experiments: small-scale grid for visualization (Figure 5) and large-scale extraction for downstream experiments.

## Data Requirements

**Prerequisites:**
- Objaverse 3D object renders [here](https://drive.google.com/file/d/13OplCWjxE_aU_9F6p2Tq1ln8mmgoNmJ2/view?usp=drive_link)

**Input Structure:**
```
data/objects/
├── {object_1}.png
├── {object_2}.png
└── ...                              # Objaverse 3D model renders
```

## Experiment 1: Small-Scale Grid for Figure 5

Generates a small grid dataset (2 object pairs) and extracts spatial IDs for visualization purposes.

### Steps to Reproduce Figure 5

1. Download Objaverse renders to `data/objects/`
2. Generate grid data: `bash dataengine/grid_data.sh`
3. Extract spatial IDs from grid: `bash small_id_grid.sh`
4. Open `fig5.ipynb` in parent directory
5. Run all cells to generate Figure 5

### Running Small-Scale Experiment

```bash
# 1. Generate small grid data (2 object pairs, 4×4 grid)
bash dataengine/grid_data.sh [OUTPUT_DIR]

# 2. Extract spatial IDs for visualization
bash small_id_grid.sh [IMAGE_BASE] [OUTPUT_BASE]
```

**Arguments and Default Values:**
- `OUTPUT_DIR`: `data/grid_data` (generated grid images)
- `IMAGE_BASE`: `data/grid_data` (input for extraction)
- `OUTPUT_BASE`: `embeds/grid_id` (output spatial IDs)

**Hyperparameters (set in grid_data.sh):**
- Grid size: `4×4`
- Object sizes: `[224, 174, 124, 74]` pixels
- Number of object pairs: `2`

### Generated Data Structure

```
data/grid_data/
└── {object_a}_{object_b}/
    └── {object_a}_{object_b}_{i}_{j}_{size}.png    # 4×4 grid × 4 sizes = 64 images per pair
```

## Experiment 2: Large-Scale Extraction for Universal IDs

Generates a large extraction dataset (90 object pairs) and extracts universal spatial IDs used by all downstream experiments.

### Steps to Extract Universal IDs

1. Download Objaverse renders to `data/objects/`
2. Generate extraction data: `bash dataengine/extraction_data.sh`
3. Extract universal spatial IDs: `bash batch_extraction.sh`
4. Universal IDs are now available in `embeds/universal_id/` for other experiments

### Running Large-Scale Extraction

```bash
# 1. Generate large extraction data (90 object pairs)
bash dataengine/extraction_data.sh [OUTPUT_DIR]

# 2. Extract universal spatial IDs for all models
bash batch_extraction.sh [IMAGE_BASE] [OUTPUT_BASE]
```

**Arguments and Default Values:**
- `OUTPUT_DIR`: `data/extraction_data` (generated images)
- `IMAGE_BASE`: `data/extraction_data` (input for extraction)
- `OUTPUT_BASE`: `embeds/universal_id` (output spatial IDs)

**Hyperparameters (set in extraction_data.sh):**
- Grid size: `4×4`
- Object sizes: `[224, 174, 124, 74]` pixels
- Number of object pairs: `90`

The script extracts spatial IDs for all models: llava-7b, llava-13b, llama-11b, qwen-7b, qwen-3b, qwen2-2b, internvl-1b, internvl-2b, internvl-8b, internvl-14b, llava-cot, gemma-4b, gemma-12b

### Generated Data Structure

```
data/extraction_data/
├── {object_a}_{object_b}/
│   └── {object_a}_{object_b}_{i}_{j}_{size}.png
└── ...                              # 90 object pairs
```

## Output Structure

```
embeds/
├── grid_id/                         # Small-scale for Figure 5
│   └── {model}.pt
└── universal_id/                    # Large-scale for downstream experiments
    ├── {object_a}_{object_b}/
    │   └── {model}.pt               # Per-object-pair spatial IDs
    ├── {model}.pt                   # Universal spatial IDs (averaged across objects)
    ├── {model}_x.pt                 # Horizontal spatial axis
    └── {model}_y.pt                 # Vertical spatial axis
```