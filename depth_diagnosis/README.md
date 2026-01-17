# Depth Diagnosis

Analyzes depth representation in VLMs by testing whether vertical spatial IDs (height) incorrectly drive depth-related beliefs.

## Data Requirements

**Prerequisites:**
- Extracted universal spatial IDs from `spatial_id_derivation` experiment (copy from `../spatial_id_derivation/embeds/universal_id/llava-7b.pt`)
- COCO-Spatial [dataset](https://drive.google.com/file/d/1efplEVOBN-Pas2nXkFPaZeLPHJPflyRq/view?usp=drive_link)

**Input Structure:**
```
data/coco/
├── 000000000139.jpg
├── 000000000285.jpg
├── ...                              # COCO images
└── coco_qa_two_obj.json            # COCO-Spatial annotations

embeds/universal_id/
└── llava-7b.pt                   # Extracted spatial IDs (copy from spatial_id_derivation)
```

## Steps to Reproduce Figure 7

1. Ensure spatial IDs extracted and compied into `embeds/universal_id/`
2. Download COCO-Spatial dataset to `data/coco/`
3. Run depth diagnosis: `bash depth_diagnosis.sh`
4. Open `fig7.ipynb` in parent directory
5. Run all cells to generate Figure 7

## Running the Experiment

```bash
bash depth_diagnosis.sh [DATA_DIR] [SPATIAL_IDS_PATH] [OUTPUT_DIR]
```

**Arguments and Default Values:**
- `DATA_DIR`: `data` (COCO images directory)
- `SPATIAL_IDS_PATH`: `embeds/universal_id/llava-7b.pt` (path to extracted spatial IDs)
- `OUTPUT_DIR`: `embeds/depth_diagnosis` (output directory)

**Hyperparameters (set as constants in script):**
- `MULT_FACTOR`: `5` (steering magnitude)
- `LAYERS`: `12,13,14` (layers to intervene on)

The script runs two experiments:
1. **Height**: Steer with spatial IDs varying in y-dimension, measure effect on "above"/"below"
2. **Depth**: Steer with same IDs, measure effect on "front"/"behind"

## Output Structure

```
embeds/depth_diagnosis/
├── depth_diagnosis_results_height.pt       # Height steering results (above/below)
├── depth_diagnosis_queries_height.pt       # Queries used for height
├── depth_diagnosis_results_depth.pt        # Depth steering results (front/behind)
└── depth_diagnosis_queries_depth.pt        # Queries used for depth
```

Each `.pt` file contains:
```python
{(image_id, ground_truth_label): {layer: {(i, j, size): (Δlog P(word1), Δlog P(word2))}}}
```

Where `(i, j, size)` is the spatial ID coordinate used for steering (e.g., (0, 0, 224)).
wy