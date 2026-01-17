# Ground Truth Deviation

Diagnoses VLM failures by analyzing spatial ID correctness and vision encoder performance.

## Data Requirements

**Prerequisites:**
- Extracted universal spatial IDs from `spatial_id_derivation` experiment
- COCO-Spatial [dataset](https://drive.google.com/file/d/1efplEVOBN-Pas2nXkFPaZeLPHJPflyRq/view?usp=drive_link)

**Input Structure:**
```
data/
├── coco/
│   ├── 000000000139.jpg
│   ├── 000000000285.jpg
│   ├── ...                              # COCO images and annotations
└── coco_qa_two_obj.json                # COCO-Spatial annotations
```

## Steps to Reproduce Figure 8

1. Download COCO-Spatial dataset to `data/`
2. Run spatial ID deviation: `bash coco_deviation.sh`
3. Run blur sensitivity: `bash coco_blur.sh`
4. Open `fig_8a.ipynb` and `fig_8b.ipynb` in parent directory
5. Run cells to generate Figure 8A and 8B

## Running the Experiments

### Figure 8A: Spatial ID Deviation Analysis

Extract spatial IDs from COCO images and measure deviation from ground truth:

```bash
bash coco_deviation.sh [COCO_IMG_DIR] [COCO_QA_JSON] [OUTPUT_BASE]
```

**Arguments and Default Values:**
- `COCO_IMG_DIR`: `data/coco` (COCO images directory)
- `COCO_QA_JSON`: `data/coco_qa_two_obj.json` (COCO-Spatial annotations)
- `OUTPUT_BASE`: `embeds/coco_deviation` (output directory)

Analyzes LLaMA-11B and LLaVA-7B models.

### Figure 8B: Image Masking (D-RISE) Analysis

Test model sensitivity to blurring object bboxes vs random locations:

```bash
bash coco_blur.sh [COCO_IMG_ROOT] [COCO_ANN] [CAPTIONS_JSON] [OUTPUT_DIR]
```

**Arguments and Default Values:**
- `COCO_IMG_ROOT`: `data/coco` (COCO images directory)
- `COCO_ANN`: `data/coco/annotations/instances_val2017.json` (COCO bounding boxes)
- `CAPTIONS_JSON`: `data/coco_qa_two_obj.json` (COCO-Spatial annotations)
- `OUTPUT_DIR`: `embeds/blur` (output directory)

**Hyperparameters (set as constants in script):**
- `BLUR_RADIUS`: `20` (Gaussian blur radius)

## Output Structure

```
embeds/
├── coco_deviation/
│   ├── llama-11b/
│   │   └── {object_a}_{object_b}_{pos}_{neg}_{id}/
│   │       ├── embeds.pt
│   │       ├── logits.pt
│   │       ├── sequences.pt
│   │       └── text.pt
│   └── llava-7b/
│       └── ...
└── blur/
    ├── llava_blur.pt
    └── llama_blur.pt
```

**Deviation format:** Model's extracted spatial IDs and ground truth labels per image
**Blur format:** Log probability changes when masking object bbox vs. random regions
