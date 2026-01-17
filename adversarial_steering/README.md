# Adversarial Steering

Tests steerability of spatial IDs by adversarially intervening on object representations to flip spatial predictions. Demonstrates causal role of spatial IDs across multiple VLMs.

## Data Requirements

**Prerequisites:**
- Extracted universal spatial IDs from `spatial_id_derivation` experiment (copy to `embeds/universal_id/`)
- COCO-Spatial [dataset](https://drive.google.com/file/d/1efplEVOBN-Pas2nXkFPaZeLPHJPflyRq/view?usp=sharing)

**Input Structure:**
```
data/coco/
├── 000000000139.jpg
├── 000000000285.jpg
├── ...                              # COCO images
└── coco_qa_two_obj.json            # COCO-Spatial annotations

embeds/universal_id/
├── llava-7b.pt                     # Copy from spatial_id_derivation
├── llama-11b.pt
├── qwen-7b.pt
└── ...                              # All model spatial IDs
```

## Steps to Reproduce Figures 2 & 9

1. Ensure spatial IDs extracted
2. Copy spatial IDs: `cp -r ../spatial_id_derivation/embeds/universal_id/* embeds/universal_id/`
3. Download COCO-Spatial dataset to `data/coco/`
4. Run steering experiments: `bash adversarial_steering.sh`
5. Open `fig2.ipynb` and `fig9.ipynb` in parent directory and run all cells

We also included our results metadata [here](https://drive.google.com/file/d/1hxeVP3ejePN0BXM8oVOACg62BiD7WVBE/view?usp=drive_link) for reproducibility.

## Running the Experiment

```bash
bash adversarial_steering.sh [COCO_IMG_DIR] [COCO_QA_JSON] [UNIVERSAL_ID_DIR] [OUTPUT_BASE] [METADATA_DIR]
```

**Arguments and Default Values:**
- `COCO_IMG_DIR`: `data/coco` (COCO images directory)
- `COCO_QA_JSON`: `data/coco/coco_qa_two_obj.json` (COCO-Spatial annotations)
- `UNIVERSAL_ID_DIR`: `embeds/universal_id` (extracted spatial IDs directory)
- `OUTPUT_BASE`: `embeds/adversarial_steering` (output directory)
- `METADATA_DIR`: `metadata` (accuracy and log-prob stats)

**Hyperparameters (set as constants in script):**
- `FACTOR`: `5` (intervention scaling factor)

The script runs interventions across 12 models with spatial IDs vs. noise control.

## Output Structure

```
embeds/adversarial_steering/{model}_{factor}/
└── {object_a}_{object_b}_{pos}_{neg}_{id}/
    ├── embeds.pt                          # baseline
    ├── embeds_intervention_{layer}.pt     # spatial ID intervention
    ├── embeds_noise_{layer}.pt            # noise control
    ├── logits*.pt
    ├── sequences*.pt
    └── text.pt

metadata/
└── {model}_{factor}.json                  # accuracy and log-prob stats
```

**Metadata format:** `{intervention_name: {image_id: {"verdict": "correct"|"wrong"|"nonsense", "log_prob": (pos, neg)}}}`
