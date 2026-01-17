# Spatial Finetuning

Finetunes VLMs with an auxiliary spatial ID loss module to improve spatial reasoning performance. Demonstrates that spatial IDs can serve as a learning signal for better generalization.

## Data Requirements

**Prerequisites:**
- Extracted universal spatial IDs from `spatial_id_derivation` experiment (copy to `embeds/universal_id_90/`)
- Synthetic Objaverse training data (TBD: This archive is very large, please email author)
- COCO-Spatial validation [dataset](https://drive.google.com/file/d/1efplEVOBN-Pas2nXkFPaZeLPHJPflyRq/view?usp=drive_link)

**Input Structure:**
```
data/
├── train/                              # Synthetic Objaverse paired grid images
│   ├── {object_a}_{object_b}_{i}_{j}_{size}.png
│   └── ...
├── coco/
│   ├── 000000000139.jpg
│   ├── 000000000285.jpg
│   └── ...                              # COCO validation images
└── coco_qa_two_obj.json                # COCO-Spatial annotations

embeds/universal_id_90/
├── qwen2-2b.pt                         # Copy from spatial_id_derivation
└── ...                                  # Other model spatial IDs
```

## Steps to Reproduce Results

1. Ensure spatial IDs extracted: `bash ../spatial_id_derivation/batch_extraction.sh`
2. Copy spatial IDs: `cp -r ../spatial_id_derivation/embeds/universal_id/* embeds/universal_id_90/`
3. Download training data to `data/train/`
4. Download COCO-Spatial dataset to `data/coco/`
5. Run baseline training: `bash run_vision_baseline.sh [MODEL]`
6. Run training with spatial loss: `bash run_vision_spatial_loss.sh [MODEL]`
7. Compare validation accuracies on COCO-Spatial

## Running the Experiments

### Baseline Training (Control)

```bash
bash run_vision_baseline.sh [MODEL]
```

**Arguments and Default Values:**
- `MODEL`: `qwen2-2b` (model to finetune)

### Training with Spatial ID Loss

```bash
bash run_vision_spatial_loss.sh [MODEL]
```

**Arguments and Default Values:**
- `MODEL`: `qwen2-2b` (model to finetune)

**Hyperparameters (set in scripts):**
- `param_budget_frac`: `0.3` (fraction of parameters to train)
- `batch_size`: `15` (training batch size)
- `selector`: `vision_mlp_lastk` (which layers to train - last 6 vision encoder MLPs)
- `eval_save_steps`: `200` (evaluation frequency)
- `spatial_id_weight`: `1` (weight for spatial ID loss, only in spatial_loss script)

