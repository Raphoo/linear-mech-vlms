# Arbitrary Spatial Steering

Tests spatial ID steering on real COCO images for various spatial relationships (left/right, near/far, in-between). Contains two experiments: single-image steering (Figure 6A) and batch steering (Figures 6B, 6C, 6D).

## Data Requirements

**Prerequisites:**
- Extracted universal spatial IDs. We used a slightly different Spatial ID extracted with background for this experiment. Linked [here](https://drive.google.com/file/d/1oxGCDUbYjlUqoEqbbOWVqK8OpcvSf5Um/view?usp=drive_link).
- COCO-Spatial [dataset](https://drive.google.com/file/d/1efplEVOBN-Pas2nXkFPaZeLPHJPflyRq/view?usp=sharing)
- Synthetic Objaverse data for [near/far](https://drive.google.com/file/d/1y3F4IoSQQDBgjIptrgcbFWE2_TLQtMXI/view?usp=drive_link) and (in-between)[https://drive.google.com/file/d/1vYBE9NXEZANh_EO2U5yDoTHaR6CR9Ny6/view?usp=drive_link] tests.
- A single image in `data/val1.jpg` linked [here](https://drive.google.com/file/d/1ZHfGdI8MMXuTrr4T1dx40f9McGzvwVTG/view?usp=drive_link).

**Input Structure:**
```
data/
├── val1.jpg                         # Single COCO image for Figure 6A
├── coco/
│   ├── 000000000139.jpg
│   ├── 000000000285.jpg
│   ├── ...                          # COCO images
│   └── coco_qa_two_obj.json        # COCO-Spatial annotations
├── objaverse_nearfar/              # Synthetic near/far images
└── objaverse_inbetween/            # Synthetic in-between images

embeds/universal_llava.pt
```

## Experiment 1: Single Image Steering for Figure 6A

Demonstrates steering effects on a single image across different layers for visualization.

### Steps to Reproduce Figure 6A

1. Download/copy universal spatial IDs with background to `embeds`
2. Ensure single image exists at `data/val1.jpg`
3. Run single steering: `bash single_steering.sh`
4. Open `fig6a.ipynb` in parent directory
5. Run all cells to generate Figure 6A

### Running Single Image Experiment

```bash
bash single_steering.sh [IMAGE_PATH] [UNIVERSAL_ID_PATH] [OUTPUT_DIR]
```

**Arguments and Default Values:**
- `IMAGE_PATH`: `data/val1.jpg` (single COCO image)
- `UNIVERSAL_ID_PATH`: `embeds/universal_id/llava-7b.pt` (extracted spatial IDs)
- `OUTPUT_DIR`: `embeds/lr_steering_single` (output directory)

**Hyperparameters (set as constants in script):**
- `MULT_FACTOR`: `5` (steering magnitude)

### Output Structure

```
embeds/lr_steering_single/
├── image_intervention_appendbind_to_noise_loc_3-1_0-1_thisthat_universal_50x_BOTTLE_double_fixed_moved.csv            # Steering left across layers
└── image_intervention_appendbind_to_noise_loc_0-1_3-1_thisthat_universal_50x_BOTTLE_double_fixed_moved.csv           # Steering right across layers
```

Please note that the progam is sensitive to the output path. It uses the path to determine some settings.

## Experiment 2: Batch Steering for Figures 6B, 6C, 6D

Runs steering on multiple images to test generalization across spatial relationships.

### Steps to Reproduce Figures 6B, 6C, 6D

1. Download/copy universal spatial IDs with background to `embeds/universal_llava.pt/`
2. Download COCO-Spatial dataset to `data/coco/`
3. Download/generate synthetic data to `data/objaverse_nearfar/` and `data/objaverse_inbetween/` 
4. Run batch steering: `bash batch_steering.sh`
5. Open `fig6_bcd.ipynb` in parent directory
6. Run all cells to generate Figures 6B, 6C, and 6D

### Running Batch Experiment

```bash
bash batch_steering.sh [DATA_DIR] [UNIVERSAL_ID_PATH] [OUTPUT_DIR]
```

**Arguments and Default Values:**
- `DATA_DIR`: `data` (contains coco/, objaverse_nearfar/, objaverse_inbetween/)
- `UNIVERSAL_ID_PATH`: `embeds/universal_llava.pt` (extracted spatial IDs)
- `OUTPUT_DIR`: `embeds` (output directory)

**Hyperparameters (set as constants in script):**
- `MULT_FACTOR`: `5` (steering magnitude)

### Steering Types

The script runs three types of spatial steering:

1. **Left/Right (`batch_lr.py`)**: Steers objects left or right on COCO images
2. **Near/Far (`batch_nearfar.py`)**: Steers relative distance between objects on synthetic data
3. **In-between (`batch_inbetween.py`)**: Tests if object is sandwiched between two others (commented out in script)

### Output Structure

```
embeds/
├── batch_lr/                    # left/right steering results (Figure 6B)
├── batch_nearfar/               # near/far steering results (Figure 6C)
└── batch_inbetween/             # inbetween steering results
```

Each subdirectory contains `.pt` files with log probability changes for spatial relationships.
