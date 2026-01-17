# Mirror and Attribute Swapping

Tests information flow during spatial reasoning by swapping activations between mirrored or attribute-modified images at different layers. This experiment identifies where spatial information transfers from image patches to text tokens.

## Data Requirements

**Prerequisites:**
- COCO-Spatial [dataset](https://drive.google.com/file/d/1efplEVOBN-Pas2nXkFPaZeLPHJPflyRq/view?usp=drive_link)

**Input Structure:**
```
data/
├── coco/
│   ├── 000000000139.jpg
│   ├── 000000000285.jpg
│   └── ...                          # COCO images
└── coco_qa_two_obj.json            # COCO-Spatial annotations
```

## Steps to Reproduce Figure 4

1. Download COCO-Spatial dataset to `data/`
2. Run swapping experiments: `bash mirror_attr_swapping.sh`
3. Open `fig4.ipynb` in parent directory
4. Run all cells to generate Figure 4

## Running the Experiment

```bash
bash mirror_attr_swapping.sh [DATA_DIR] [OUTPUT_DIR]
```

**Arguments and Default Values:**
- `DATA_DIR`: `data/coco`
- `OUTPUT_DIR`: `embeds/mirror_swapping`

The script runs two experiments:
1. **Mirror Swapping**: Swap activations between original and horizontally-flipped images
2. **Attribute Swapping**: Swap activations between original and color-modified images (control)

## Output Structure

```
embeds/mirror_swapping/
├── mirror_swap_objwords_dict_of_all_res.pt
├── mirror_swap_objwords_unified_query_dict.pt
├── attribute_swap_objwords_dict_of_all_res.pt
└── attribute_swap_objwords_unified_query_dict.pt
```

Each `.pt` file contains:
```python
{(img_id, direction): {layer: {intervention_type: (Δlog P(left), Δlog P(right))}}}
```

Where `intervention_type` is one of: `all_text`, `all_img`, `obj_words`
