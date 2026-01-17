import re
import numpy as np
from pycocotools.coco import COCO
import torch
from transformers import AutoTokenizer, AutoProcessor, LlavaForConditionalGeneration
import os
import json
import random
from PIL import Image
import argparse


def extract_subject_object(caption):
    """
    Given a caption like:
    "A photo of a dog to the left of a person"
    or
    "A photo of an potted plant to the right of an umbrella"

    Return: ("potted plant", "umbrella", "right")
    """
    # Updated regex:
    # - Allow "a" or "an" (case-insensitive)
    # - Capture 1–3 word noun phrases for subject and object (you can adjust upper bound as needed)
    # - Allow flexible whitespace
    lr_pattern = r"A photo of a[n]? ([a-z]+(?: [a-z]+){0,2}) to the (left|right) of a[n]? ([a-z]+(?: [a-z]+){0,2})"
    ab_pattern = r"A photo of a[n]? ([a-z]+(?: [a-z]+){0,2}) (above|below) a[n]? ([a-z]+(?: [a-z]+){0,2})"

    lr_match = re.match(lr_pattern, caption.strip(), re.IGNORECASE)
    ab_match = re.match(ab_pattern, caption.strip(), re.IGNORECASE)
    if lr_match:
        subject, direction, obj = lr_match.groups()
        return subject.lower(), obj.lower(), direction.lower()
    elif ab_match:
        subject, direction, obj = ab_match.groups()
        return subject.lower(), obj.lower(), direction.lower()
    else:
        raise ValueError(f"Caption format unexpected: {caption}")


def make_unified_query(caption1, caption2, return_dir=False):
    """
    Given two opposing captions, return the unified query string.
    Assumes the captions only differ in the spatial relation.
    """
    subj1, obj1, dir1 = extract_subject_object(caption1)
    subj2, obj2, dir2 = extract_subject_object(caption2)

    assert subj1 == subj2, "Subjects do not match"
    assert obj1 == obj2, "Objects do not match"

    if {dir1, dir2} == {"left", "right"}:
        query = f"Is the {subj1} to the left or right of the {obj1}?"
    elif {dir1, dir2} == {"above", "below"}:
        query = f"Is the {subj1} above or below the {obj1}?"
    else:
        raise Exception("Directions must be left and right or above and below")

    if return_dir:
        return query, subj1, obj1, dir1, dir2
    else:
        return query, subj1, obj1


import numpy as np
from pycocotools.coco import COCO


def load_coco_annotations(labels_path):
    return COCO(labels_path)


def get_bbox_for_subject(coco, image_id, subject):
    ann_ids = coco.getAnnIds(imgIds=image_id)
    anns = coco.loadAnns(ann_ids)
    for ann in anns:
        cat_name = coco.loadCats(ann["category_id"])[0]["name"]
        if subject.lower() in cat_name.lower():
            return ann["bbox"]  # [x, y, width, height]
    return None


def compute_spatial_id_from_bbox(bbox, image_width, image_height, spatial_id_dict):
    x, y, w, h = bbox
    cx, cy = x + w / 2, y + h / 2

    # Normalize center to [0, 1]
    nx, ny = cx / image_width, cy / image_height

    # Map to 4x4 grid indices
    fx, fy = nx * 4, ny * 4  # grid index as float
    x0, y0 = int(np.floor(fx)), int(np.floor(fy))
    x1, y1 = min(x0 + 1, 3), min(y0 + 1, 3)

    wx = fx - x0
    wy = fy - y0

    weights = {
        (x0, y0): (1 - wx) * (1 - wy),
        (x1, y0): wx * (1 - wy),
        (x0, y1): (1 - wx) * wy,
        (x1, y1): wx * wy,
    }

    # Initialize output tensor
    example_tensor = next(iter(spatial_id_dict.values()))
    spatial = torch.zeros_like(example_tensor)

    for (i, j), w in weights.items():
        key = (i, j, 224)
        if key in spatial_id_dict:
            spatial += w * spatial_id_dict[key]

    return spatial  # shape [1, N]


def get_subject_spatial_id(image_id, subject, coco, spatial_id_dict):
    img_info = coco.loadImgs(image_id)[0]
    width, height = img_info["width"], img_info["height"]

    bbox = get_bbox_for_subject(coco, image_id, subject)
    if bbox is None:
        raise ValueError(
            f"No bounding box found for subject '{subject}' in image ID {image_id}"
        )

    spatial = compute_spatial_id_from_bbox(bbox, width, height, spatial_id_dict)
    return bbox, spatial


if __name__ == "__main__":
# EXAMPLE USAGE:

    parser = argparse.ArgumentParser(
        description="Intervene on LLaVA token embeddings at specific layers."
    )
    parser.add_argument(
        "--data_dir",
        type=str,
        required=True,
        help="Path to the image file.",
    )

    args = parser.parse_args()
    data_dir = args.data_dir


    MODEL_NAME = "llava-hf/llava-1.5-7b-hf"
    DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
    DTYPE = torch.float16

    tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME, use_fast=True)
    torch.set_grad_enabled(False)

    model = LlavaForConditionalGeneration.from_pretrained(
        "/groups/perona/raphi/vlm_pretrained/llava-1.5-7b-hf",
        torch_dtype=torch.float16,
        low_cpu_mem_usage=True,
        device_map="auto",
    )
    processor = AutoProcessor.from_pretrained(MODEL_NAME)

    coco_json = os.path.join(data_dir, "coco_qa_two_obj.json")
    coco_json_data = json.load(open(coco_json))

    # for the sake of this experiment, we'll only use the left/right questions.
    coco_json_data_leftrightonly = []
    coco_json_data_leftright_tracker = []
    for i in range(len(coco_json_data)):
        if "to the left of " in coco_json_data[i][1]:

            coco_json_data_leftrightonly.append(coco_json_data[i])
            coco_json_data_leftright_tracker.append("left")
        elif "to the right of " in coco_json_data[i][1]:
            coco_json_data_leftrightonly.append(coco_json_data[i])
            coco_json_data_leftright_tracker.append("right")

    print("num images:", len(coco_json_data_leftrightonly))

    random_indices = random.sample(range(len(coco_json_data_leftrightonly)), 100)
    coco_json_data_leftrightonly = [coco_json_data_leftrightonly[i] for i in random_indices]
    coco_json_data_leftright_tracker = [
        coco_json_data_leftright_tracker[i] for i in random_indices
    ]

    main_img_dir = os.path.join(data_dir, "val2017")


    # load annotations here
    labels_path = "/content/instances_val2017.json"
    coco = load_coco_annotations(labels_path)

    # load universal spatial IDs here
    univ_embs = torch.load("/content/universal_spatial_IDs_partial_layers_incl_L19.pt")
    univ_embs_plain = dict()
    for l in univ_embs:
        univ_embs_plain[l] = dict()
        for k in univ_embs[l]:
            if type(k) != str:
                univ_embs_plain[l][k] = univ_embs[l][k]


    unified_query_dict = dict()
    dict_of_all_res = dict()

    for iii, (img_id, caption1, caption2) in enumerate(coco_json_data_leftrightonly):

        dict_of_all_res[(img_id, coco_json_data_leftright_tracker[iii])] = dict()

        image_path = os.path.join(main_img_dir, f"{img_id:012d}.jpg")

        unified_query, subj1, obj1 = make_unified_query(caption1, caption2)
        unified_query_dict[img_id] = unified_query

        text_prompt = "QUESTION: " + unified_query + "Answer left or right. ANSWER: "

        control_img = Image.open(image_path)

        # CODE FOR GETTING SPATIAL IDs FOR SUBJECT
        bbox, spatial_id = get_subject_spatial_id(img_id, subj1, coco, univ_embs_plain[13])

        DEVICE = next(model.parameters()).device
        prompt = f"<image>\n{text_prompt}"
        inputs = tokenizer(prompt, return_tensors="pt").to(DEVICE)
        tokens = tokenizer.convert_ids_to_tokens(inputs.input_ids[0])

        control_inputs = processor(text=prompt, images=control_img, return_tensors="pt").to(
            DEVICE
        )
