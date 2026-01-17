import torch
from transformers import (
    LlavaForConditionalGeneration,
    AutoTokenizer,
    CLIPProcessor,
    CLIPModel,
    AutoProcessor,
)
from PIL import Image
import einops
import matplotlib.pyplot as plt
import argparse
from transformers.models.llama.modeling_llama import LlamaRotaryEmbedding
from torch.nn.functional import cosine_similarity, softmax
import numpy as np
import re
import csv
import os
import math
import json
import random
from image_intervention_helper import (
    generate_non_overlapping_pairs,
    make_unified_query,
    log_probs_to_csv,
    make_noise_image,
    make_white_background,
)

random.seed(42)

from tqdm import tqdm

# Argument parser
parser = argparse.ArgumentParser(
    description="Intervene on LLaVA token embeddings at specific layers."
)
parser.add_argument(
    "--data_dir",
    type=str,
    required=True,
    help="Path to the image file.",
)

parser.add_argument(
    "--save_output_path",
    type=str,
    required=True,
    help="Path to save the output data to a csv.",
)


args = parser.parse_args()
data_dir = args.data_dir


MODEL_NAME = "llava-hf/llava-1.5-7b-hf"
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
DTYPE = torch.float16

# Load model and tokenizer
tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME, use_fast=True)
torch.set_grad_enabled(False)

model = LlavaForConditionalGeneration.from_pretrained(
    MODEL_NAME,
    torch_dtype=torch.float16,
    low_cpu_mem_usage=True,
    device_map="auto",
)
processor = AutoProcessor.from_pretrained(MODEL_NAME)

print(model)


def recolor_with_random_rainbow(
    control_img, strength=0.75, sat_boost=1.1, value_gain=1.0, seed=None
):
    """
    Recolors by mixing original hue with a random rainbow gradient; preserves
    value (brightness) to keep structure legible.

    strength  in [0,1]: 1 = full rainbow hue, 0 = no change
    sat_boost >= 0    : multiply saturation
    value_gain>= 0    : multiply value/brightness
    """
    if seed is not None:
        np.random.seed(seed)

    img = control_img.convert("RGB")
    w, h = img.size

    # --- random rainbow gradient in hue space ---
    theta = np.random.uniform(0, 2 * np.pi)  # orientation
    freq = np.random.uniform(0.8, 2.5)  # cycles across image
    phase = np.random.uniform(0.0, 1.0)  # hue phase
    xs = np.linspace(0, 1, w, dtype=np.float32)
    ys = np.linspace(0, 1, h, dtype=np.float32)
    X, Y = np.meshgrid(xs, ys)
    U = np.cos(theta) * X + np.sin(theta) * Y
    U = (U - U.min()) / max(1e-8, U.max() - U.min())
    hue_rainbow = (U * freq + phase) % 1.0  # [0,1) hue

    # --- mix hue; keep V (brightness) ---
    H, S, V = img.convert("HSV").split()
    H_np = np.asarray(H, dtype=np.float32) / 255.0
    S_np = np.asarray(S, dtype=np.float32) / 255.0
    V_np = np.asarray(V, dtype=np.float32) / 255.0

    H_new = (1.0 - strength) * H_np + strength * hue_rainbow
    S_new = np.clip(S_np * sat_boost, 0.0, 1.0)
    V_new = np.clip(V_np * value_gain, 0.0, 1.0)

    H8 = (np.clip(H_new, 0.0, 1.0) * 255).astype(np.uint8)
    S8 = (np.clip(S_new, 0.0, 1.0) * 255).astype(np.uint8)
    V8 = (np.clip(V_new, 0.0, 1.0) * 255).astype(np.uint8)

    return Image.merge(
        "HSV", [Image.fromarray(H8), Image.fromarray(S8), Image.fromarray(V8)]
    ).convert("RGB")


def cache_resid_post(storage_dict):
    def hook(module, inputs, outputs):
        # print("caching:", outputs)
        if isinstance(outputs, tuple):
            hidden_states = outputs[0]
        else:
            hidden_states = outputs
        storage_dict["resid_post"] = hidden_states.detach()
        return outputs

    return hook


def patch_resid_post(intervene_resid_post):
    def hook(module, inputs, outputs):
        # print("patching:", outputs)
        if isinstance(outputs, tuple):
            return (intervene_resid_post,)
        else:
            return intervene_resid_post

    return hook


def run_patch_resid_post_onlysubj(
    model,
    control_inputs,
    intervene_inputs,
    layer,
    target_pos_added_all,
):
    # Step 1. Get control resid_post
    control_cache = {}
    hook_control = model.language_model.layers[layer].register_forward_hook(
        cache_resid_post(control_cache)
    )
    with torch.no_grad():
        control_logits = model(**control_inputs, use_cache=False).logits
    hook_control.remove()

    # Step 2. Get intervene resid_post
    intervene_cache = {}
    hook_intervene = model.language_model.layers[layer].register_forward_hook(
        cache_resid_post(intervene_cache)
    )
    with torch.no_grad():
        intervene_logits = model(**intervene_inputs, use_cache=False).logits
    hook_intervene.remove()

    patched_cache = control_cache["resid_post"].clone()

    # )
    patched_cache[0, target_pos_added_all] = intervene_cache["resid_post"][
        0, target_pos_added_all
    ]

    # Step 3. Patch resid_post during control run
    hook_patch = model.language_model.layers[layer].register_forward_hook(
        patch_resid_post(patched_cache)
    )
    with torch.no_grad():
        patched_logits = model(**control_inputs, use_cache=False).logits
    hook_patch.remove()

    return control_logits, intervene_logits, patched_logits


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

main_img_dir = data_dir

unified_query_dict = dict()
dict_of_all_res = dict()

for iii, (img_id, caption1, caption2) in tqdm(
    enumerate(coco_json_data_leftrightonly), total=len(coco_json_data_leftrightonly)
):

    dict_of_all_res[(img_id, coco_json_data_leftright_tracker[iii])] = dict()

    image_path = os.path.join(main_img_dir, f"{img_id:012d}.jpg")

    unified_query, subj1, obj1 = make_unified_query(caption1, caption2)
    unified_query_dict[img_id] = unified_query

    text_prompt = "QUESTION: " + unified_query + "Answer left or right. ANSWER: "

    control_img = Image.open(image_path)
    intervene_img = recolor_with_random_rainbow(control_img)

    plt.figure()
    plt.imshow(intervene_img)
    plt.savefig(args.save_output_path[:-4] + "_noise_img_rainbow.png")
    plt.close()
    # image_intervene = intervene_img

    DEVICE = next(model.parameters()).device
    prompt = f"<image>\n{text_prompt}"
    inputs = tokenizer(prompt, return_tensors="pt").to(DEVICE)
    tokens = tokenizer.convert_ids_to_tokens(inputs.input_ids[0])

    control_inputs = processor(text=prompt, images=control_img, return_tensors="pt").to(
        DEVICE
    )
    intervene_inputs = processor(
        text=prompt, images=intervene_img, return_tensors="pt"
    ).to(DEVICE)

    # Tokenize subj1 and obj1 separately to identify their last subtoken
    subj1_tokens = tokenizer.tokenize(" " + subj1)
    obj1_tokens = tokenizer.tokenize(" " + obj1)

    # We use the last token of each to represent the position of the word
    subj1_last_token = subj1_tokens[-1]
    obj1_last_token = obj1_tokens[-1]

    # Now find their index in the full token list (after <image> token)
    # Since <image> token is treated as one text token but stands for 576 image tokens,
    # tokens[1:] corresponds to what comes after "<image>", so offset indices by +1
    subj1_index_in_tokens = (
        tokens[1:].index(subj1_last_token) + 1
    )  # +1 to correct offset
    obj1_index_in_tokens = tokens[1:].index(obj1_last_token) + 1

    # Adjust to align with model input positions after 576 image tokens
    # but we add 575 because the <s> token is also counted as one of the image tokens.
    subj1_position = subj1_index_in_tokens + 575
    obj1_position = obj1_index_in_tokens + 575

    for layer in range(1, 32):
        # print(f"\n[INFO] Running patching at layer {layer}")

        dict_of_all_res[(img_id, coco_json_data_leftright_tracker[iii])][layer] = dict()
        for modality in ["image", "text", "text_objwords"]:  # []:
            if modality == "image":
                target_pos_added = [1 + u for u in list(range(576))]
            elif modality == "text_objwords":
                target_pos_added = [subj1_position, obj1_position]
            else:
                target_pos_added = list(range(575, len(tokens) + 575))

            original_logits, intervene_logits, patched_logits = (
                run_patch_resid_post_onlysubj(
                    model, control_inputs, intervene_inputs, layer, target_pos_added
                )
            )

            left_id = tokenizer.convert_tokens_to_ids("left")
            right_id = tokenizer.convert_tokens_to_ids("right")

            probs_1 = torch.nn.functional.log_softmax(original_logits[0, -1], dim=-1)
            probs_2 = torch.nn.functional.log_softmax(intervene_logits[0, -1], dim=-1)
            probs_patched = torch.nn.functional.log_softmax(
                patched_logits[0, -1], dim=-1
            )

            dict_of_all_res[(img_id, coco_json_data_leftright_tracker[iii])][layer][
                modality
            ] = (
                (probs_1[left_id] - probs_patched[left_id])
                / (probs_1[left_id] - probs_2[left_id]),
                (probs_1[right_id] - probs_patched[right_id])
                / (probs_1[right_id] - probs_2[right_id]),
            )

    # export final_leftright_probs into some csv.
    # so that I can load them in later!

    torch.save(dict_of_all_res, args.save_output_path + "_dict_of_all_res.pt")
    torch.save(unified_query_dict, args.save_output_path + "_unified_query_dict.pt")
