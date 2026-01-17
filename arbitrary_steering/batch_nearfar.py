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
from PIL import Image
import argparse
from transformers.models.llama.modeling_llama import LlamaRotaryEmbedding
from torch.nn.functional import cosine_similarity, softmax
import numpy as np
import re
import csv
import os

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

# ----------------------------- #
#         Argparser            #
# ----------------------------- #
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
    "--universal_id_path",
    type=str,
    required=True
)

parser.add_argument(
    "--save_output_path",
    type=str,
    required=True,
    help="Path to save the output data to a csv.",
)


parser.add_argument(
    "--mult_factor",
    type=float,
    required=False,
    default=10,
    help="Multiplier for the spatial ID.",
)

args = parser.parse_args()

mult_factor = args.mult_factor

data_dir = args.data_dir


# some text manipulation code


# for LAYER in [5,7,9,11,13,15,17,19,21,23,25,27,29,31]:
# ------------------------
# CONFIG
# ------------------------
MODEL_NAME = "llava-hf/llava-1.5-7b-hf"
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
DTYPE = torch.float16
# IMAGE_PATH = "/content/drive/MyDrive/Raphi Kang Research/3. BINDING VLM /data/camel_0-0_frog_2-0.png" # change this!

# ------------------------
# LOAD MODEL
# ------------------------
tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME, use_fast=True)
torch.set_grad_enabled(False)

model = LlavaForConditionalGeneration.from_pretrained(
    MODEL_NAME,
    torch_dtype=torch.float16,
    low_cpu_mem_usage=True,
    device_map="auto",
)
processor = AutoProcessor.from_pretrained(MODEL_NAME)


def cache_resid_post(storage_dict):
    def hook(module, inputs, outputs):
        #print("caching:", outputs)
        if isinstance(outputs, tuple):
            hidden_states = outputs[0]
        else:
            hidden_states = outputs
        storage_dict["resid_post"] = hidden_states.detach()
        return outputs

    return hook


def patch_resid_post(intervene_resid_post):
    def hook(module, inputs, outputs):
        #print("patching:", outputs)
        if isinstance(outputs, tuple):
            return (intervene_resid_post,)
        else:
            return intervene_resid_post

    return hook


def run_patch_resid_post_onlysubj(
    model,
    control_inputs,
    layer,
    chosen_spatial_ID_subj,
    chosen_spatial_ID_obj,
    target_pos_added_subj,
    target_pos_added_obj,
):
    # Step 1. Get control resid_post
    control_cache = {}
    hook_control = model.language_model.layers[layer].register_forward_hook(
        cache_resid_post(control_cache)
    )
    with torch.no_grad():
        control_logits = model(**control_inputs, use_cache=False).logits
    hook_control.remove()

    patched_cache = control_cache["resid_post"].clone()
    # print("mean el shape:", mean_el.shape,"\ncontrolchache shape:", control_cache["resid_post"].shape)

    #print("added_subj_targ:", target_pos_added_subj)
    #print("resid post shape:", control_cache["resid_post"].shape)
    # print("mean frog shape:", mean_frog.shape, "\nmean camel shape:", mean_camel.shape)

    patched_cache[0, target_pos_added_subj] = (
        control_cache["resid_post"][0, target_pos_added_subj]
        + chosen_spatial_ID_subj.to(DEVICE) * mult_factor
        - chosen_spatial_ID_obj.to(DEVICE) * mult_factor
    )  # - mean_frog + mean_camel

    patched_cache[0, target_pos_added_obj] = (
        control_cache["resid_post"][0, target_pos_added_obj]
        # + chosen_spatial_ID_obj.to(DEVICE) * mult_factor
        # - chosen_spatial_ID_subj.to(DEVICE) * mult_factor
    )  # - mean_camel + mean_frog

    # Step 3. Patch resid_post during control run
    hook_patch = model.language_model.layers[layer].register_forward_hook(
        patch_resid_post(patched_cache)
    )
    with torch.no_grad():
        patched_logits = model(**control_inputs, use_cache=False).logits
    hook_patch.remove()

    return control_logits, patched_logits


def get_delta_embds_from_dict(words_list, embdict, layer=0):
    """
    words_list should have 1 item.
    return the delta between the mean tensor for this token embedding for that layer, as well as the mean tensor itself
    """
    # camel_emb_list = []

    delta_dict = dict()

    cc = words_list[0]

    el_embeds = embdict[layer][cc]
    print("el shape:", cc, "___", list(el_embeds.values())[0].shape)

    stacked_vals = torch.stack([v for v in el_embeds.values() if v is not None])
    cc_intermed = stacked_vals.mean(dim=0)
    # camel_emb_list.append(cc_intermed)

    for keyy in el_embeds:
        delta_dict[keyy] = el_embeds[keyy] - cc_intermed

    return delta_dict, cc_intermed


# coco_json = os.path.join(data_dir, "coco_qa_two_obj.json")
# coco_json_data = json.load(open(coco_json))

# # for the sake of this experiment, we'll only use the left/right questions.
# coco_json_data_leftrightonly = []
# coco_json_data_leftright_tracker = []
# for i in range(len(coco_json_data)):
#     if "left " in coco_json_data[i][1]:

#         coco_json_data_leftrightonly.append(coco_json_data[i])
#         coco_json_data_leftright_tracker.append("left")
#     elif "right " in coco_json_data[i][1]:
#         coco_json_data_leftrightonly.append(coco_json_data[i])
#         coco_json_data_leftright_tracker.append("right")

# random_indices = random.sample(range(len(coco_json_data_leftrightonly)), 100)
# coco_json_data_leftrightonly = [coco_json_data_leftrightonly[i] for i in random_indices]
# coco_json_data_leftright_tracker = [
#     coco_json_data_leftright_tracker[i] for i in random_indices
# ]

# main_img_dir = os.path.join(data_dir, "val2017")

# let's load in images from paired grid instead.
main_img_dir = data_dir

unified_query_dict = dict()
dict_of_all_res = dict()

# shuffl images in main_img_dir and grab just 100 of them. They may not start with MIRROR.
images_in_main_img_dir = os.listdir(main_img_dir)
random.shuffle(images_in_main_img_dir)
# we want to grab the first 100 images that start with MIRROR.
images_in_main_img_dir = [
    img_id for img_id in images_in_main_img_dir if "MIRROR" not in img_id
]

# the image_ids look like "amplifier_0-0_artichoke_2-1.jpg" or other objects. We only want to grab images of the form "OBJECT1_0-0_OBJECT2_3-0.jpg"
images_in_main_img_dir = [
    img_id for img_id in images_in_main_img_dir if "0-0" in img_id and "3-0" in img_id
]
images_in_main_img_dir = images_in_main_img_dir[:100]


# we will fill out a leftright tracker. If object is in 0-0, it is left. If object is in 3-0, it is right.
leftright_tracker = dict()
for img_id in images_in_main_img_dir:
    object1, position1, object2, position2 = img_id.removesuffix(".png").split("_")
    if position1 == "0-0":
        leftright_tracker[img_id] = "near"
    else:
        leftright_tracker[img_id] = "far"

from tqdm import tqdm

iii = -1
for img_id in tqdm(images_in_main_img_dir):
    iii += 1
    # grab the two objects and their positions from the image_id.
    object1, position1, object2, position2 = img_id.removesuffix(".png").split("_")

    # now make query:
    unified_query = f"Is the {object1} near or far from the {object2}?"
    subj1, obj1 = object1, object2

    image_path = os.path.join(main_img_dir, img_id)

    dict_of_all_res[(img_id, leftright_tracker[img_id])] = dict()

    unified_query_dict[img_id] = unified_query

    text_prompt = "QUESTION: " + unified_query + "Answer Near or Far. ANSWER: "

    control_img = Image.open(image_path)

    DEVICE = next(model.parameters()).device
    prompt = f"<image>\n{text_prompt}"
    inputs = tokenizer(prompt, return_tensors="pt").to(DEVICE)

    tokens = tokenizer.convert_ids_to_tokens(inputs.input_ids[0])

    big_embeds_dicts_foralllayers = torch.load(
        args.universal_id_path,
        map_location="cpu",
        weights_only=False,
    )

    # big_embeds_dicts_foralllayers2 = torch.load("/groups/perona/raphi/vlm-bind-save/output_embeds_v1_neighbors.pt", map_location="cpu", weights_only=False)

    target_pos_subj = []
    target_pos_obj = []
    subj_token = [tokenizer.tokenize(" " + subj1, add_special_tokens=False)[-1]]
    obj_token = [tokenizer.tokenize(" " + obj1, add_special_tokens=False)[-1]]

    for word in subj_token:
        target_pos_subj += [idx for idx, tok in enumerate(tokens) if tok == word]
    for word in obj_token:
        target_pos_obj += [idx for idx, tok in enumerate(tokens) if tok == word]

    # what happens if the word appears multiple times? we just hope it doesnt lol

    target_pos_added_subj = [576 + u for u in target_pos_subj]
    target_pos_added_obj = [576 + u for u in target_pos_obj]

    control_inputs = processor(text=prompt, images=control_img, return_tensors="pt").to(
        DEVICE
    )
    # intervene_inputs = processor(text=prompt, images=image_intervene, return_tensors="pt").to(DEVICE)

    for layer in range(12, 15):  # [5, 14, 16, 20, 30]:
        print(f"\n[INFO] Running patching at layer {layer}")

        dict_of_all_res[(img_id, leftright_tracker[img_id])][layer] = dict()

        # valid_pairs = generate_non_overlapping_pairs()
        # for coord_1, coord_2 in valid_pairs:
        one_coord_list = [(i, j) for i in range(4) for j in range(4)]
        for coord_1 in one_coord_list:

            chosen_spatial_ID_first = big_embeds_dicts_foralllayers[layer][
                coord_1
            ].squeeze()

            coord_2 = (
                3 - coord_1[0],
                coord_1[1],
            )  # for the purpose of subtracting the opposite ID from the sub latent, we
            # need to flip the x coordinate.
            chosen_spatial_ID_second = big_embeds_dicts_foralllayers[layer][
                coord_2
            ].squeeze()

            original_logits, patched_logits = run_patch_resid_post_onlysubj(
                model,
                control_inputs,
                layer,
                chosen_spatial_ID_first,
                chosen_spatial_ID_second,  # chosen_spatial_ID_second,
                target_pos_added_subj,
                target_pos_added_obj,
            )

            left_id = tokenizer.convert_tokens_to_ids("▁Near")
            right_id = tokenizer.convert_tokens_to_ids("▁Far")

            # left == near, right == far

            probs_1 = torch.nn.functional.log_softmax(original_logits[0, -1], dim=-1)

            probs_patched = torch.nn.functional.log_softmax(
                patched_logits[0, -1], dim=-1
            )

            topk_1 = probs_1.topk(10)
            topk_patched = probs_patched.topk(10)

            # Convert token IDs to strings
            tokens_1 = tokenizer.convert_ids_to_tokens(topk_1.indices.tolist())
            tokens_patched = tokenizer.convert_ids_to_tokens(
                topk_patched.indices.tolist()
            )

            # Format them with probabilities
            topk_1_str = ", ".join(
                [
                    f"{tok} ({prob.item():.2f})"
                    for tok, prob in zip(tokens_1, topk_1.values)
                ]
            )
            topk_patched_str = ", ".join(
                [
                    f"{tok} ({prob.item():.2f})"
                    for tok, prob in zip(tokens_patched, topk_patched.values)
                ]
            )
            dict_of_all_res[(img_id, leftright_tracker[img_id])][layer][coord_1] = (
                probs_1[left_id] - probs_patched[left_id],
                probs_1[right_id] - probs_patched[right_id],
            )
    plt.figure(figsize=(10, 10))
    plt.axis("off")
    plt.imshow(control_img)
    plt.suptitle(
        f"Control Image: {unified_query}\n"
        f"Original Topk: {topk_1_str}\n"
        f"Patched Topk: {topk_patched_str}"
    )
    plt.tight_layout()
    #plt.savefig(args.save_output_path[:-4] + "_noise_img.png")
    plt.close()
    # image_intervene = intervene_img
    # export final_leftright_probs into some csv.
    # so that I can load them in later!

    torch.save(dict_of_all_res, args.save_output_path + "/dict_of_all_res.pt")
    torch.save(unified_query_dict, args.save_output_path + "/unified_query_dict.pt")
