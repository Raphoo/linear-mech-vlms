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


# ----------------------------- #
#         Argparser            #
# ----------------------------- #
parser = argparse.ArgumentParser(
    description="Intervene on LLaVA token embeddings at specific layers."
)
parser.add_argument(
    "--image_path",
    type=str,
    required=False,
    default=r"data/camel_3-1_frog_0-1.png",
    help="Path to the image file.",
)
parser.add_argument(
    "--query", type=str, required=True, help="The prompt/question to ask LLaVA."
)
# parser.add_argument("--layer", type=int, required=False, default=16, help="layer to patch from.")
parser.add_argument(
    "--patch_type",
    type=str,
    required=True,
    help="what to patch? if <image>, patch frog and camel locations. if <objects>, patch object words. otherwise, enter string.",
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


def log_probs_to_csv(
    csv_path, patch_type, layer, left1, left2, left_patch, right1, right2, right_patch
):
    header = [
        "patch_type",
        "layer",
        "left_img1",
        "left_img2",
        "left_img1_patched",
        "right_img1",
        "right_img2",
        "right_img1_patched",
    ]
    row = [
        patch_type,
        layer,
        left1,
        left2,
        left_patch,
        right1,
        right2,
        right_patch,
    ]

    write_header = not os.path.exists(csv_path)
    with open(csv_path, "a", newline="") as f:
        writer = csv.writer(f)
        if write_header:
            writer.writerow(header)
        writer.writerow(row)


def extract_locs(s):
    # Match pattern like loc_1-2_3-4
    match = re.search(r"loc_(\d+)-(\d+)_(\d+)-(\d+)", s)
    if not match:
        raise ValueError("No loc_*_*_*_* pattern found in the string.")

    # Extract and convert to integers
    w, z, x, y = map(int, match.groups())
    return w, z, x, y


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


def make_white_background(image_path):
    """Loads an image with a transparent background and makes the background white."""
    try:
        img = Image.open(image_path)
        if img.mode == "RGBA":
            # Create a white background image
            background = Image.new("RGB", img.size, (255, 255, 255))
            # Paste the image onto the background, only where the image is not transparent
            background.paste(img, mask=img.split()[3])  # 3 is the alpha channel
            return background
        elif img.mode == "RGB":
            return img  # Image already has a non-transparent background
        else:
            print(
                f"Warning: Image mode {img.mode} is not supported. Returning the original image."
            )
            return img
    except FileNotFoundError:
        print(f"Error: Image file not found at {image_path}")
        return None


def make_noise_image(image_control):
    width, height = image_control.size
    mode = image_control.mode

    if mode == "RGB":
        noise = np.random.randint(0, 256, (height, width, 3), dtype=np.uint8)
    elif mode == "L":
        noise = np.random.randint(0, 256, (height, width), dtype=np.uint8)
    elif mode == "RGBA":
        noise = np.random.randint(0, 256, (height, width, 4), dtype=np.uint8)
    else:
        raise ValueError(f"Unsupported image mode: {mode}")

    return Image.fromarray(noise, mode=mode)


def make_noise_image(image_control):
    width, height = image_control.size
    mode = image_control.mode

    if mode == "RGB":
        noise = np.random.randint(0, 256, (height, width, 3), dtype=np.uint8)
    elif mode == "L":
        noise = np.random.randint(0, 256, (height, width), dtype=np.uint8)
    elif mode == "RGBA":
        noise = np.random.randint(0, 256, (height, width, 4), dtype=np.uint8)
    else:
        # Default to RGB if mode is unsupported
        noise = np.random.randint(0, 256, (height, width, 3), dtype=np.uint8)
        mode = "RGB"

    noise_img = Image.fromarray(noise, mode)
    return noise_img


image_path = args.image_path
# control_img = make_white_background(image_path)

# intervene_img = control_img.transpose(Image.FLIP_LEFT_RIGHT)#load_image(intervene_img_path)
# text_prompt="QUESTION: Is the frog to the left or right of the camel? Answer left or right. ANSWER: "
text_prompt = args.query
# layer = args.layer


# image_control = make_noise_image(control_img)
# control_img = image_control


control_img = Image.open(image_path)
if "flipped" in args.save_output_path:
    control_img = control_img.transpose(Image.FLIP_LEFT_RIGHT)

plt.figure()
plt.imshow(control_img)
plt.savefig(args.save_output_path[:-4] + "_noise_img.png")
plt.close()
# image_intervene = intervene_img

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


if args.patch_type == "<image>":
    matches = re.findall(r"(\d)-(\d)", image_path)
    flat_tuple = tuple(int(num) for pair in matches for num in pair)
    x, y, w, z = flat_tuple
    img_camel_patches = [
        24 * i + 6 * x + 24 * (6 * y) + k for k in range(6) for i in range(6)
    ]
    img_frog_patches = [
        24 * i + 6 * w + 24 * (6 * z) + k for k in range(6) for i in range(6)
    ]
    target_pos = img_frog_patches + img_camel_patches

    all_other_imgs = [pp for pp in range(576) if pp not in target_pos]
    target_pos_added = [1 + u for u in all_other_imgs]  # target_pos] #bc of <s> token

    target_pos_added = list(range(1, 575))
else:
    # tok_strs = ['<s>','<image>','▁','<0x0A>','QUEST','ION',':','▁Is','▁the','▁to','▁the','▁left','▁or','▁right','▁of','▁the','?','▁Answer','▁left','▁or','▁right','.','▁A','NS','W','ER',':','▁']
    # target_pos = [next(i for i, tok in enumerate(tokens) if token_str in tok) for token_str in tok_strs]
    # target_pos_added = [575 + u for u in target_pos]
    target_pos_frog = []
    target_pos_camel = []
    if "bottle" in text_prompt:
        firstword = ["le"]
        secondword = ["▁plant"]
    elif "chair" in text_prompt:
        firstword = ["▁book"]
        secondword = ["▁chair"]
    for word in firstword:  # ["▁glo", "ve"]:  # ["▁f", "rog"]:
        target_pos_frog += [
            idx for idx, tok in enumerate(tokens) if tok == word
        ]  # [next(i for i, tok in enumerate(tokens) if token_str in tok) for token_str in ["▁f", "rog"]]
    for word in secondword:  # ["▁back", "pack"]:  # ["▁cam", "el"]:
        target_pos_camel += [
            idx for idx, tok in enumerate(tokens) if tok == word
        ]  # [next(i for i, tok in enumerate(tokens) if token_str in tok) for token_str in [ "▁cam", "el"]]

    # target_pos = [tokens.index(f) for f in ["▁f", "rog", "▁cam", "el", '▁left', '▁right']] #get the first index only
    # let's add a buffer of 1 pos.
    # print("frog:", target_pos_frog)
    # print("camel:", target_pos_camel)

    # target_pos_added_frog = (
    #     [target_pos_frog[0] - 1] + target_pos_frog + [target_pos_frog[-1] + 1]
    # )
    target_pos_added_frog = [575 + u for u in target_pos_frog]

    # target_pos_added_camel = (
    #     [target_pos_camel[0] - 1] + target_pos_camel + [target_pos_camel[-1] + 1]
    # )
    target_pos_added_camel = [575 + u for u in target_pos_camel]

    # new_target_pos = []
    # for tt in target_pos_frog:
    #     if tt-1 not in target_pos_frog and tt-1 not in new_target_pos:
    #         new_target_pos.append(tt-1)
    #     new_target_pos.append(tt)
    #     if tt+1 not in target_pos_frog and tt-1 not in new_target_pos:
    #         new_target_pos.append(tt+1)
    # target_pos_added_frog = [575 + u for u in new_target_pos]  #because in "tokens" there is already 1 <image> token. but this later becomes 576

    # new_target_pos = []
    # for tt in target_pos_camel:
    #     if tt-1 not in target_pos_camel and tt-1 not in new_target_pos:
    #         new_target_pos.append(tt-1)
    #     new_target_pos.append(tt)
    #     if tt+1 not in target_pos_camel and tt-1 not in new_target_pos:
    #         new_target_pos.append(tt+1)
    # target_pos_added_camel= [575 + u for u in new_target_pos]  #because in "tokens" there is already 1 <image> token. but this later becomes 576

    # target_pos_added = list(range(575, 607))


control_inputs = processor(text=prompt, images=control_img, return_tensors="pt").to(
    DEVICE
)
# intervene_inputs = processor(text=prompt, images=image_intervene, return_tensors="pt").to(DEVICE)


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


def run_patch_resid_post(
    model, control_inputs, layer, chosen_spatial_ID_frog, chosen_spatial_ID_camel
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

    #print("added_frog_targ:", target_pos_added_frog)
    #print("resid post shape:", control_cache["resid_post"].shape)
    # print("mean frog shape:", mean_frog.shape, "\nmean camel shape:", mean_camel.shape)

    patched_cache[0, target_pos_added_frog] = (
        control_cache["resid_post"][0, target_pos_added_frog]
        + chosen_spatial_ID_frog.to(DEVICE) * mult_factor
        - chosen_spatial_ID_camel.to(DEVICE) * mult_factor
    )  # - mean_frog + mean_camel

    patched_cache[0, target_pos_added_camel] = (
        control_cache["resid_post"][0, target_pos_added_camel]
        + chosen_spatial_ID_camel.to(DEVICE) * mult_factor
        - chosen_spatial_ID_frog.to(DEVICE) * mult_factor
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


# try fixed layer at 13!
for layer in range(5, 20):  # [5, 14, 16, 20, 30]:
    print(f"\n[INFO] Running patching at layer {layer}")
    # print("dict1:" , big_embeds_dicts_foralllayers)
    # print("\n\ndict2:", big_embeds_dicts_foralllayers2)

    w, z, x, y = extract_locs(args.save_output_path)
    # chosen_spatial_ID_camel = camel_delta_dict[((w, z), (x, y))].squeeze()
    # chosen_spatial_ID_frog = frog_delta_dict[((w, z), (x, y))].squeeze()

    chosen_spatial_ID_first = big_embeds_dicts_foralllayers[13][(w, z)].squeeze()
    chosen_spatial_ID_second = big_embeds_dicts_foralllayers[13][(x, y)].squeeze()

    original_logits, patched_logits = run_patch_resid_post(
        model, control_inputs, layer, chosen_spatial_ID_first, chosen_spatial_ID_second
    )

    left_id = tokenizer.convert_tokens_to_ids("left")
    right_id = tokenizer.convert_tokens_to_ids("right")

    final_leftright_probs = dict()

    probs_1 = torch.nn.functional.log_softmax(original_logits[0, -1], dim=-1)
    final_leftright_probs["img1"] = [probs_1[left_id], probs_1[right_id]]
    print(f"orig\nP(Left) = {probs_1[left_id]:.6f}")
    print(f"P(Right) = {probs_1[right_id]:.6f}\n\n")

    probs_patched = torch.nn.functional.log_softmax(patched_logits[0, -1], dim=-1)
    final_leftright_probs["img1_patched"] = [
        probs_patched[left_id],
        probs_patched[right_id],
    ]
    print(f"patched\nP(Left) = {probs_patched[left_id]:.6f}")
    print(f"P(Right) = {probs_patched[right_id]:.6f}\n\n")
    # 5. Write to CSV
    log_probs_to_csv(
        csv_path=args.save_output_path,
        patch_type=args.patch_type,
        layer=layer,
        left1=probs_1[left_id].item(),
        left2=0.0,
        left_patch=probs_patched[left_id].item(),
        right1=probs_1[right_id].item(),
        right2=0.0,
        right_patch=probs_patched[right_id].item(),
    )


# export final_leftright_probs into some csv.
# so that I can load them in later!
