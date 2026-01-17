"""
Spatial blur diagnosis experiment for vision-language models.

This script evaluates VLM reasoning by applying Gaussian blur to different
spatial locations in images and measuring the effect on spatial relationship predictions.
"""

import json
import math
import os
import random
import argparse
from dataclasses import dataclass
from typing import Dict, List, Tuple, Optional

import torch
import torch.nn.functional as F
from torchvision import transforms
from PIL import Image, ImageFilter
from pycocotools.coco import COCO
import matplotlib.pyplot as plt

from tqdm import tqdm

from transformers import (
    AutoProcessor,
    AutoTokenizer,
    AutoModelForCausalLM,
    AutoModelForVision2Seq,
)

# -----------------------------
# Globals & small utils
# -----------------------------
RELATION_MAP = {
    "to the left of": "Left",
    "to the right of": "Right",
    "above": "Above",
    "below": "Below",
}
RELATION_CANONICAL = list(RELATION_MAP.keys())
ANSWER_TOKENS = [" Left", " Right", " Above", " Below"]  # note leading space

to_tensor = transforms.ToTensor()
to_pil = transforms.ToPILImage()


@dataclass
class COCOHandles:
    coco: COCO
    img_root: str  # path to val2017 images folder


def _parse_caption(caption: str) -> Tuple[str, str, str]:
    cap = caption.strip().rstrip(".").lower()
    cap = cap.replace("  ", " ")
    rel_used = None
    for r in sorted(RELATION_CANONICAL, key=len, reverse=True):
        if f" {r} " in cap:
            rel_used = r
            break
    if rel_used is None:
        raise ValueError(f"Caption lacks a supported relation: {caption}")

    left, right = cap.split(f" {rel_used} ")

    def _strip_prefix(s):
        for pref in [
            "a photo of a ",
            "a photo of an ",
            "a photo of the ",
            "photo of a ",
            "photo of an ",
            "photo of the ",
        ]:
            if s.startswith(pref):
                return s[len(pref) :]
        return s

    w1_phrase = _strip_prefix(left).strip()
    w2_phrase = right.strip()
    for pref in ["a ", "an ", "the "]:
        if w2_phrase.startswith(pref):
            w2_phrase = w2_phrase[len(pref) :]
    return w1_phrase, rel_used, w2_phrase


def _load_image(img_root: str, coco: COCO, image_id: int) -> Image.Image:
    info = coco.loadImgs([image_id])[0]
    fp = os.path.join(img_root, info["file_name"])
    return Image.open(fp).convert("RGB")


def _coco_cat_id_by_name(coco: COCO, name: str) -> Optional[int]:
    cats = coco.getCatIds(catNms=[name])
    if len(cats) == 0:
        return None
    return cats[0]


def _get_instances_for_cat(coco: COCO, image_id: int, cat_id: int) -> List[Dict]:
    ann_ids = coco.getAnnIds(imgIds=[image_id], catIds=[cat_id], iscrowd=None)
    anns = coco.loadAnns(ann_ids)
    return [a for a in anns if a.get("bbox")]


def _bbox_center(b: List[float]) -> Tuple[float, float]:
    return (b[0] + b[2] / 2.0, b[1] + b[3] / 2.0)


def _pick_pair_by_relation(
    anns_w1: List[Dict], anns_w2: List[Dict], relation: str
) -> Tuple[List[float], List[float]]:
    def score_rel(c1, c2):
        if relation == "to the left of":
            return c2[0] - c1[0]
        if relation == "to the right of":
            return c1[0] - c2[0]
        if relation == "above":
            return c2[1] - c1[1]
        if relation == "below":
            return c1[1] - c2[1]
        return -1e9

    best, best_score = None, -1e18
    for a in anns_w1:
        c1 = _bbox_center(a["bbox"])
        for b in anns_w2:
            c2 = _bbox_center(b["bbox"])
            s = score_rel(c1, c2)
            if s > best_score:
                best_score = s
                best = (a["bbox"], b["bbox"])
    if best is None:
        aw1 = max(anns_w1, key=lambda x: x["bbox"][2] * x["bbox"][3])["bbox"]
        aw2 = max(anns_w2, key=lambda x: x["bbox"][2] * x["bbox"][3])["bbox"]
        return aw1, aw2
    return best


def _boxes_intersect(b1, b2) -> bool:
    x1, y1, w1, h1 = b1
    x2, y2, w2, h2 = b2
    return not (x1 + w1 <= x2 or x2 + w2 <= x1 or y1 + h1 <= y2 or y2 + h2 <= y1)


def _place_outside_patches(
    img_w: int,
    img_h: int,
    box: List[float],
    forbidden: List[List[float]],
    n: int,
    seed: int,
    allow_outside_overlap: bool = True,  # <- new
    max_tries: int = 50000,  # <- give it more room before giving up
) -> List[List[float]]:
    random.seed(seed)
    bw, bh = int(round(box[2])), int(round(box[3]))
    placements, tries = [], 0

    def _fits(x, y):
        cand = [x, y, bw, bh]
        # must not intersect forbidden (e.g., w1, maybe w2)
        if any(_boxes_intersect(cand, f) for f in forbidden):
            return False
        # optionally allow overlap among outside patches
        if not allow_outside_overlap:
            if any(_boxes_intersect(cand, p) for p in placements):
                return False
        return True

    while len(placements) < n and tries < max_tries:
        tries += 1
        x = random.randint(0, max(0, img_w - bw))
        y = random.randint(0, max(0, img_h - bh))
        if _fits(x, y):
            placements.append([x, y, bw, bh])

    # If we still didn’t place all n, just repeat some valid ones (keeps size fixed)
    while len(placements) < n and placements:
        placements.append(random.choice(placements))

    # If image is too small and nothing could be placed, fall back to putting one copy
    if not placements and img_w >= bw and img_h >= bh:
        placements.append([0, 0, bw, bh])

    return placements


def _apply_gaussian_blur_patch(
    pil_img: Image.Image, box_xywh: List[float], radius: int = 15
) -> Image.Image:
    x, y, w, h = [int(round(v)) for v in box_xywh]
    x = max(0, min(x, pil_img.width - 1))
    y = max(0, min(y, pil_img.height - 1))
    w = max(1, min(w, pil_img.width - x))
    h = max(1, min(h, pil_img.height - y))
    patch = pil_img.crop((x, y, x + w, y + h)).filter(
        ImageFilter.GaussianBlur(radius=radius)
    )
    out = pil_img.copy()
    out.paste(patch, (x, y))
    return out


# -----------------------------
# Required function 1: get_blurred_imgs
# -----------------------------
def get_blurred_imgs(
    image_id: int,
    caption: str,
    coco_handles: COCOHandles,
    blur_radius: int = 15,
    device: str = "cpu",
) -> List[torch.Tensor]:
    """
    Returns 16 tensors in order:
      [w1_blurred, w1_blurred_outside_1..15]

    Off-target masks are placed elsewhere in the image with the SAME size as w1_bbox,
    avoiding intersection with w1_bbox. (We do not forbid overlap with w2.)
    """
    coco = coco_handles.coco
    img_root = coco_handles.img_root
    w1_name, relation, w2_name = _parse_caption(caption)

    img = _load_image(img_root, coco, image_id)
    W, H = img.width, img.height

    w1_cat = _coco_cat_id_by_name(coco, w1_name)
    w2_cat = _coco_cat_id_by_name(coco, w2_name)
    if w1_cat is None or w2_cat is None:
        raise ValueError(f"Cannot resolve COCO categories: '{w1_name}' or '{w2_name}'")

    anns_w1 = _get_instances_for_cat(coco, image_id, w1_cat)
    anns_w2 = _get_instances_for_cat(coco, image_id, w2_cat)
    if not anns_w1 or not anns_w2:
        raise ValueError(
            f"No instances for '{w1_name}' or '{w2_name}' in image {image_id}"
        )

    # Still pick the w1/w2 pair that best matches the relation, but only use w1 for masking
    w1_bbox, _w2_bbox_unused = _pick_pair_by_relation(anns_w1, anns_w2, relation)

    # 1) on-target blur at w1
    w1_blurred_pil = _apply_gaussian_blur_patch(img, w1_bbox, radius=blur_radius)

    # 2) 15 off-target same-size masks that do NOT intersect w1
    forbidden = [
        w1_bbox,
        _w2_bbox_unused,
    ]  # only forbid w1; allow overlap with other content
    seed_base = (image_id * 1315423911) ^ hash((w1_name, w2_name, relation))
    w1_outside_boxes = _place_outside_patches(
        W,
        H,
        w1_bbox,
        forbidden=forbidden,
        n=10,
        seed=seed_base & 0xFFFF,
        allow_outside_overlap=True,
        max_tries=500,
    )

    w1_outside_pils = [
        _apply_gaussian_blur_patch(img, b, radius=blur_radius) for b in w1_outside_boxes
    ]

    out_tensors = [
        to_tensor(w1_blurred_pil),
        *[to_tensor(p) for p in w1_outside_pils],
    ]
    return out_tensors


# -----------------------------
# Model loading & prompting
# -----------------------------
def _resolve_model_name(key: str) -> str:
    k = key.strip().lower()
    if k == "llava":
        return "llava-hf/llava-1.5-7b-hf"
    if k == "llama":
        return "meta-llama/Llama-3.2-11B-Vision-Instruct"
    if k == "qwen":
        return "Qwen/Qwen2.5-VL-7B-Instruct"
    # allow fully qualified names too
    return key


def _load_any_vlm_from_key(
    model_key="llava",
    device="cuda",
    device_map="auto",
    offload_folder=None,
    attn_impl=None,  # e.g., "flash_attention_2" if installed
):
    model_name = _resolve_model_name(model_key)
    try:
        processor = AutoProcessor.from_pretrained(
            model_name, trust_remote_code=True, use_fast=True
        )
    except Exception:
        processor = None

    torch_dtype = torch.bfloat16 if torch.cuda.is_available() else torch.float32

    common_kwargs = dict(
        trust_remote_code=True,
        low_cpu_mem_usage=True,
        device_map=device_map,
        torch_dtype=torch_dtype,
    )

    if offload_folder:
        common_kwargs["offload_folder"] = offload_folder

    if attn_impl is not None:
        common_kwargs["attn_implementation"] = attn_impl

    # Prefer Vision2Seq; fallback to CausalLM
    try:
        model = AutoModelForVision2Seq.from_pretrained(model_name, **common_kwargs)
    except Exception:
        model = AutoModelForCausalLM.from_pretrained(model_name, **common_kwargs)

    try:
        tokenizer = AutoTokenizer.from_pretrained(
            model_name, trust_remote_code=True, use_fast=True
        )
    except Exception:
        tokenizer = getattr(processor, "tokenizer", None)
        if tokenizer is None:
            raise

    if hasattr(model, "to") and device == "cuda":
        model = model.to(device)

    return model, processor, tokenizer, model_name


def _apply_chat_template(processor, tokenizer, question: str, family: str) -> str:
    """
    Build a model-appropriate chat prompt with an image placeholder.
    """
    # If processor supplies a template that supports multimodal messages, use it.
    if processor is not None and hasattr(processor, "apply_chat_template"):
        messages = [
            {
                "role": "user",
                "content": [{"type": "text", "text": question}, {"type": "image"}],
            }
        ]
        return processor.apply_chat_template(messages, add_generation_prompt=True)

    # Family-specific conservative fallbacks
    fam = family.lower()
    if "llava" in fam:
        return f"USER: <image>\n{question}\nASSISTANT:"
    if "llama" in fam:
        # Llama 3.* Vision usually prefers a chat format, but this simple fallback works.
        return f"<|user|>\n<image>\n{question}\n<|assistant|>\n"
    if "qwen" in fam:
        return f"User: <image>\n{question}\nAssistant:"
    return f"USER: <image>\n{question}\nASSISTANT:"


def _first_token_ids(tokenizer) -> Dict[str, int]:
    ids = {}
    for tok in ANSWER_TOKENS:
        toks = tokenizer(tok, add_special_tokens=False).input_ids
        if not toks:
            raise ValueError(f"No ids for token {tok!r}")
        ids[tok.strip()] = toks[0]
    return ids  # "Left","Right","Above","Below" -> id


def _question_from_caption(correct_caption: str) -> Tuple[str, str, str, str]:
    w1, rel, w2 = _parse_caption(correct_caption)
    if rel in ["to the left of", "to the right of"]:
        q = f"Is the {w1} to the left or right of the {w2}? Answer Left or Right."
        corr = "Left" if rel == "to the left of" else "Right"
        wrong = "Right" if corr == "Left" else "Left"
        mode = "lr"
    else:
        q = f"Is the {w1} above or below the {w2}? Answer Above or Below."
        corr = "Above" if rel == "above" else "Below"
        wrong = "Below" if corr == "Above" else "Above"
        mode = "ud"
    return q, corr, wrong, mode


def get_token_in_model_scheme(word, tokenizer):
    return tokenizer.encode(word)[0]


# -----------------------------
# Required function 2: driver
# -----------------------------
def run_vlm_eval(
    model_key: str = "llava",
    device: str = "cuda" if torch.cuda.is_available() else "cpu",
    device_map: str = "auto",
    offload_folder: str = None,
    attn_impl: str = None,
    blur_radius: int = 15,
    captions_json_path: str = None,
    coco_ann_file: str = None,
    coco_img_root: str = None,
    output_path: str = None,
):
    """
    Saves a .pt with:
      {
        "results_dict": {
            "corr": { image_id: [11 logprobs] },
            "wrong": { image_id: [11 logprobs] }
        },
        "queries_dict": { image_id: "question string" },
        "model_name": "<resolved hf id>"
      }
    The 11 logprobs correspond to:
      [original, w1_blur, w2_blur, w1_out1..4, w2_out1..4]
    """
    coco = COCO(coco_ann_file)
    coco_handles = COCOHandles(coco=coco, img_root=coco_img_root)

    model, processor, tokenizer, resolved_model_name = _load_any_vlm_from_key(
        model_key,
        device=device,
        device_map=device_map,
        offload_folder=offload_folder,
        attn_impl=attn_impl,
    )
    model.eval()
    first_tok_ids = _first_token_ids(tokenizer)

    with open(captions_json_path, "r") as f:
        rows = json.load(f)

    results_corr: Dict[int, List[float]] = {}
    results_wrong: Dict[int, List[float]] = {}
    queries_dict: Dict[int, str] = {}

    print_a_few = 0

    print("Running eval for: ", resolved_model_name)

    for row in tqdm(rows):
        image_id = int(row[0])
        correct_caption = row[1]
        wrong_caption = row[
            2
        ]  # not used for the question, only to define "wrong" label

        question, corr_label, wrong_label, mode = _question_from_caption(
            correct_caption
        )
        # print(corr_label, wrong_label)
        # # if these don't look right, I can update them tbh.
        # print("▁" in corr_label, "▁" in wrong_label, "was ▁ in the labels?")
        queries_dict[image_id] = question

        # let's get the correct token ids for corr_label and wrong_label
        tokenized_outputs = tokenizer(
            text=[corr_label, wrong_label],
            add_special_tokens=False,
            return_tensors="pt",
            padding=True,
            padding_side="right",
        )

        corr_id = tokenized_outputs.input_ids[0][0]
        wrong_id = tokenized_outputs.input_ids[1][0]

        # if print_a_few < 5:
        #     print(tokenizer.convert_ids_to_tokens([corr_id, wrong_id]))
        #     print_a_few += 1

        # Prepare original + 10 blurred images
        pil_img = _load_image(coco_img_root, coco, image_id)
        blurred_tensors = get_blurred_imgs(
            image_id=image_id,
            caption=correct_caption,
            coco_handles=coco_handles,
            blur_radius=blur_radius,
            device=device,
        )
        imgs_pil = [pil_img] + [to_pil(t) for t in blurred_tensors]

        # ---- NEW: dump a panel of sample blurs on the 3rd example ----
        # if print_a_few == 3:
        #     ncols = 4
        #     nimgs = len(imgs_pil)
        #     nrows = math.ceil(nimgs / ncols)
        #     fig, axes = plt.subplots(nrows, ncols, figsize=(3 * ncols, 3 * nrows))
        #     axes = axes.flatten()
        #     for i, im in enumerate(imgs_pil):
        #         axes[i].imshow(im)
        #         axes[i].axis("off")
        #         if i == 0:
        #             axes[i].set_title("Original")
        #         elif i == 1:
        #             axes[i].set_title("Object blur")
        #         else:
        #             axes[i].set_title(f"Blur {i}")
        #     for j in range(i + 1, len(axes)):
        #         axes[j].axis("off")
        #     plt.suptitle(f"Question: {question}")
        #     plt.tight_layout()

        #     # construct export path from output_path
        #     out_dir = os.path.dirname(output_path)
        #     out_file = f"{resolved_model_name.replace('/','_')}_sampleblursv2.png"
        #     out_path = os.path.join(out_dir, out_file)
        #     plt.savefig(out_path, dpi=150)
        #     plt.close(fig)
        #     print(f"Saved blur sample grid to {out_path}")

        corr_list, wrong_list = [], []
        for im in imgs_pil:
            prompt = _apply_chat_template(
                processor, tokenizer, question, family=resolved_model_name
            )

            if processor is not None:
                inputs = processor(text=prompt, images=im, return_tensors="pt").to(
                    device
                )
            else:
                inputs = tokenizer(prompt, return_tensors="pt").to(device)

            with torch.no_grad():
                gen = model.generate(
                    **inputs,
                    max_new_tokens=1,
                    do_sample=False,
                    output_scores=True,
                    return_dict_in_generate=True,
                )
            first_scores = gen.scores[0].squeeze(0)
            logp = F.log_softmax(first_scores, dim=-1)

            corr_list.append(logp[corr_id].item())
            wrong_list.append(logp[wrong_id].item())

        # assert len(corr_list) == 11 and len(wrong_list) == 11
        results_corr[image_id] = corr_list
        results_wrong[image_id] = wrong_list

    results_dict = {"corr": results_corr, "wrong": results_wrong}
    (
        os.makedirs(os.path.dirname(output_path), exist_ok=True)
        if os.path.dirname(output_path)
        else None
    )
    torch.save(
        {
            "results_dict": results_dict,
            "queries_dict": queries_dict,
            "model_name": resolved_model_name,
        },
        output_path,
    )
    print(f"Saved to {output_path}")


# -----------------------------
# CLI
# -----------------------------
if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Run spatial blur diagnosis experiments on vision-language models",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )

    # Model arguments
    parser.add_argument(
        "--model_key",
        default="llava",
        help='Model key: "llava", "llama", "qwen", or full HF repo ID',
    )
    parser.add_argument(
        "--device",
        default="cuda" if torch.cuda.is_available() else "cpu",
        help="Device to use",
    )
    parser.add_argument(
        "--device_map", default="auto", help="Device map for model loading"
    )
    parser.add_argument(
        "--offload_folder", default=None, help="Folder for offloading model weights"
    )
    parser.add_argument(
        "--attn_impl",
        default=None,
        help="Attention implementation (e.g., flash_attention_2, sdpa)",
    )

    # Data arguments
    parser.add_argument(
        "--captions_json", required=True, help="Path to captions JSON file"
    )
    parser.add_argument(
        "--coco_ann",
        required=True,
        help="Path to COCO annotations file (e.g., instances_val2017.json)",
    )
    parser.add_argument(
        "--coco_img_root",
        required=True,
        help="Path to COCO images directory (e.g., val2017/)",
    )
    parser.add_argument(
        "--output", required=True, help="Output path for results (.pt file)"
    )

    # Experiment arguments
    parser.add_argument(
        "--blur_radius", type=int, default=15, help="Gaussian blur radius for masking"
    )

    args = parser.parse_args()

    run_vlm_eval(
        captions_json_path=args.captions_json,
        model_key=args.model_key,
        device=args.device,
        device_map=args.device_map,
        offload_folder=args.offload_folder,
        attn_impl=args.attn_impl,
        blur_radius=args.blur_radius,
        coco_ann_file=args.coco_ann,
        coco_img_root=args.coco_img_root,
        output_path=args.output,
    )
