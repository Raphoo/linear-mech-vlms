"""
Depth Diagnosis: Steering Experiments for Height vs Depth Spatial IDs

Tests whether spatial IDs conflate height (above/below) with depth (front/behind).
Figure 7a from paper Section 4.1.
"""

import torch
from transformers import (
    LlavaForConditionalGeneration,
    AutoTokenizer,
    AutoProcessor,
)
from PIL import Image
import argparse
import os
import json
import random

random.seed(42)


def cache_resid_post(storage_dict):
    """Hook to cache residual stream activations"""
    def hook(module, inputs, outputs):
        if isinstance(outputs, tuple):
            hidden_states = outputs[0]
        else:
            hidden_states = outputs
        storage_dict["resid_post"] = hidden_states.detach()
        return outputs
    return hook


def patch_resid_post(intervene_resid_post):
    """Hook to patch residual stream activations with spatial IDs"""
    def hook(module, inputs, outputs):
        if isinstance(outputs, tuple):
            return (intervene_resid_post,)
        else:
            return intervene_resid_post
    return hook


def run_spatial_id_steering(
    model,
    control_inputs,
    layer,
    chosen_spatial_ID_subj,
    chosen_spatial_ID_obj,
    target_pos_added_subj,
    target_pos_added_obj,
    mult_factor,
    device,
):
    """
    Steer spatial IDs in object word activations and measure effect on logits.

    Args:
        model: LLaVA model
        control_inputs: Processed inputs
        layer: Layer to intervene at
        chosen_spatial_ID_subj: Spatial ID to add to subject
        chosen_spatial_ID_obj: Spatial ID to subtract from subject (opposite)
        target_pos_added_subj: Token positions for subject
        target_pos_added_obj: Token positions for object
        mult_factor: Scaling factor for spatial IDs
        device: Device to run on

    Returns:
        control_logits: Unmodified logits
        patched_logits: Logits after spatial ID steering
    """
    # Step 1: Get control activations
    control_cache = {}
    hook_control = model.language_model.layers[layer].register_forward_hook(
        cache_resid_post(control_cache)
    )
    with torch.no_grad():
        control_logits = model(**control_inputs, use_cache=False).logits
    hook_control.remove()

    # Step 2: Patch subject token with spatial IDs
    patched_cache = control_cache["resid_post"].clone()
    patched_cache[0, target_pos_added_subj] = (
        control_cache["resid_post"][0, target_pos_added_subj]
        + chosen_spatial_ID_subj.to(device) * mult_factor
        - chosen_spatial_ID_obj.to(device) * mult_factor
    )

    # Step 3: Apply patched activations
    hook_patch = model.language_model.layers[layer].register_forward_hook(
        patch_resid_post(patched_cache)
    )
    with torch.no_grad():
        patched_logits = model(**control_inputs, use_cache=False).logits
    hook_patch.remove()

    return control_logits, patched_logits


def make_unified_query(caption1, caption2, spatial_words):
    """
    Extract unified query from two spatial captions.

    Args:
        caption1: e.g., "the dog is above the cat"
        caption2: e.g., "the dog is below the cat"
        spatial_words: Set of spatial relation words for the query (e.g., {"above", "below"} or {"front", "behind"})

    Returns:
        query: Unified question
        subj: Subject object
        obj: Object
    """
    words1 = caption1.lower().split()
    words2 = caption2.lower().split()

    # Find the spatial relation word in caption (always above/below for our data)
    spatial_word_in_caption = None
    for word in ["above", "below"]:
        if word in words1:
            spatial_word_in_caption = word
            break

    if spatial_word_in_caption is None:
        raise ValueError(f"No spatial word (above/below) found in caption")

    # Extract subject and object
    # Assuming format: "the <subj> is <spatial> the <obj>"
    try:
        spatial_idx = words1.index(spatial_word_in_caption)
        subj_words = words1[1:spatial_idx-1]  # Skip "the" and "is"
        obj_words = words1[spatial_idx+2:]  # Skip "the"

        subj = " ".join(subj_words)
        obj = " ".join(obj_words)

        # Create binary query using the requested spatial_words
        # Determine order based on which spatial words are being used
        if "above" in spatial_words:
            query = f"Is the {subj} above or below the {obj}?"
        elif "front" in spatial_words or "behind" in spatial_words:
            query = f"Is the {subj} in front or behind the {obj}?"
        else:
            # Fallback: use alphabetical order
            spatial_list = sorted(list(spatial_words))
            query = f"Is the {subj} {spatial_list[0]} or {spatial_list[1]} the {obj}?"

        return query, subj, obj
    except (ValueError, IndexError) as e:
        raise ValueError(f"Failed to parse captions: {caption1}, {caption2}") from e


def main():
    parser = argparse.ArgumentParser(
        description="Run depth diagnosis steering experiments."
    )
    parser.add_argument(
        "--data_dir",
        type=str,
        required=True,
        help="Path to COCO data directory containing val2017/ and coco_qa_two_obj.json",
    )
    parser.add_argument(
        "--output_path",
        type=str,
        required=True,
        help="Path to save output .pt files",
    )
    parser.add_argument(
        "--spatial_ids_path",
        type=str,
        required=True,
        help="Path to spatial IDs .pt file",
    )
    parser.add_argument(
        "--direction",
        type=str,
        choices=["height", "depth"],
        required=True,
        help="Test height (above/below) or depth (front/behind)",
    )
    parser.add_argument(
        "--mult_factor",
        type=float,
        default=5,
        help="Multiplier for spatial ID steering",
    )
    parser.add_argument(
        "--layers",
        type=str,
        default="12,13,14",
        help="Comma-separated list of layers to intervene at",
    )
    parser.add_argument(
        "--num_samples",
        type=int,
        default=100,
        help="Number of random samples to test (default: 100)",
    )

    args = parser.parse_args()

    # Parse layers
    layers = [int(l) for l in args.layers.split(",")]

    # Setup
    MODEL_NAME = "llava-hf/llava-1.5-7b-hf"
    DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

    print(f"Loading model {MODEL_NAME}...")
    tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME, use_fast=True)
    torch.set_grad_enabled(False)

    model = LlavaForConditionalGeneration.from_pretrained(
        MODEL_NAME,
        torch_dtype=torch.float16,
        low_cpu_mem_usage=True,
        device_map="auto",
    )
    processor = AutoProcessor.from_pretrained(MODEL_NAME)
    DEVICE = next(model.parameters()).device

    # Load data
    print(f"Loading COCO data from {args.data_dir}...")
    coco_json = os.path.join(args.data_dir, "coco_qa_two_obj.json")
    coco_json_data = json.load(open(coco_json))

    # Filter for images with above/below relations (used for both experiments)
    # For height: test if y-axis IDs affect "above/below" judgments
    # For depth: test if same y-axis IDs affect "front/behind" judgments (conflation test)
    spatial_words_for_filtering = {"above", "below"}

    coco_filtered = []
    coco_labels = []
    for img_id, caption1, caption2 in coco_json_data:
        for word in spatial_words_for_filtering:
            if f" {word} " in f" {caption1.lower()} ":
                coco_filtered.append((img_id, caption1, caption2))
                coco_labels.append(word)
                break

    # Set query words based on direction
    if args.direction == "height":
        spatial_words = {"above", "below"}
        answer_words = ["Above", "Below"]
    else:  # depth
        spatial_words = {"front", "behind"}
        answer_words = ["Front", "Behind"]  # Note: tokenizer uses "▁Be" for Behind, "▁Front"

    # Random sample
    if len(coco_filtered) > args.num_samples:
        random_indices = random.sample(range(len(coco_filtered)), args.num_samples)
        coco_filtered = [coco_filtered[i] for i in random_indices]
        coco_labels = [coco_labels[i] for i in random_indices]

    print(f"Testing {len(coco_filtered)} images for {args.direction}")

    # Load spatial IDs
    print(f"Loading spatial IDs from {args.spatial_ids_path}...")
    spatial_ids = torch.load(
        args.spatial_ids_path,
        map_location="cpu",
        weights_only=False,
    )

    # Main experiment loop
    main_img_dir = os.path.join(args.data_dir)
    unified_query_dict = {}
    results_dict = {}

    for iii, (img_id, caption1, caption2) in enumerate(coco_filtered):
        print(f"[{iii+1}/{len(coco_filtered)}] Processing image {img_id}...")

        results_dict[(img_id, coco_labels[iii])] = {}

        image_path = os.path.join(main_img_dir, f"{img_id:012d}.jpg")

        # Parse query
        query, subj, obj = make_unified_query(caption1, caption2, spatial_words)
        unified_query_dict[img_id] = query

        if args.direction == "height":
            text_prompt = f"QUESTION: {query} Answer Above or Below. ANSWER: "
        else:
            text_prompt = f"QUESTION: {query} Answer Front or Behind. ANSWER: "

        # Load image
        control_img = Image.open(image_path)
        prompt = f"<image>\n{text_prompt}"
        inputs = tokenizer(prompt, return_tensors="pt").to(DEVICE)
        tokens = tokenizer.convert_ids_to_tokens(inputs.input_ids[0])

        # Find target token positions
        subj_token = [tokenizer.tokenize(" " + subj, add_special_tokens=False)[-1]]
        obj_token = [tokenizer.tokenize(" " + obj, add_special_tokens=False)[-1]]

        target_pos_subj = []
        target_pos_obj = []
        for word in subj_token:
            target_pos_subj += [idx for idx, tok in enumerate(tokens) if tok == word]
        for word in obj_token:
            target_pos_obj += [idx for idx, tok in enumerate(tokens) if tok == word]

        # Adjust for image tokens (576 tokens prepended)
        target_pos_added_subj = [576 + u for u in target_pos_subj]
        target_pos_added_obj = [576 + u for u in target_pos_obj]

        # Process inputs
        control_inputs = processor(text=prompt, images=control_img, return_tensors="pt").to(DEVICE)

        # Test each layer
        for layer in layers:
            print(f"  Layer {layer}...")
            results_dict[(img_id, coco_labels[iii])][layer] = {}

            # Test spatial ID positions varying y-coordinate (height), x fixed at 0
            # Paper uses: (0,0), (0,1), (0,2), (0,3)
            coord_list = [(0, j, 224) for j in range(4)]

            for coord_1 in coord_list:
                # Get spatial IDs
                chosen_spatial_ID_first = spatial_ids[layer]['universal'][coord_1].squeeze()

                # For opposing direction: flip the y-coordinate for height/depth
                if args.direction == "height":
                    coord_2 = (coord_1[0], 3 - coord_1[1], 224)  # Flip y (vertical)
                else:  # depth - also uses y-axis per paper findings
                    coord_2 = (coord_1[0], 3 - coord_1[1], 224)  # Flip y (vertical)

                chosen_spatial_ID_second = spatial_ids[layer]['universal'][coord_2].squeeze()

                # Run steering
                original_logits, patched_logits = run_spatial_id_steering(
                    model,
                    control_inputs,
                    layer,
                    chosen_spatial_ID_first,
                    chosen_spatial_ID_second,
                    target_pos_added_subj,
                    target_pos_added_obj,
                    args.mult_factor,
                    DEVICE,
                )

                # Get token IDs for answer words
                if args.direction == "height":
                    word1_id = tokenizer.convert_tokens_to_ids("▁A")  # "Above"
                    word2_id = tokenizer.convert_tokens_to_ids("▁Below")
                else:  # depth
                    word1_id = tokenizer.convert_tokens_to_ids("▁Be")  # "Behind"
                    word2_id = tokenizer.convert_tokens_to_ids("▁Front")

                # Compute log probabilities
                probs_original = torch.nn.functional.log_softmax(original_logits[0, -1], dim=-1)
                probs_patched = torch.nn.functional.log_softmax(patched_logits[0, -1], dim=-1)

                # Store change in log probability
                results_dict[(img_id, coco_labels[iii])][layer][coord_1] = (
                    (probs_original[word1_id] - probs_patched[word1_id]).item(),
                    (probs_original[word2_id] - probs_patched[word2_id]).item(),
                )

    # Save results
    os.makedirs(os.path.dirname(args.output_path), exist_ok=True)

    results_path = args.output_path + f"_results_{args.direction}.pt"
    queries_path = args.output_path + f"_queries_{args.direction}.pt"

    torch.save(results_dict, results_path)
    torch.save(unified_query_dict, queries_path)

    print(f"\nSaved results to:")
    print(f"  {results_path}")
    print(f"  {queries_path}")


if __name__ == "__main__":
    main()
