#!/usr/bin/env python3
"""
Mirror Swapping Experiment for Video Models (Figure 10a)

This script performs mirror swapping on videos - swapping activations between
normal and time-reversed videos to measure temporal information transfer
across layers.
"""

import argparse
import json
import os
import warnings
from pathlib import Path

import torch
import numpy as np
from decord import VideoReader, cpu

warnings.filterwarnings("ignore")


def load_video(video_path, max_frames_num, fps=1, force_sample=False):
    """Load and sample frames from a video."""
    if max_frames_num == 0:
        return np.zeros((1, 336, 336, 3))

    vr = VideoReader(video_path, ctx=cpu(0), num_threads=1)
    total_frame_num = len(vr)
    video_time = total_frame_num / vr.get_avg_fps()
    fps = round(vr.get_avg_fps() / fps)
    frame_idx = [i for i in range(0, len(vr), fps)]
    frame_time = [i / fps for i in frame_idx]

    if len(frame_idx) > max_frames_num or force_sample:
        sample_fps = max_frames_num
        uniform_sampled_frames = np.linspace(0, total_frame_num - 1, sample_fps, dtype=int)
        frame_idx = uniform_sampled_frames.tolist()
        frame_time = [i / vr.get_avg_fps() for i in frame_idx]

    frame_time = ",".join([f"{i:.2f}s" for i in frame_time])
    spare_frames = vr.get_batch(frame_idx).asnumpy()

    return spare_frames, frame_time, video_time


def make_input_from_path(video_path, question_part, image_processor, tokenizer, device, max_frames_num=8):
    """Create model inputs from a video path and question."""
    from llava.mm_utils import tokenizer_image_token
    from llava.constants import IMAGE_TOKEN_INDEX, DEFAULT_IMAGE_TOKEN
    from llava.conversation import conv_templates
    import copy

    video, frame_time, video_time = load_video(video_path, max_frames_num, 1, force_sample=True)
    video = image_processor.preprocess(video, return_tensors="pt")["pixel_values"].to(device).bfloat16()

    video = [video]
    conv_template = "qwen_1_5"
    time_instruction = f"The video lasts for {video_time:.2f} seconds, and {len(video[0])} frames are uniformly sampled from it. These frames are located at {frame_time}. Please answer the following questions related to this video."
    question = DEFAULT_IMAGE_TOKEN + f"{time_instruction}\n{question_part}"
    conv = copy.deepcopy(conv_templates[conv_template])
    conv.append_message(conv.roles[0], question)
    conv.append_message(conv.roles[1], None)
    prompt_question = conv.get_prompt()

    input_ids = tokenizer_image_token(prompt_question, tokenizer, IMAGE_TOKEN_INDEX, return_tensors="pt").unsqueeze(0).to(device)

    control_inputs = {
        "input_ids": input_ids,
        "images": video,
        "modalities": ["video"]
    }

    return control_inputs


def cache_resid_post(storage_dict):
    """Hook to cache residual post-layer activations."""
    def hook(module, inputs, outputs):
        if isinstance(outputs, tuple):
            hidden_states = outputs[0]
        else:
            hidden_states = outputs
        storage_dict["resid_post"] = hidden_states.detach()
        return outputs
    return hook


def patch_resid_post(intervene_resid_post):
    """Hook to patch residual post-layer activations."""
    def hook(module, inputs, outputs):
        if isinstance(outputs, tuple):
            return (intervene_resid_post,)
        else:
            return intervene_resid_post
    return hook


def run_patch_resid_post_video_on_single(
    model,
    control_inputs,
    intervene_inputs,
    layer,
    target_pos_added,
):
    """
    Run activation patching at a specific layer and token positions.

    Args:
        model: The LLaVA-Video model
        control_inputs: Inputs for the normal (forward) video
        intervene_inputs: Inputs for the intervene (reversed) video
        layer: Layer index to patch at
        target_pos_added: List of token positions to swap

    Returns:
        Tuple of (control_logits, intervene_logits, patched_logits)
    """
    # Step 1: Cache control activations
    control_cache = {}
    hook_control = model.model.layers[layer].register_forward_hook(
        cache_resid_post(control_cache)
    )
    with torch.no_grad():
        control_logits = model(
            input_ids=control_inputs["input_ids"],
            images=control_inputs["images"],
            modalities=control_inputs["modalities"],
            use_cache=False
        ).logits
    hook_control.remove()

    # Step 2: Cache intervene activations
    intervene_cache = {}
    hook_intervene = model.model.layers[layer].register_forward_hook(
        cache_resid_post(intervene_cache)
    )
    with torch.no_grad():
        intervene_logits = model(
            input_ids=intervene_inputs["input_ids"],
            images=intervene_inputs["images"],
            modalities=intervene_inputs["modalities"],
            use_cache=False
        ).logits
    hook_intervene.remove()

    # Step 3: Patch and run forward pass
    patched_cache = control_cache["resid_post"].clone()
    patched_cache[0, target_pos_added] = intervene_cache["resid_post"][0, target_pos_added]

    hook_patch = model.model.layers[layer].register_forward_hook(
        patch_resid_post(patched_cache)
    )
    with torch.no_grad():
        patched_logits = model(
            input_ids=control_inputs["input_ids"],
            images=control_inputs["images"],
            modalities=control_inputs["modalities"],
            use_cache=False
        ).logits
    hook_patch.remove()

    return control_logits, intervene_logits, patched_logits


def compute_swap_effects(
    model,
    control_inputs,
    intervene_inputs,
    target_pos_added,
    consider_layers,
    before_token_id=24885,
    after_token_id=46893,
):
    """
    Compute the effect of swapping activations on before/after token probabilities.

    Returns:
        Dict mapping layer -> (delta_before, delta_after)
    """
    final_probs = {}

    for layer in consider_layers:
        original_logits, intervene_logits, patched_logits = run_patch_resid_post_video_on_single(
            model, control_inputs, intervene_inputs, layer, target_pos_added
        )

        probs_1 = torch.nn.functional.log_softmax(original_logits[0, -1], dim=-1)
        probs_2 = torch.nn.functional.log_softmax(intervene_logits[0, -1], dim=-1)
        probs_patched = torch.nn.functional.log_softmax(patched_logits[0, -1], dim=-1)

        final_probs[layer] = (
            (probs_1[before_token_id] - probs_patched[before_token_id])
            / (probs_1[before_token_id] - probs_2[before_token_id]),
            (probs_1[after_token_id] - probs_patched[after_token_id])
            / (probs_1[after_token_id] - probs_2[after_token_id]),
        )

    return final_probs


def main():
    parser = argparse.ArgumentParser(description="Mirror swapping experiment for temporal video models")
    parser.add_argument("--manifest_path", type=str, required=True,
                        help="Path to MVBench scene queries JSON manifest")
    parser.add_argument("--video_dir", type=str, required=True,
                        help="Directory containing video files")
    parser.add_argument("--output_path", type=str, required=True,
                        help="Path to save output results (.pt file)")
    parser.add_argument("--model_name", type=str, default="lmms-lab/LLaVA-Video-7B-Qwen2",
                        help="HuggingFace model name for LLaVA-Video")
    parser.add_argument("--max_samples", type=int, default=200,
                        help="Maximum number of videos to process")
    parser.add_argument("--num_layers", type=int, default=28,
                        help="Number of transformer layers to analyze")
    parser.add_argument("--max_frames", type=int, default=8,
                        help="Number of frames to sample from each video")
    parser.add_argument("--device", type=str, default="cuda",
                        help="Device to run on (cuda/cpu)")

    args = parser.parse_args()

    print("="*60)
    print("Mirror Swapping Experiment - Temporal IDs")
    print("="*60)
    print(f"Model: {args.model_name}")
    print(f"Manifest: {args.manifest_path}")
    print(f"Video directory: {args.video_dir}")
    print(f"Max samples: {args.max_samples}")
    print(f"Output: {args.output_path}")
    print("="*60)

    # Load model
    print("\nLoading model...")
    from llava.model.builder import load_pretrained_model

    tokenizer, model, image_processor, max_length = load_pretrained_model(
        args.model_name, None, "llava_qwen",
        torch_dtype="bfloat16", device_map="auto",
        attn_implementation="eager"
    )
    model.eval()
    print("✓ Model loaded")

    # Load manifest
    print(f"\nLoading manifest from {args.manifest_path}...")
    with open(args.manifest_path, "r") as f:
        entries = json.load(f)
    print(f"✓ Loaded {len(entries)} entries")

    # Process videos
    big_dict = {}
    consider_layers = list(range(args.num_layers))

    # Get length of image and text tokens for position calculations
    print("\nDetermining token lengths...")
    sample_video_path = os.path.join(args.video_dir, entries[0]["video_path"])
    sample_query = entries[0]["query"].replace(" before or after the ", "earlier or later than ") + " Answer in one word."
    sample_inputs = make_input_from_path(sample_video_path, sample_query, image_processor, tokenizer, args.device, args.max_frames)
    len_img_toks = 1679  # Fixed for LLaVA-Video
    len_text_toks = sample_inputs["input_ids"].shape[1]
    print(f"✓ Image tokens: {len_img_toks}, Text tokens: {len_text_toks}")

    print(f"\nProcessing videos (cap at {args.max_samples})...")
    from tqdm import tqdm
    for ii, entry in enumerate(tqdm(entries)):
        if ii >= args.max_samples:
            break

        video_filename = entry["video_path"]
        video_path = os.path.join(args.video_dir, video_filename)

        if not os.path.exists(video_path):
            print(f"⚠️  Skipping {video_filename} - file not found")
            continue

        scene1 = entry["scene1"]
        scene2 = entry["scene2"]
        query = entry["query"].replace(" before or after the ", "earlier or later than ") + " Answer in one word."

        # Compute ground truth
        correct_order = entry["correct_order"]
        query_order = entry["query_order"]
        true_before = (query_order == correct_order)

        try:
            # Create normal (forward) video inputs
            control_inputs = make_input_from_path(video_path, query, image_processor, tokenizer, args.device, args.max_frames)

            # Create reversed video inputs
            intervene_inputs = {
                "input_ids": control_inputs["input_ids"],
                "images": [torch.stack([control_inputs["images"][0][-ee] for ee in range(1, args.max_frames + 1)])],
                "modalities": control_inputs["modalities"]
            }

            big_dict[(Path(video_path).name, true_before)] = {}

            # Test three modalities: text tokens, image patches, object word tokens
            for modality in ["text", "image", "text-objonly"]:
                if modality == "text":
                    # Swap all text tokens
                    target_pos_added = [-pp for pp in range(1, len_text_toks + 1)]
                elif modality == "image":
                    # Swap all image patches
                    target_pos_added = [ww for ww in range(len_img_toks)]
                elif modality == "text-objonly":
                    # Swap only object word tokens (scene1 and scene2)
                    startind = -20 - len(scene2.split(" "))
                    target_pos_added = [startind - kk for kk in range(len(scene1.split(" ")))]
                    other_startind = -12
                    target_pos_added += [other_startind - kk for kk in range(len(scene2.split(" ")))]

                output_dict = compute_swap_effects(
                    model,
                    control_inputs,
                    intervene_inputs,
                    target_pos_added,
                    consider_layers
                )
                big_dict[(Path(video_path).name, true_before)][modality] = output_dict

            if (ii + 1) % 10 == 0:
                print(f"  Processed {ii + 1}/{min(args.max_samples, len(entries))} videos")
                # Save checkpoint
                torch.save(big_dict, args.output_path)

        except Exception as e:
            print(f"⚠️  Error processing {video_filename}: {e}")
            continue

    # Save final results
    torch.save(big_dict, args.output_path)
    print(f"\n✓ Results saved to {args.output_path}")
    print(f"✓ Processed {len(big_dict)} videos across {len(consider_layers)} layers")


if __name__ == "__main__":
    main()
