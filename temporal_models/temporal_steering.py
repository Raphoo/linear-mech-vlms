"""
Temporal ID Steering Experiment (Figure 10c)
Tests controllability with arbitrary temporal IDs by steering model beliefs.
Outputs .pt file for plotting.
"""

import argparse
import os
import json
import copy
import warnings
from pathlib import Path

import torch
import numpy as np
from decord import VideoReader, cpu

from llava.model.builder import load_pretrained_model
from llava.mm_utils import tokenizer_image_token
from llava.constants import IMAGE_TOKEN_INDEX, DEFAULT_IMAGE_TOKEN
from llava.conversation import conv_templates

warnings.filterwarnings("ignore")


def load_video(video_path, max_frames_num, fps=1, force_sample=False):
    """Load and sample frames from video."""
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


def make_input_from_path(video_path, question_part, model_components):
    """Create model inputs from video path and question."""
    tokenizer, model, image_processor = model_components

    max_frames_num = 8
    video, frame_time, video_time = load_video(video_path, max_frames_num, 1, force_sample=True)
    video = image_processor.preprocess(video, return_tensors="pt")["pixel_values"].cuda().bfloat16()

    video = [video]
    conv_template = "qwen_1_5"
    time_instruction = f"The video lasts for {video_time:.2f} seconds, and {len(video[0])} frames are uniformly sampled from it. These frames are located at {frame_time}.Please answer the following questions related to this video."
    question = DEFAULT_IMAGE_TOKEN + f"{time_instruction}\n{question_part}"

    conv = copy.deepcopy(conv_templates[conv_template])
    conv.append_message(conv.roles[0], question)
    conv.append_message(conv.roles[1], None)
    prompt_question = conv.get_prompt()

    input_ids = tokenizer_image_token(prompt_question, tokenizer, IMAGE_TOKEN_INDEX, return_tensors="pt").unsqueeze(0).cuda()

    return {
        "input_ids": input_ids,
        "images": video,
        "modalities": ["video"]
    }


def cache_resid_post(storage_dict):
    """Hook to cache residual post-activation."""
    def hook(module, inputs, outputs):
        if isinstance(outputs, tuple):
            hidden_states = outputs[0]
        else:
            hidden_states = outputs
        storage_dict["resid_post"] = hidden_states.detach()
        return outputs
    return hook


def patch_resid_post(intervene_resid_post):
    """Hook to patch residual with intervention."""
    def hook(module, inputs, outputs):
        if isinstance(outputs, tuple):
            return (intervene_resid_post,)
        else:
            return intervene_resid_post
    return hook


def run_patch_resid_post_video_on_single(model, control_inputs, layer, target_pos_added,
                                          chosen_spatial_ID_obj1, chosen_spatial_ID_obj2, mult_factor):
    """
    Run patching with temporal IDs on single video.
    Matches the original notebook function exactly.
    """
    DEVICE = model.device

    # Cache control activations
    control_cache = {}
    hook_control = model.model.layers[layer].register_forward_hook(cache_resid_post(control_cache))

    with torch.no_grad():
        control_logits = model(
            input_ids=control_inputs["input_ids"],
            images=control_inputs["images"],
            modalities=control_inputs["modalities"],
            use_cache=False
        ).logits
    hook_control.remove()

    patched_cache = control_cache["resid_post"].clone()

    for ttt in target_pos_added:
        patched_cache[0, ttt] = (
            control_cache["resid_post"][0, ttt]
            + chosen_spatial_ID_obj1.to(DEVICE) * mult_factor
            - chosen_spatial_ID_obj2.to(DEVICE) * mult_factor
        )

    hook_patch = model.model.layers[layer].register_forward_hook(patch_resid_post(patched_cache))

    with torch.no_grad():
        patched_logits = model(
            input_ids=control_inputs["input_ids"],
            images=control_inputs["images"],
            modalities=control_inputs["modalities"],
            use_cache=False
        ).logits
    hook_patch.remove()

    del control_cache, patched_cache
    torch.cuda.empty_cache()

    return control_logits, patched_logits


def save_diff_patch(model, control_inputs, temporal_ids, consider_layers, mult_factor, id1, patch_locs):
    """
    Compute delta log probs for before/after tokens.
    Matches the original notebook function.
    """
    # Token IDs for before/after (from original notebook)
    cir_index_possibility = torch.tensor([10227])  # before
    tri_index_possibility = torch.tensor([6025])   # after

    final_leftright_probs = {}
    for lll in consider_layers:
        final_leftright_probs[lll] = {}

    for layer in consider_layers:
        chosen_spatial_ID_obj1 = temporal_ids[layer][id1]
        chosen_spatial_ID_obj2 = temporal_ids[layer][8 - id1]

        original_logits, patched_logits = run_patch_resid_post_video_on_single(
            model, control_inputs, layer, patch_locs,
            chosen_spatial_ID_obj1, chosen_spatial_ID_obj2, mult_factor
        )

        probs_1 = torch.nn.functional.log_softmax(original_logits[0, -1], dim=-1)
        cir_prob = torch.stack([probs_1[gl] for gl in cir_index_possibility]).max()
        tri_prob = torch.stack([probs_1[bp] for bp in tri_index_possibility]).max()

        probs_patched = torch.nn.functional.log_softmax(patched_logits[0, -1], dim=-1)
        cir_prob3 = torch.stack([probs_patched[gl] for gl in cir_index_possibility]).max()
        tri_prob3 = torch.stack([probs_patched[bp] for bp in tri_index_possibility]).max()

        final_leftright_probs[layer]['delta_before'] = cir_prob3 - cir_prob
        final_leftright_probs[layer]['delta_after'] = tri_prob3 - tri_prob

    return final_leftright_probs


def main(args):
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    # Load temporal IDs
    print(f"Loading temporal IDs from {args.temporal_ids_path}")
    temporal_data = torch.load(args.temporal_ids_path)
    temporal_ids = temporal_data["average_embeds_dict"]

    # Load model
    print(f"Loading model: {args.model_name}")
    tokenizer, model, image_processor, _ = load_pretrained_model(
        args.model_path, None, args.model_name,
        torch_dtype="bfloat16", device_map="auto",
        attn_implementation="eager"
    )
    model.eval()
    model_components = (tokenizer, model, image_processor)

    # Load JSON manifest
    print(f"Loading manifest from {args.manifest_path}")
    with open(args.manifest_path, "r") as f:
        entries = json.load(f)

    big_dict = {}
    already_seen = set()

    for ii, entry in enumerate(entries):
        if ii >= args.num_samples:
            break

        video_filename = entry["video_path"]
        video_path = os.path.join(args.video_dir, video_filename)
        scene1 = entry["scene1"]
        scene2 = entry["scene2"]
        query = entry["query"] + " Answer in one word."

        # Skip duplicates
        pair = tuple(sorted([scene1, scene2]))
        if pair in already_seen:
            continue
        already_seen.add(pair)

        # Compute ground truth order
        correct_order = entry["correct_order"]
        query_order = entry["query_order"]
        true_before = (query_order == correct_order)

        print(f"Processing {ii+1}/{min(len(entries), args.num_samples)}: {Path(video_path).name} (true_before={true_before})")

        try:
            control_inputs = make_input_from_path(video_path, query, model_components)
        except Exception as e:
            print(f"Skipping {video_path}: {e}")
            continue

        big_dict[(Path(video_path).name, true_before)] = {}

        # Compute target positions based on scene2 length (from original notebook)
        startind = -20 - len(scene2.split(" "))
        target_pos_added = [startind + kk for kk in range(len(scene2.split(" ")))]

        for id1 in range(1, 8):
            output_dict = save_diff_patch(
                model, control_inputs, temporal_ids,
                args.steering_layers,
                args.mult_factor,
                id1,
                target_pos_added
            )
            big_dict[(Path(video_path).name, true_before)][id1] = output_dict

        # Periodically save
        if ii % 10 == 0:
            torch.save(big_dict, output_dir / "steering_results.pt")

        del control_inputs
        torch.cuda.empty_cache()

    # Final save
    output_file = output_dir / "steering_results.pt"
    torch.save(big_dict, output_file)
    print(f"\nResults saved to {output_file}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Temporal ID Steering Experiment")
    parser.add_argument("--model_path", type=str, default="lmms-lab/LLaVA-Video-7B-Qwen2")
    parser.add_argument("--model_name", type=str, default="llava_qwen")
    parser.add_argument("--manifest_path", type=str, required=True,
                        help="Path to MVBench scene queries JSON manifest")
    parser.add_argument("--video_dir", type=str, required=True,
                        help="Directory containing the video files")
    parser.add_argument("--temporal_ids_path", type=str, required=True)
    parser.add_argument("--output_dir", type=str, default="data/temporal_steering")
    parser.add_argument("--num_samples", type=int, default=300)
    parser.add_argument("--steering_layers", type=int, nargs="+", default=[13])
    parser.add_argument("--mult_factor", type=float, default=5.0)

    args = parser.parse_args()
    main(args)
