"""
Temporal ID Extraction with PCA (Figure 10b)
Extracts temporal IDs from synthetic videos and performs PCA projection.
"""

import argparse
import os
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

    control_inputs = {
        "input_ids": input_ids,
        "images": video,
        "modalities": ["video"]
    }

    return control_inputs


def extract_temporal_ids(model, tokenizer, image_processor, video_dir, obj_pairs):
    """
    Extract temporal IDs from videos where objects appear at different frames.

    Returns:
        big_embeds_dict: Dictionary of embeddings per object, layer, and frame
    """
    model_components = (tokenizer, model, image_processor)

    big_embeds_dict = {}

    for obj1, obj2 in obj_pairs:
        print(f"Processing object pair: {obj1}, {obj2}")

        # Initialize storage
        for obj in [obj1, obj2]:
            if obj not in big_embeds_dict:
                big_embeds_dict[obj] = {}
                for layer_num in range(29):
                    big_embeds_dict[obj][layer_num] = {}

        # Query template
        query_text = f"In this video, two objects appear at different times. Does the {obj1} come before or after the {obj2}? Answer in one word."

        # Token indices for object words in query (adjust based on tokenization)
        indices = [-18, -12]

        # Process videos with different frame appearances
        for obj1_frame in range(1, 8):
            for obj2_frame in range(1, 8):
                if obj1_frame != obj2_frame:
                    # Construct video filename
                    vid_name = f"{video_dir}/appear_{obj1}{obj1_frame}_{obj2}{obj2_frame}.mp4"

                    if not os.path.exists(vid_name):
                        continue

                    # Get model inputs
                    control_inputs = make_input_from_path(vid_name, query_text, model_components)

                    # Extract hidden states
                    with torch.no_grad():
                        control_out = model(
                            input_ids=control_inputs["input_ids"],
                            images=control_inputs["images"],
                            modalities=control_inputs["modalities"],
                            use_cache=False,
                            output_hidden_states=True
                        )

                    # Store embeddings for each layer (detach and move to CPU to prevent VRAM leak)
                    for layer_num in range(29):
                        big_embeds_dict[obj1][layer_num][obj1_frame] = control_out.hidden_states[layer_num][0, indices[0], :].detach().cpu()
                        big_embeds_dict[obj2][layer_num][obj2_frame] = control_out.hidden_states[layer_num][0, indices[1], :].detach().cpu()

                    # Free GPU memory
                    del control_out, control_inputs
                    torch.cuda.empty_cache()

    return big_embeds_dict


def compute_average_embeds(big_embeds_dict):
    """
    Compute average embeddings across all objects for each layer and frame.

    Returns:
        average_embeds_dict: Dictionary of average embeddings per layer and frame
    """
    average_embeds_dict = {}

    for layer_num in range(29):
        average_embeds_dict[layer_num] = {}

        for frame_num in range(1, 8):
            # Collect all vectors for this frame and layer
            all_vecs = []
            for obj in big_embeds_dict.keys():
                if frame_num in big_embeds_dict[obj][layer_num]:
                    all_vecs.append(big_embeds_dict[obj][layer_num][frame_num])

            if all_vecs:
                average_embeds_dict[layer_num][frame_num] = torch.mean(torch.stack(all_vecs), dim=0)

    return average_embeds_dict


def get_text_embeddings_from_query(model, tokenizer):
    """Get text embeddings for 'before' and 'after' from within a temporal query.

    Runs the full temporal query through the model (text-only, no video) and extracts
    embeddings at the token positions for "before" and "after".

    Returns:
        before_hidden: List of embeddings for "before" at each layer
        after_hidden: List of embeddings for "after" at each layer
    """
    # Use the same query format as temporal ID extraction
    query_text = "In this video, two objects appear at different times. Does the circle come before or after the triangle? Answer in one word."

    # Create text-only input (no video)
    conv_template = "qwen_1_5"
    conv = copy.deepcopy(conv_templates[conv_template])
    conv.append_message(conv.roles[0], query_text)
    conv.append_message(conv.roles[1], None)
    prompt = conv.get_prompt()

    input_ids = tokenizer(prompt, return_tensors="pt").input_ids.to(model.device)

    with torch.no_grad():
        outputs = model(input_ids=input_ids, output_hidden_states=True)

    # Token positions for "before" and "after" in the query
    # "...come before or after the..." -> positions -16 and -14 from end
    before_pos = -16
    after_pos = -14

    before_hidden = []
    after_hidden = []

    for layer_output in outputs.hidden_states:
        # layer_output shape: (1, seq_len, hidden_dim)
        before_hidden.append(layer_output[0, before_pos, :].detach().cpu())
        after_hidden.append(layer_output[0, after_pos, :].detach().cpu())

    del outputs
    torch.cuda.empty_cache()

    return before_hidden, after_hidden


def main(args):
    # Setup paths
    embeds_dir = Path(args.embeds_dir)
    embeds_dir.mkdir(parents=True, exist_ok=True)

    # Load model
    print(f"Loading model: {args.model_name}")
    tokenizer, model, image_processor, max_length = load_pretrained_model(
        args.model_path, None, args.model_name,
        torch_dtype="bfloat16", device_map="auto",
        attn_implementation="eager"
    )
    model.eval()

    # Get object pairs from video directory
    video_dir = Path(args.video_dir)

    # Extract object pairs from filenames
    obj_pairs = []
    for video_file in video_dir.glob("appear_*.mp4"):
        filename = video_file.stem
        # Parse filename: appear_{obj1}{num1}_{obj2}{num2}
        parts = filename.replace("appear_", "").split("_")
        if len(parts) == 2:
            obj1 = ''.join([c for c in parts[0] if not c.isdigit()])
            obj2 = ''.join([c for c in parts[1] if not c.isdigit()])

            if obj1 and obj2 and (obj1, obj2) not in obj_pairs and (obj2, obj1) not in obj_pairs:
                obj_pairs.append((obj1, obj2))

    if not obj_pairs:
        # Use default pairs if none found
        obj_pairs = [("circle", "triangle"), ("square", "star")]

    obj_pairs = obj_pairs[:args.num_obj_pairs]
    print(f"Using {len(obj_pairs)} object pairs: {obj_pairs}")

    # Extract temporal IDs
    print("\nExtracting temporal IDs...")
    big_embeds_dict = extract_temporal_ids(model, tokenizer, image_processor, str(video_dir), obj_pairs)

    # Compute average embeddings
    print("\nComputing average embeddings...")
    average_embeds_dict = compute_average_embeds(big_embeds_dict)

    # Get text embeddings for "before" and "after" from query context
    print("\nExtracting text embeddings for 'before' and 'after'...")
    before_hidden, after_hidden = get_text_embeddings_from_query(model, tokenizer)

    # Save embeddings (including text embeddings for plotting)
    embeddings_file = embeds_dir / "temporal_ids.pt"
    torch.save({
        "big_embeds_dict": big_embeds_dict,
        "average_embeds_dict": average_embeds_dict
    }, embeddings_file)
    print(f"Embeddings saved to {embeddings_file}")

    # Save text embeddings separately for plotting
    text_embeds_file = embeds_dir / "text_embeddings.pt"
    torch.save({
        "before_hidden": before_hidden,
        "after_hidden": after_hidden
    }, text_embeds_file)
    print(f"Text embeddings saved to {text_embeds_file}")

    print("\nExtraction complete!")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Temporal ID Extraction and PCA")
    parser.add_argument("--model_path", type=str, default="lmms-lab/LLaVA-Video-7B-Qwen2",
                        help="Path to pretrained model")
    parser.add_argument("--model_name", type=str, default="llava_qwen",
                        help="Model name")
    parser.add_argument("--video_dir", type=str, required=True,
                        help="Directory containing synthetic videos")
    parser.add_argument("--embeds_dir", type=str, default="embeds",
                        help="Directory to save embeddings")
    parser.add_argument("--num_obj_pairs", type=int, default=5,
                        help="Number of object pairs to use")

    args = parser.parse_args()
    main(args)
