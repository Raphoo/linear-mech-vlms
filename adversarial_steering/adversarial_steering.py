import sys
import os
project_root = os.path.abspath("..")
sys.path.insert(0, project_root)

import json
from os import path
from utils.coco import make_unified_query
from utils.extract_embeds import VLMForExtraction
from PIL import Image
import torch
import argparse
from tqdm import tqdm
import gc
import traceback


def run_coco(
    model: VLMForExtraction,
    image_path,
    pos_caption,
    neg_caption,
    id,
    intervention_layer,
    intervention_factor,
    universal_embed,
    reasoning,
    use_noise,
    control_dir,
):
    try:
        image = Image.open(image_path)
        query, object_a, object_b, pos, neg = make_unified_query(
            pos_caption, neg_caption, return_dir=True
        )

        tokenizer = model.processor.tokenizer
        if not reasoning:
            if "left or right" in query:
                query += ' Answer "left" or "right" only.'
            else:
                query += ' Answer "above" or "below" only.'

        target_tokens = [
            tokenizer.tokenize(" " + tw)[-1] for tw in [object_a, object_b]
        ]
        token_a, token_b = target_tokens
        object_a = object_a.replace(" ", "+")
        object_b = object_b.replace(" ", "+")

        base_dir = f"{object_a}_{object_b}_{pos}_{neg}_{id}"

        if intervention_layer != -1:
            control_case = path.join(control_dir, base_dir)
            control_sequences = torch.load(
                path.join(control_case, "sequences.pt"),
                weights_only=False,
                map_location="cpu",
            )
            control_embeds = torch.load(
                path.join(control_case, "embeds.pt"),
                weights_only=False,
                map_location="cpu",
            )
            if use_noise:
                steering_a = torch.randn_like(
                    next(
                        iter(universal_embed[intervention_layer]["universal"].values())
                    )
                )
                steering_b = torch.randn_like(
                    next(
                        iter(universal_embed[intervention_layer]["universal"].values())
                    )
                )
            else:
                response = (
                    tokenizer.decode(control_sequences[0])
                    .split("assistant")[-1]
                    .split("ASSISTANT")[-1]
                    .split("<start_of_turn>model")[-1]
                    .lower()
                )
                # print(universal_embed[intervention_layer].keys())
                # print(response)
                if "right" in response and "left" not in response:
                    steering_a = (
                        -universal_embed[intervention_layer]["universal"][(3, 0, 224)]
                        + universal_embed[intervention_layer]["universal"][(0, 0, 224)]
                    )
                    steering_b = (
                        -universal_embed[intervention_layer]["universal"][(0, 0, 224)]
                        + universal_embed[intervention_layer]["universal"][(3, 0, 224)]
                    )
                elif "left" in response and "right" not in response:
                    steering_a = (
                        -universal_embed[intervention_layer]["universal"][(0, 0, 224)]
                        + universal_embed[intervention_layer]["universal"][(3, 0, 224)]
                    )
                    steering_b = (
                        -universal_embed[intervention_layer]["universal"][(3, 0, 224)]
                        + universal_embed[intervention_layer]["universal"][(0, 0, 224)]
                    )
                elif "above" in response and "below" not in response:
                    steering_a = (
                        -universal_embed[intervention_layer]["universal"][(0, 0, 224)]
                        + universal_embed[intervention_layer]["universal"][(0, 3, 224)]
                    )
                    steering_b = (
                        -universal_embed[intervention_layer]["universal"][(0, 3, 224)]
                        + universal_embed[intervention_layer]["universal"][(0, 0, 224)]
                    )
                elif "below" in response and "above" not in response:
                    steering_a = (
                        -universal_embed[intervention_layer]["universal"][(0, 3, 224)]
                        + universal_embed[intervention_layer]["universal"][(0, 0, 224)]
                    )
                    steering_b = (
                        -universal_embed[intervention_layer]["universal"][(0, 0, 224)]
                        + universal_embed[intervention_layer]["universal"][(0, 3, 224)]
                    )
                else:
                    # print("brah")
                    return None, None, None, None

            norm_a = (
                control_embeds[intervention_layer][target_tokens[0]].norm(p=2)
                * intervention_factor
            )
            norm_b = (
                control_embeds[intervention_layer][target_tokens[1]].norm(p=2)
                * intervention_factor
            )

            def intervention_a(activations):
                # scale steering_a to norm
                scaling = norm_a / steering_a.norm(p=2)
                steering = steering_a * scaling
                return activations + steering

            def intervention_b(activations):
                scaling = norm_b / steering_b.norm(p=2)
                steering = steering_b * scaling
                return activations + steering

            intervention_dict = {
                intervention_layer: [
                    (target_tokens[0], 0, intervention_a),
                    (target_tokens[1], 0, intervention_b),
                ]
            }
        else:
            intervention_dict = {}

        embeds, outputs = model.extract_embeds(
            query,
            image,
            target_tokens,
            last_only=False,
            num_generate=250,
            interventions=intervention_dict,
            return_outputs=True,
        )
        text_embeds = model.extract_embeds(query, None, target_tokens)
        return embeds, text_embeds, outputs, base_dir

    except Exception as e:
        traceback.print_exc()
        return None, None, None, None


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Run COCO")
    parser.add_argument(
        "image_dir",
        type=str,
        help="Directory containing input PNG images",
    )
    parser.add_argument(
        "output_dir",
        type=str,
        help="Directory to save the output embeddings",
    )
    parser.add_argument(
        "metadata_path",
        type=str,
        help="Path to the filtered metadata",
    )
    parser.add_argument(
        "--reasoning",
        action="store_true",
        help="Output reasoning trace or not"
    )
    parser.add_argument(
        "--use_noise", action="store_true", help="Use noise for intervention"
    )
    parser.add_argument(
        "--model",
        type=str,
        default="llama",
        help="Model to use for extraction",
    )
    parser.add_argument(
        "--intervention_layer",
        type=int,
        default=-1,
        help="Layer to apply intervention",
    )
    parser.add_argument(
        "--intervention_factor",
        type=float,
        default=0.5,
        help="Intervention factor",
    )
    parser.add_argument(
        "--intervention_embed",
        type=str,
        help="Path to Universal ID, required for intervention",
    )
    parser.add_argument(
        "--control_dir",
        type=str,
        help="Path to the previous control run, required for intervention",
    )

    args = parser.parse_args()

    model = VLMForExtraction(model=args.model)
    with open(args.metadata_path, "r") as f:
        metadata = json.load(f)

    if args.intervention_layer != -1:
        assert args.control_dir is not None
        assert args.intervention_embed is not None
        assert args.intervention_factor is not None
        assert (
            args.reasoning is not None
        )  # reasoning with intervention is not implemented
        universal_embed = torch.load(
            args.intervention_embed, map_location="cuda", weights_only=False
        )
    else:
        coco = None
        universal_embed = None

    for id, pos_caption, neg_caption in tqdm(metadata, desc="Running COCO"):
        image_path = path.join(args.image_dir, f"{id:012d}.jpg")
        embeds, text_embeds, outputs, base_dir = run_coco(
            model,
            image_path,
            pos_caption,
            neg_caption,
            int(id),
            args.intervention_layer,
            args.intervention_factor,
            universal_embed,
            args.reasoning,
            args.use_noise,
            args.control_dir,
        )
        if embeds is None:
            continue
        if not path.isdir(path.join(args.output_dir, base_dir)):
            os.makedirs(path.join(args.output_dir, base_dir))

        embeds_name = "embeds"
        logits_name = "logits"
        sequences_name = "sequences"
        if args.intervention_layer != -1:
            if args.use_noise:
                embeds_name += f"_noise_{args.intervention_layer}"
                logits_name += f"_noise_{args.intervention_layer}"
                sequences_name += f"_noise_{args.intervention_layer}"
            else:
                embeds_name += f"_intervention_{args.intervention_layer}"
                logits_name += f"_intervention_{args.intervention_layer}"
                sequences_name += f"_intervention_{args.intervention_layer}"
        torch.save(
            embeds,
            path.join(
                args.output_dir,
                base_dir,
                embeds_name + ".pt",
            ),
        )
        torch.save(
            outputs.logits,
            path.join(
                args.output_dir,
                base_dir,
                logits_name + ".pt",
            ),
        )
        torch.save(
            outputs.sequences,
            path.join(
                args.output_dir,
                base_dir,
                sequences_name + ".pt",
            ),
        )
        torch.save(
            text_embeds,
            path.join(args.output_dir, base_dir, "text.pt"),
        )

        del text_embeds
        del outputs
        del embeds
        gc.collect()
