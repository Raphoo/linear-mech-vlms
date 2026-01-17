import json
import sys
import os
project_root = os.path.abspath("..")
sys.path.insert(0, project_root)
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
    reasoning,
):
    try:
        image = Image.open(image_path)
        query, object_a, object_b, pos, neg = make_unified_query(
            pos_caption, neg_caption, return_dir=True
        )

        tokenizer = model.processor.tokenizer
        if not reasoning:
            if "left or right" in query:
                query += " Answer left or right only."
            else:
                query += " Answer above or below only."

        target_tokens = [
            tokenizer.tokenize(" " + tw)[-1] for tw in [object_a, object_b]
        ]
        token_a, token_b = target_tokens
        object_a = object_a.replace(" ", "+")
        object_b = object_b.replace(" ", "+")

        base_dir = f"{object_a}_{object_b}_{pos}_{neg}_{id}"

        embeds, outputs = model.extract_embeds(
            query,
            image,
            target_tokens,
            last_only=False,
            num_generate=250,
            return_outputs=True,
        )
        text_embeds = model.extract_embeds(query, None, target_tokens)
        return embeds, text_embeds, outputs, base_dir

    except Exception as e:
        traceback.print_exc()
        return None, None, None, None


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Run COCO spatial reasoning extraction"
    )
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
        "--reasoning", action="store_true", help="Output reasoning trace or not"
    )
    parser.add_argument(
        "--model",
        type=str,
        default="llama",
        help="Model to use for extraction",
    )

    args = parser.parse_args()

    model = VLMForExtraction(model=args.model)
    with open(args.metadata_path, "r") as f:
        metadata = json.load(f)

    for id, pos_caption, neg_caption in tqdm(metadata, desc="Running COCO"):
        image_path = path.join(args.image_dir, f"{id:012d}.jpg")
        embeds, text_embeds, outputs, base_dir = run_coco(
            model,
            image_path,
            pos_caption,
            neg_caption,
            int(id),
            args.reasoning,
        )
        if embeds is None:
            continue
        if not path.isdir(path.join(args.output_dir, base_dir)):
            os.makedirs(path.join(args.output_dir, base_dir))

        torch.save(
            embeds,
            path.join(args.output_dir, base_dir, "embeds.pt"),
        )
        torch.save(
            outputs.logits,
            path.join(args.output_dir, base_dir, "logits.pt"),
        )
        torch.save(
            outputs.sequences,
            path.join(args.output_dir, base_dir, "sequences.pt"),
        )
        torch.save(
            text_embeds,
            path.join(args.output_dir, base_dir, "text.pt"),
        )

        del text_embeds
        del outputs
        del embeds
        gc.collect()
