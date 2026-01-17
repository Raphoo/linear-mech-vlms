from collections import defaultdict
import sys
import os
project_root = os.path.abspath("..")
sys.path.insert(0, project_root)
from utils.extract_embeds import VLMForExtraction
import re
import os
from PIL import Image
import torch
import argparse
from tqdm import tqdm

def extract_spatial_id(model, query, target_words, image_dir):
    layer_indices = model.layer_indices

    embeds = {
        layer: {tw: defaultdict(list) for tw in target_words} for layer in layer_indices
    }

    pattern = re.compile(r"(\d+)_(\d+)_(\d+)\.png")

    for fname in os.listdir(image_dir):
        match = pattern.match(fname)
        if not match:
            continue

        x, y, size = map(int, match.groups())
        loc = (x, y, size)

        image_path = os.path.join(image_dir, fname)
        image = Image.open(image_path).convert("RGB")
        layerwise_embeds = model.extract_embeds(query, image, target_words)

        for layer in layer_indices:
            for token in target_words:
                embeds[layer][token][loc] = layerwise_embeds[layer][token]

    # mean-center each token's embeddings
    for layer in layer_indices:
        for token in target_words:
            vals = list(embeds[layer][token].values())
            if not vals:
                continue
            average_embed = torch.mean(torch.stack(vals), dim=0)
            for loc in embeds[layer][token]:
                embeds[layer][token][loc] -= average_embed

    return embeds

if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Extract spatial ID embeddings from images using a specified model, query, and target words"
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
        "query",
        type=str,
        help="Text query for embeddings extraction",
    )
    parser.add_argument(
        "target_words",
        type=str,
        help="Comma-separated list of target words (e.g. frog,cat,dog)",
    )
    parser.add_argument(
        "--model",
        type=str,
        default="llama-11b",
        help="Model to use for extraction",
    )

    args = parser.parse_args()

    # parse comma-separated list into Python list
    target_words = [tw.strip() for tw in args.target_words.split(",") if tw.strip()]

    model = VLMForExtraction(model=args.model)
    output_path = os.path.join(args.output_dir, f"{args.model}.pt")

    if "tokenizer" in model.processor.__dict__:
        target_words = [
            model.processor.tokenizer.tokenize(" " + tw)[-1] for tw in target_words
        ]
    else:
        target_words = [model.processor.tokenize(" " + tw)[-1] for tw in target_words]

    # run extraction
    first_item = os.listdir(args.image_dir)[0]
    if first_item.lower().endswith(".png"):
        embeds = extract_spatial_id(model, args.query, target_words, args.image_dir)
    else:
        # handle subdirectories
        embeds_dict_list = []
        for subdir in tqdm(os.listdir(args.image_dir), desc="Computing Spatial ID"):
            subdir_path = os.path.join(args.image_dir, subdir)
            if not os.path.isdir(subdir_path):
                continue
            embeds_dict_list.append(
                extract_spatial_id(model, args.query, target_words, subdir_path)
            )

        # average across directories
        average_embeds = {}
        for embeds_dict in embeds_dict_list:
            for layer in model.layer_indices:
                average_embeds.setdefault(layer, {})
                for token in target_words:
                    average_embeds[layer].setdefault(token, {})
                    for loc, embed in embeds_dict[layer][token].items():
                        average_embeds[layer][token].setdefault(loc, []).append(embed)

        for layer, toks in average_embeds.items():
            for token, locs in toks.items():
                for loc, embeds_list in locs.items():
                    average_embeds[layer][token][loc] = torch.stack(embeds_list).mean(
                        dim=0
                    )

        embeds = average_embeds

    # save to disk
    torch.save(embeds, output_path)
