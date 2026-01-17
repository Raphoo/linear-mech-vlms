import os
import torch
import argparse

def compute_universal_id(embeds_dir, model):
    subdirs = os.listdir(embeds_dir)
    subdirs = [d for d in subdirs if os.path.isdir(os.path.join(embeds_dir, d))]

    avg_embeds = {}
    for subdir in subdirs:
        embeds = torch.load(os.path.join(embeds_dir, subdir, f"{model}.pt"), weights_only=True)
        for layer, layer_embed in embeds.items():
            if layer not in avg_embeds:
                avg_embeds[layer] = {}
            for token, token_embed in layer_embed.items():
                if "universal" not in avg_embeds[layer]:
                    avg_embeds[layer]["universal"] = {}
                for loc, embed in token_embed.items():
                    if loc not in avg_embeds[layer]["universal"]:
                        avg_embeds[layer]["universal"][loc] = []
                    avg_embeds[layer]["universal"][loc].append(embed)

    for layer, layer_embed in avg_embeds.items():
        for token, token_embed in layer_embed.items():
            for loc in token_embed.keys():
                avg_embeds[layer][token][loc] = torch.mean(torch.stack(avg_embeds[layer][token][loc]), dim=0)

    return avg_embeds

if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Compute univeral ID by averaging over spatial IDs"
    )
    parser.add_argument(
        "embeds_dir",
        type=str,
        help="Directory containing extracted spatial ID",
    )
    parser.add_argument(
        "output_path",
        type=str,
        help="Directory to save the output embeddings",
    )
    parser.add_argument(
        "--model",
        type=str,
        default="llama-13b",
        help="Model to use for extraction",
    )

    args = parser.parse_args()

    embeds = compute_universal_id(args.embeds_dir, args.model)
    torch.save(embeds, args.output_path)
