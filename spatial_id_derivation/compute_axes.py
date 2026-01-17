import torch
import argparse


def compute_xy(embed_path, output_path, size=224, grid_wh=4):
    embeds = torch.load(embed_path, map_location="cpu", weights_only=False)

    x_axis = {}
    y_axis = {}
    for layer, layer_embed in embeds.items():
        x_axis[layer] = {}
        y_axis[layer] = {}
        for token, embed in layer_embed.items():
            x_axis[layer][token] = []
            y_axis[layer][token] = []
            for x in range(grid_wh):
                for y_1 in range(grid_wh - 1):
                    for y_2 in range(y_1, grid_wh):
                        y_axis[layer][token].append(
                            embed[(x, y_2, size)] - embed[(x, y_1, size)]
                        )

            y_axis[layer][token] = torch.mean(
                torch.stack(list(y_axis[layer][token])), dim=0
            )

            for y in range(grid_wh):
                for x_1 in range(grid_wh - 1):
                    for x_2 in range(x_1, grid_wh):
                        x_axis[layer][token].append(
                            embed[(x_2, y, size)] - embed[(x_1, y, size)]
                        )

            x_axis[layer][token] = torch.mean(
                torch.stack(list(x_axis[layer][token])), dim=0
            )

    torch.save(x_axis, f"{output_path}_x.pt")
    torch.save(y_axis, f"{output_path}_y.pt")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Compute average x/y directional embeddings from spatial embeddings"
    )
    parser.add_argument(
        "embed_path", type=str, help="Path to the input embeddings .pt file"
    )
    parser.add_argument(
        "output_path",
        type=str,
        help="Prefix for output files (suffix '_x.pt' and '_y.pt' will be appended)",
    )
    parser.add_argument(
        "--size",
        type=int,
        default=224,
        help="Size dimension key used in embeddings (default: 224)",
    )
    parser.add_argument(
        "--grid_wh", type=int, default=4, help="Grid width/height (default: 4)"
    )

    args = parser.parse_args()
    compute_xy(
        args.embed_path,
        args.output_path,
        size=args.size,
        grid_wh=args.grid_wh,
    )
