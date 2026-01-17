import os
import random
import argparse
from PIL import Image
from os import path


def paste_two_objects_on_grid(
    a_path,
    b_path,
    object_wh,
    cell_wh,
    output_dir,
    no_same_row_or_col,
    bg_dir=None,
    local_variation=False,
):
    assert object_wh <= cell_wh

    a_img = Image.open(a_path)
    b_img = Image.open(b_path)
    grid_coords = [(x * cell_wh, y * cell_wh) for x in range(4) for y in range(4)]
    canvas_size = (cell_wh * 4, cell_wh * 4)

    if bg_dir is not None:
        bg_files = [
            f
            for f in os.listdir(bg_dir)
            if f.lower().endswith((".png", ".jpg", ".jpeg"))
        ]
        bg_path = path.join(bg_dir, random.choice(bg_files))
        bg = Image.open(bg_path).convert("RGBA").resize((cell_wh * 4, cell_wh * 4))

    for a_x, a_y in grid_coords:
        for b_x, b_y in grid_coords:
            if no_same_row_or_col and (a_x == b_x or a_y == b_y):
                continue
            if (a_x, a_y) == (b_x, b_y):
                continue

            # prepare A
            if local_variation:
                # size ±5%
                factor = random.uniform(0.95, 1.05)
                w_a = h_a = int(object_wh * factor)
                temp_a = a_img.resize((w_a, h_a), Image.LANCZOS)
                # rotation ±15°
                angle = random.uniform(-15, 15)
                temp_a = temp_a.rotate(angle, expand=True)
                ow_a, oh_a = temp_a.size
            else:
                temp_a = a_img.resize((object_wh, object_wh), Image.LANCZOS)
                ow_a = oh_a = object_wh

            # prepare B
            if local_variation:
                factor = random.uniform(0.9, 1.1)
                w_b = h_b = int(object_wh * factor)
                temp_b = b_img.resize((w_b, h_b), Image.LANCZOS)
                angle = random.uniform(-15, 15)
                temp_b = temp_b.rotate(angle, expand=True)
                ow_b, oh_b = temp_b.size
            else:
                temp_b = b_img.resize((object_wh, object_wh), Image.LANCZOS)
                ow_b = oh_b = object_wh

            # center in cells
            ax = int(a_x + (cell_wh - ow_a) / 2)
            ay = int(a_y + (cell_wh - oh_a) / 2)
            bx = int(b_x + (cell_wh - ow_b) / 2)
            by = int(b_y + (cell_wh - oh_b) / 2)

            if bg_dir is not None:
                # paste onto background (using each object's alpha as mask)
                canvas = bg.copy()
            else:
                canvas = Image.new("RGBA", canvas_size, (0, 0, 0, 0))
            canvas.paste(temp_a, (ax, ay), temp_a)
            canvas.paste(temp_b, (bx, by), temp_b)

            row_dir = f"{int(a_x/cell_wh)}_{int(a_y/cell_wh)}"
            os.makedirs(path.join(output_dir, row_dir), exist_ok=True)
            output_path = path.join(
                output_dir,
                row_dir,
                f"{int(b_x/cell_wh)}_{int(b_y/cell_wh)}_{object_wh}.png",
            )
            canvas.save(output_path)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Place two resized objects onto different cells of a 4×4 grid, with optional local variation."
    )
    parser.add_argument(
        "a_path", type=str, help="Path to the first object image (object A)."
    )
    parser.add_argument(
        "b_path", type=str, help="Path to the second object image (object B)."
    )
    parser.add_argument(
        "output_dir",
        type=str,
        help="Directory where the generated grid images will be saved.",
    )
    parser.add_argument(
        "--cell_wh",
        type=int,
        default=224,
        help="Width and height (in pixels) of each grid cell (default: %(default)s).",
    )
    parser.add_argument(
        "--object_wh",
        type=int,
        default=224,
        help="Base width and height (in pixels) to resize each object before placing (default: %(default)s).",
    )
    parser.add_argument(
        "--no_same_row_or_col",
        action="store_true",
        help=(
            "If set, prevent the two objects from ever sharing the same row or column "
            "in the grid."
        ),
    )
    parser.add_argument(
        "--bg_dir",
        default=None,
        type=str,
        help="If provided, use a random background.",
    )
    parser.add_argument(
        "--local_variation",
        action="store_true",
        help="If set, apply random size (±5%) and rotation (±15°) to each object per placement.",
    )

    args = parser.parse_args()

    os.makedirs(args.output_dir, exist_ok=True)
    paste_two_objects_on_grid(
        args.a_path,
        args.b_path,
        args.object_wh,
        args.cell_wh,
        args.output_dir,
        args.no_same_row_or_col,
        args.bg_dir,
        args.local_variation,
    )
