import re
import os
import csv
from PIL import Image
import numpy as np
from itertools import product


def generate_non_overlapping_pairs():
    grid_range = range(4)
    positions = list(
        product(grid_range, grid_range)
    )  # all (row, col) pairs in 4x4 grid

    valid_pairs = []
    for w, z in positions:
        for x, y in positions:
            if (w, z) == (x, y):  # same position, skip
                continue
            if w == x:  # same row, skip
                continue
            valid_pairs.append(((w, z), (x, y)))

    return valid_pairs


def extract_subject_object(caption, direction_choices={"left", "right"}):
    """
    Given a caption like:
    "A photo of a dog to the left of a person"
    or
    "A photo of an potted plant to the right of an umbrella"

    Return: ("potted plant", "umbrella", "right")
    """
    # Updated regex:
    # - Allow "a" or "an" (case-insensitive)
    # - Capture 1–3 word noun phrases for subject and object (you can adjust upper bound as needed)
    # - Allow flexible whitespace
    if "left" in direction_choices:
        pattern = r"A photo of a[n]? ([a-z]+(?: [a-z]+){0,2}) to the (left|right) of a[n]? ([a-z]+(?: [a-z]+){0,2})"
    elif "front" in direction_choices:
        pattern = r"A photo of a[n]? ([a-z]+(?: [a-z]+){0,2}) (above|below) a[n]? ([a-z]+(?: [a-z]+){0,2})"
    elif "above" in direction_choices:
        pattern = r"A photo of a[n]? ([a-z]+(?: [a-z]+){0,2}) (above|below) a[n]? ([a-z]+(?: [a-z]+){0,2})"
    else:
        raise ValueError(
            f"Direction choices must be left or right: {direction_choices}"
        )

    match = re.match(pattern, caption.strip(), re.IGNORECASE)
    if match:
        subject, direction, obj = match.groups()
        return subject.lower(), obj.lower(), direction.lower()
    else:

        raise ValueError(f"Caption format unexpected: {caption}")


def extract_subject_object_old(caption):
    """
    Given a caption like:
    "A photo of a dog to the left of a person"
    or
    "A photo of an potted plant to the right of an umbrella"

    Return: ("potted plant", "umbrella", "right")
    """
    # Updated regex:
    # - Allow "a" or "an" (case-insensitive)
    # - Capture 1–3 word noun phrases for subject and object (you can adjust upper bound as needed)
    # - Allow flexible whitespace
    pattern = r"A photo of a[n]? ([a-z]+(?: [a-z]+){0,2}) to the (left|right) of a[n]? ([a-z]+(?: [a-z]+){0,2})"

    match = re.match(pattern, caption.strip(), re.IGNORECASE)
    if match:
        subject, direction, obj = match.groups()
        return subject.lower(), obj.lower(), direction.lower()
    else:
        raise ValueError(f"Caption format unexpected: {caption}")


def make_unified_query_old(caption1, caption2):
    """
    Given two opposing captions, return the unified query string.
    Assumes the captions only differ in the spatial relation.
    """
    subj1, obj1, dir1 = extract_subject_object(caption1)
    subj2, obj2, dir2 = extract_subject_object(caption2)

    assert subj1 == subj2, "Subjects do not match"
    assert obj1 == obj2, "Objects do not match"
    assert {dir1, dir2} == {"left", "right"}, "Directions must be left and right"

    # Optionally randomize order: here we use left-right ordering
    return f"Is the {subj1} to the left or right of the {obj1}?", subj1, obj1


def make_unified_query(caption1, caption2, direction_choices={"left", "right"}):
    """
    Given two opposing captions, return the unified query string.
    Assumes the captions only differ in the spatial relation.
    """
    subj1, obj1, dir1 = extract_subject_object(caption1, direction_choices)
    subj2, obj2, dir2 = extract_subject_object(caption2, direction_choices)

    assert subj1 == subj2, "Subjects do not match"
    assert obj1 == obj2, "Objects do not match"
    # assert {dir1, dir2} == direction_choices, "Directions must be left and right"
    if "left" in direction_choices:
        out = f"Is the {subj1} to the left or right of the {obj1}?"
    elif "front" in direction_choices:
        out = f"Is the {subj1} to the front or behind of the {obj1}?"
    elif "above" in direction_choices:
        out = f"Is the {subj1} above or below the {obj1}?"
    else:
        raise ValueError(
            f"Direction choices must be left or right: {direction_choices}"
        )

    # Optionally randomize order: here we use left-right ordering
    return out, subj1, obj1


def log_probs_to_csv(
    csv_path, patch_type, layer, left1, left2, left_patch, right1, right2, right_patch
):
    header = [
        "patch_type",
        "layer",
        "left_img1",
        "left_img2",
        "left_img1_patched",
        "right_img1",
        "right_img2",
        "right_img1_patched",
    ]
    row = [
        patch_type,
        layer,
        left1,
        left2,
        left_patch,
        right1,
        right2,
        right_patch,
    ]

    write_header = not os.path.exists(csv_path)
    with open(csv_path, "a", newline="") as f:
        writer = csv.writer(f)
        if write_header:
            writer.writerow(header)
        writer.writerow(row)


def extract_locs(s):
    # Match pattern like loc_1-2_3-4
    match = re.search(r"loc_(\d+)-(\d+)_(\d+)-(\d+)", s)
    if not match:
        raise ValueError("No loc_*_*_*_* pattern found in the string.")

    # Extract and convert to integers
    w, z, x, y = map(int, match.groups())
    return w, z, x, y


def make_white_background(image_path):
    """Loads an image with a transparent background and makes the background white."""
    try:
        img = Image.open(image_path)
        if img.mode == "RGBA":
            # Create a white background image
            background = Image.new("RGB", img.size, (255, 255, 255))
            # Paste the image onto the background, only where the image is not transparent
            background.paste(img, mask=img.split()[3])  # 3 is the alpha channel
            return background
        elif img.mode == "RGB":
            return img  # Image already has a non-transparent background
        else:
            print(
                f"Warning: Image mode {img.mode} is not supported. Returning the original image."
            )
            return img
    except FileNotFoundError:
        print(f"Error: Image file not found at {image_path}")
        return None


def make_noise_image(image_control):
    width, height = image_control.size
    mode = image_control.mode

    if mode == "RGB":
        noise = np.random.randint(0, 256, (height, width, 3), dtype=np.uint8)
    elif mode == "L":
        noise = np.random.randint(0, 256, (height, width), dtype=np.uint8)
    elif mode == "RGBA":
        noise = np.random.randint(0, 256, (height, width, 4), dtype=np.uint8)
    else:
        # Default to RGB if mode is unsupported
        noise = np.random.randint(0, 256, (height, width, 3), dtype=np.uint8)
        mode = "RGB"

    noise_img = Image.fromarray(noise, mode)
    return noise_img