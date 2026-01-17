import numpy as np
import colorsys
from matplotlib.cm import get_cmap


def bivariate_color_map(x, y, x_max, y_max, min_light=0.2, max_light=0.8):
    """
    Create RGBA colors for points classified by x-category and scaled by y-value.
    x_max is unused (kept for backwards compatibility).
    y_max: maximum y-value for normalization.
    """
    # Define base colors in tab10: red, green, orange, blue
    cmap = get_cmap("tab10")
    # Tab10 indices: red=3, green=2, orange=1, blue=0
    base_indices = [3, 1, 2, 0]

    x_arr = np.asarray(x, dtype=int)
    y_arr = np.asarray(y, dtype=float)

    # Normalize y to get lightness for shading
    y_norm = np.clip(y_arr / y_max, 0, 1)
    lightness = (1 - y_norm) * (max_light - min_light) + min_light

    # Map x to our defined four colors
    x_indices = np.clip(x_arr, 0, len(base_indices) - 1)
    base_rgb = np.array([cmap(idx)[:3] for idx in base_indices])[x_indices]

    # Adjust lightness in HLS space and append alpha
    rgba_out = []
    for rgb, l in zip(base_rgb.reshape(-1, 3), lightness.flatten()):
        h, _, s = colorsys.rgb_to_hls(*rgb)
        r, g, b = colorsys.hls_to_rgb(h, l, s)
        rgba_out.append((r, g, b, 1.0))

    rgba_arr = np.array(rgba_out).reshape(x_arr.shape + (4,))
    return np.clip(rgba_arr, 0, 1)
