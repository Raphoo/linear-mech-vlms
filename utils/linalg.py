from sklearn.decomposition import PCA
import numpy as np


def do_pca(data, n_components=2):
    pca = PCA(n_components=n_components)
    projected = pca.fit_transform(data)
    return pca, projected


def project_with_pca(data, pca):
    return pca.transform(data)


def compute_projection_variance(X, pca, components):
    X_centered = X - pca.mean_

    X_proj = pca.transform(X)

    selected_components = pca.components_[components]
    selected_projection = X_proj[:, components]

    X_reconstructed = selected_projection @ selected_components

    total_var = np.var(X_centered, axis=0).sum()

    explained_var = np.var(X_reconstructed, axis=0).sum()

    return explained_var / total_var


def project_onto_plane(embed, x_axis, y_axis):
    # 1) Orthonormalize
    u = x_axis / np.linalg.norm(x_axis)
    y_minus = y_axis - np.dot(y_axis, u) * u
    v = y_minus / np.linalg.norm(y_minus)

    # 2) Scalar coordinates (one scalar per embed along each axis)
    c1 = embed.dot(u)  # -> (N,)
    c2 = embed.dot(v)  # -> (N,)

    # 3) Stack scalars into a (N,2) array
    coords_2d = np.stack([c1, c2], axis=1)  # -> (N, 2)

    # 4) Compute explained variance ratio
    #    variance along u plus variance along v,
    #    divided by total variance in embed
    explained_var = (c1.var() + c2.var()) / np.var(embed, axis=0).sum()

    return coords_2d, explained_var


def euclidean_distances(points, point):
    p = point.reshape(-1)
    return np.sqrt(np.sum((points - p) ** 2, axis=1))
