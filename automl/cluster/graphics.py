import numpy as np
from typing_extensions import TypedDict, Any


class ClusterPlotResult(TypedDict):
    clusters: dict[str, dict[str, Any]]
    centroids: dict[str, list[float]]


import matplotlib.pyplot as plt  # type: ignore
import matplotlib.colors as mcolors  # type: ignore
import numpy as np

def generate_cluster_plot(
        X_umap: np.ndarray,
        labels: np.ndarray,
        is_hdbscan: bool = False
    ) -> ClusterPlotResult:

    unique_labels = np.unique(labels)
    n_clusters = len(unique_labels)
    clusters: dict[str, dict[str, Any]] = {}
    centroids: dict[str, list[float]] = {}

    # Генерируем цвета из colormap "tab20" (20 разных мягких цветов)
    cmap = plt.get_cmap("tab20")
    colors = [cmap(i % 20) for i in range(n_clusters)]

    for i, lbl in enumerate(unique_labels):

        cluster_points = X_umap[labels == lbl]
        cluster_name = f"cluster_{lbl}"

        # Определяем визуальные параметры
        if is_hdbscan and lbl == -1:
            color = "lightgray"
            size = 25
            alpha = 0.4
        else:
            color = mcolors.to_hex(colors[i])  # преобразуем в hex для фронта
            size = 35
            alpha = 0.8

        clusters[cluster_name] = {
            "x": cluster_points[:, 0].tolist(),
            "y": cluster_points[:, 1].tolist(),
            "color": color,
            "size": size,
            "alpha": alpha
        }

        centroid_umap = cluster_points.mean(axis=0)
        centroids[cluster_name] = centroid_umap.tolist()

    return {
        "clusters": clusters,
        "centroids": centroids
    }
