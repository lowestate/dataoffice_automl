import numpy as np
from sklearn.metrics import silhouette_score, calinski_harabasz_score, pairwise_distances

def compute_overlap_penalty(*, X: np.ndarray, labels: np.ndarray) -> float:
    labels = np.asarray(labels)
    mask = labels != -1
    X = X[mask]
    labels = labels[mask]
    unique_labels = np.unique(labels)
    if len(unique_labels) < 2:
        return 1.0

    centroids = np.vstack([X[labels == lbl].mean(axis=0) for lbl in unique_labels])
    dists = np.linalg.norm(X[:, None, :] - centroids[None, :, :], axis=2)
    true_idx = np.array([np.where(unique_labels == lbl)[0][0] for lbl in labels])
    nearest_idx = np.argmin(dists, axis=1)
    misassigned_basic = np.mean(nearest_idx != true_idx)

    max_intra, min_inter = [], []
    for lbl in unique_labels:
        points = X[labels == lbl]
        others = X[labels != lbl]
        intra = np.max(pairwise_distances(points)) if len(points) > 1 else 0.0
        inter = np.min(pairwise_distances(points, others)) if len(others) > 0 else np.inf
        max_intra.append(intra)
        min_inter.append(inter)

    merge_penalty = np.mean([min(1.0, intra / inter) for intra, inter in zip(max_intra, min_inter) if inter > 1e-6])
    overlap_score = 1.0 - (0.5 * misassigned_basic + 0.5 * merge_penalty)
    return float(np.clip(overlap_score, 0.0, 1.0))

def compute_centroid_separation(*, X: np.ndarray, labels: np.ndarray) -> float:
    mask = labels != -1
    X = X[mask]
    labels = labels[mask]
    unique = np.unique(labels)
    if len(unique) < 2:
        return 0.0
    centroids = np.vstack([X[labels == l].mean(axis=0) for l in unique])
    dists = [np.linalg.norm(centroids[i] - centroids[j])
             for i in range(len(centroids)) for j in range(i + 1, len(centroids))]
    return float(np.tanh(np.mean(dists)))

def posthoc_estimate(
        X: np.ndarray,
        labels: np.ndarray,
        *,
        weights_formula: dict[str, float] | None = None,
        weights_multipliers: dict[str, float] | None = None,
        stability: float = 0.0,
        gap: float | None = None,
        n_clusters: int | None = None,
        strong_overlap_penalty: bool = False
    ) -> tuple[float, dict[str, dict[str, float]]]:

    labels = np.asarray(labels)
    mask = labels != -1
    noise_ratio = float(np.mean(labels == -1))
    if np.sum(mask) < 2 or len(np.unique(labels[mask])) < 2:
        return -1.0, {}

    if n_clusters is None:
        n_clusters = len(np.unique(labels[mask]))

    try:
        sil = float(silhouette_score(X[mask], labels[mask]))
    except Exception:
        sil = 0.0

    overlap_score = compute_overlap_penalty(X=X, labels=labels)

    try:
        ch = float(calinski_harabasz_score(X[mask], labels[mask]))
        ch = np.log1p(ch) / (np.log1p(ch) + 5.0)
    except Exception:
        ch = 0.0

    gap_score = float(np.tanh(gap / 3)) if gap is not None and not np.isnan(gap) else 0.0
    stability = float(np.clip(stability, 0.0, 10.0)) * 100

    k_penalty = 1.0 if n_clusters <= 10 else 10 / n_clusters
    noise_penalty = 1.0 - 0.25 * noise_ratio
    separation_penalty = 1.0 if sil > 0.2 else sil / 0.2
    merge_multiplier = overlap_score ** 3 if strong_overlap_penalty else overlap_score

    centroid_sep = compute_centroid_separation(X=X, labels=labels)

    # задаём веса формулы
    default_weights_formula = {
        "silhouette": 0.15,
        "calinski_harabasz": 0.10,
        "stability": 0.50,
        "gap_score": 0.05,  
        "overlap_score": 0.15,
        "centroid_sep": 0.05
    }
    weights_formula = weights_formula or default_weights_formula

    score = (
        weights_formula.get("silhouette", 0) * sil +
        weights_formula.get("calinski_harabasz", 0) * ch +
        weights_formula.get("stability", 0) * stability +
        weights_formula.get("gap_score", 0) * gap_score +
        weights_formula.get("overlap_score", 0) * overlap_score +
        weights_formula.get("centroid_sep", 0) * centroid_sep
    )

    # мультипликаторы
    default_weights_multipliers = {
        "noise_penalty": noise_penalty,
        "separation_penalty": separation_penalty,
        "k_penalty": k_penalty,
        "merge_multiplier": merge_multiplier
    }
    weights_multipliers = weights_multipliers or default_weights_multipliers
    score *= np.prod(list(weights_multipliers.values()))

    weights = {
        "formula": {
            "silhouette": sil,
            "calinski_harabasz": ch,
            "stability": stability,
            "gap_score": gap_score,
            "overlap_score": overlap_score,
            "centroid_sep": centroid_sep
        },
        "multipliers": weights_multipliers
    }
    return float(score), weights