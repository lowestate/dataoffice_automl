import json
import numpy as np
import pandas as pd
from scipy import sparse
from umap import UMAP  # type: ignore
from sklearn.decomposition import PCA
from sklearn.decomposition import TruncatedSVD

from automl.config import ClusteringConfig

def detect_metric(X):
    # 1. Check for Sparsity: If it's a sparse matrix (from Hasher/OHE), 
    # Cosine is almost always better regardless of feature count.
    if sparse.issparse(X):
        return "cosine"
    
    # 2. Check Feature Count: 
    # Euclidean is fine for most tabular data up to ~50 features.
    # Above 50, the "Curse of Dimensionality" makes Euclidean distances 
    # look almost identical, so we switch to Cosine.
    if X.shape[1] > 50:
        return "cosine"
        
    return "euclidean"

def reduce_dimensionality(X, random_state):
    n_samples, n_features = X.shape
    # Определяем целевое кол-во компонент заранее
    target_comps = min(50, n_features - 1) if n_features > 1 else 1

    if sparse.issparse(X):
        reducer = TruncatedSVD(n_components=target_comps, random_state=random_state)
    else:
        # Сначала пробуем 95% вариации, но ограничиваем сверху через n_components
        # Если нужно жестко 50, лучше сразу ставить 50, чтобы избежать "дребезга"
        reducer = PCA(n_components=target_comps, svd_solver="full", random_state=random_state)
    
    X_red = reducer.fit_transform(X)
    return X_red, reducer

def build_cluster_spaces(X, random_state, config: ClusteringConfig):
    X_red, reducer = reduce_dimensionality(X, random_state)

    # Линейное пространство для кластеризации (PCA 95%)
    X_linear = X

    # Манифолд для визуализации
    n_samples = X_red.shape[0]
    n_neighbors = int(np.clip(np.sqrt(n_samples), 5, 50))
    metric = detect_metric(X_red)

    X_manifold = UMAP(
        n_components=2,   # сразу 2D для графика
        n_neighbors=n_neighbors,
        min_dist=config.umap_min_dist,
        metric=metric,
        random_state=random_state
    ).fit_transform(X_red)

    return {
        "linear": X_linear,        # используем для кластеризации DBSCAN/HDBSCAN/KMeans
        "manifold": X_manifold     # только для визуализации
    }, reducer

def safe_get_params(model):
    params = {}
    for k, v in model.get_params().items():
        try:
            json.dumps(v)
            params[k] = v
        except Exception:
            params[k] = str(v)
    return params

def compute_cluster_stats(df_numeric, labels):
    stats = {}
    df_numeric = pd.DataFrame(df_numeric)
    for lbl in np.unique(labels):
        cluster_df = df_numeric[labels == lbl]
        cluster_stats = {
            "mean": cluster_df.mean().to_dict(),  # type: ignore
            "median": cluster_df.median().to_dict(),  # type: ignore
            "mode": cluster_df.mode().iloc[0].to_dict() if not cluster_df.mode().empty else {},  # type: ignore
            "min": cluster_df.min().to_dict(),  # type: ignore
            "max": cluster_df.max().to_dict(),  # type: ignore
            "std": cluster_df.std().to_dict(),  # type: ignore
            "var": cluster_df.var().to_dict(),  # type: ignore
            "count": cluster_df.count().to_dict()  # type: ignore
        }
        stats[int(lbl)] = cluster_stats
    return stats

def save_clustering_report(
        *,
        training_result: dict,
        filepath: str
    ) -> None:

    models = training_result.get("models", {})

    lines: list[str] = []

    for i, (_, model_data) in enumerate(models.items(), start=1):
        # --- безопасный доступ ---
        name = model_data.get("model_name", "unknown_model")

        metrics = model_data.get("metrics", {})

        unified = metrics.get("unified_score", 0.0)
        sil = metrics.get("silhouette_score", 0.0)
        ch = metrics.get("calinski_harabasz_score", 0.0)
        gap = metrics.get("gap_statistics", None)

        # --- защита от None / NaN ---
        def safe_float(value):
            try:
                if value is None:
                    return None
                if isinstance(value, float) and np.isnan(value):
                    return None
                return float(value)
            except Exception:
                return None

        unified = safe_float(unified)
        sil = safe_float(sil)
        ch = safe_float(ch)
        gap = safe_float(gap)

        unified_str = f"{unified:.4f}" if unified is not None else "N/A"
        sil_str = f"{sil:.4f}" if sil is not None else "N/A"
        ch_str = f"{ch:.4f}" if ch is not None else "N/A"
        gap_str = f"{gap:.4f}" if gap is not None else "N/A"

        lines.append(f"{i}. {name} - ")
        lines.append(f"Unified score: {unified_str}")
        lines.append(f"Silhouette: {sil_str}")
        lines.append(f"Calinski-Harabasz: {ch_str}")
        lines.append(f"Gap: {gap_str}")
        lines.append("")  # пустая строка

    with open(filepath, "w", encoding="utf-8") as f:
        f.write("\n".join(lines))