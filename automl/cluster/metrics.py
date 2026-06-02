import numpy as np
from scipy import sparse
from sklearn.decomposition import PCA
from sklearn.cluster import KMeans

from sklearn.cluster import KMeans
from sklearn.metrics import (
    silhouette_score,
    calinski_harabasz_score,
    adjusted_rand_score
)

def normalize_metrics(results: list[dict]):
    if not results:
        return []

    sils = np.array([r["silhouette"] for r in results])
    chs = np.array([r["calinski_harabasz"] for r in results])
    gaps = np.array([0.0 if (r["gap"] is None or np.isnan(r["gap"])) else r["gap"] for r in results])
    stabs = np.array([r.get("stability", 0.0) for r in results])

    def norm(arr):
        arr_min, arr_max = arr.min(), arr.max()
        if arr_max - arr_min < 1e-10:
            return np.ones_like(arr)
        return (arr - arr_min) / (arr_max - arr_min)

    sil_norm = norm(sils)
    ch_norm = norm(chs)
    gap_norm = norm(gaps)
    stab_norm = norm(stabs)

    for i, r in enumerate(results):
        metrics_orig = {
            "silhouette": float(sils[i]),
            "calinski_harabasz": float(chs[i]),
            "gap": float(gaps[i]),
            "stability": float(stabs[i])
        }
        for k, v in r.items():
            if k.startswith("f_") or k.startswith("m_"):
                metrics_orig[k] = float(v) if v is not None else 0.0
        r["metrics_orig"] = metrics_orig
        r["metrics_norm"] = {
            "silhouette": float(sil_norm[i]),
            "calinski_harabasz": float(ch_norm[i]),
            "gap": float(gap_norm[i]),
            "stability": float(stab_norm[i])
        }
        r["score"] = r["unified_score"]
    return results

def compute_metrics(X_metric_space: np.ndarray, labels: np.ndarray):
    labels = np.asarray(labels)
    noise_ratio = float(np.mean(labels == -1))
    mask = labels != -1
    if np.sum(mask) < 2 or len(np.unique(labels[mask])) < 2:
        return 0.0, 0.0, noise_ratio

    sil = float(silhouette_score(X_metric_space[mask], labels[mask]))
    ch = float(calinski_harabasz_score(X_metric_space[mask], labels[mask]))
    sil_penalized = sil * (1.0 - noise_ratio)
    return sil_penalized, ch, noise_ratio

def clustering_stability(
        model, 
        X: np.ndarray, 
        n_runs: int,
        sample_frac: float
    ):
    X_arr = np.asarray(X)
    n = X_arr.shape[0]
    labels_runs = []
    for _ in range(n_runs):
        idx = np.random.choice(n, int(n * sample_frac), replace=False)
        X_sub = X_arr[idx]
        try:
            labels = model.fit_predict(X_sub)
        except Exception:
            return 0.0
        labels_runs.append((idx, labels))

    scores = []
    for i in range(len(labels_runs)):
        for j in range(i + 1, len(labels_runs)):
            idx_i, lab_i = labels_runs[i]
            idx_j, lab_j = labels_runs[j]
            common = np.intersect1d(idx_i, idx_j)
            if len(common) < 10:
                continue
            li = lab_i[np.isin(idx_i, common)]
            lj = lab_j[np.isin(idx_j, common)]
            try:
                scores.append(adjusted_rand_score(li, lj))
            except Exception:
                pass
    if not scores:
        return 0.0
    return float(np.mean(scores))

def gap_statistic_kmeans(
    X: np.ndarray,
    k: int,
    random_state: int,
    n_refs: int           # <--- Убрали хардкод =5
) -> float:
    if k <= 1:
        return 0.0

    # --- безопасное приведение ---
    if sparse.issparse(X):
        X_dense = sparse.csr_matrix(X).toarray()
    else:
        X_dense = np.asarray(X)

    # --- авто PCA ---
    n_samples, n_features = X_dense.shape
    if n_features > 50:
        X_dense = PCA(
            n_components=min(50, n_features - 1),
            random_state=random_state
        ).fit_transform(X_dense)

    mins = X_dense.min(axis=0)
    maxs = X_dense.max(axis=0)

    ranges = maxs - mins
    ranges[ranges < 1e-9] = 1e-9

    ref_disps = []
    for _ in range(n_refs):
        random_ref = mins + np.random.uniform(
            low=0.0,
            high=1.0,
            size=X_dense.shape
        ) * ranges

        km = KMeans(
            n_clusters=k,
            n_init=10,  # type: ignore
            random_state=random_state
        ).fit(random_ref)

        inertia = km.inertia_
        assert inertia is not None
        ref_disps.append(float(inertia))

    km_orig = KMeans(
        n_clusters=k,
        n_init=10,  # type: ignore
        random_state=random_state
    ).fit(X_dense)

    orig_disp = km_orig.inertia_
    assert orig_disp is not None
    orig_disp_val = float(orig_disp)

    return float(np.mean(np.log(ref_disps)) - np.log(orig_disp_val))