import optuna
import optuna.logging  # type: ignore
from optuna.trial import Trial  # type: ignore
import numpy as np
import hdbscan  # type: ignore
from typing import Any, Optional, cast
from logging import Logger

from sklearn.base import clone
from sklearn.model_selection import cross_val_score
from sklearn.neighbors import NearestNeighbors
from sklearn.metrics import silhouette_score, calinski_harabasz_score, pairwise_distances
from sklearn.cluster import (
    KMeans, BisectingKMeans, MeanShift, OPTICS, Birch, 
    DBSCAN, AgglomerativeClustering, estimate_bandwidth
)
from sklearn.mixture import GaussianMixture
from sklearn.preprocessing import normalize
from scipy import sparse
from joblib import parallel_backend

# ==========================================
# 1. ХЕЛПЕРЫ ДЛЯ КЛАСТЕРИЗАЦИИ (из твоего кода)
# ==========================================

def estimate_eps(X: np.ndarray) -> float:
    n_neighbors = max(10, int(np.log(X.shape[0]) * 4))
    neigh = NearestNeighbors(n_neighbors=n_neighbors)
    neigh.fit(X)
    dists, _ = neigh.kneighbors(X)
    return float(np.percentile(dists[:, -1], 90))

def compute_overlap_penalty(*, X: np.ndarray, labels: np.ndarray) -> float:
    labels = np.asarray(labels)
    mask = labels != -1
    X = X[mask]
    labels = labels[mask]
    unique_labels = np.unique(labels)
    if len(unique_labels) < 2: return 1.0

    centroids = np.vstack([X[labels == lbl].mean(axis=0) for lbl in unique_labels])
    dists = np.linalg.norm(X[:, None, :] - centroids[None, :, :], axis=2)
    true_idx = np.array([np.where(unique_labels == lbl)[0][0] for lbl in labels])
    nearest_idx = np.argmin(dists, axis=1)
    misassigned_basic = np.mean(nearest_idx != true_idx)

    max_intra, min_inter = [], []
    for lbl in unique_labels:
        points = X[labels == lbl]
        others = X[labels != lbl]
        max_intra.append(np.percentile(pairwise_distances(points), 90) if len(points) > 1 else 0.0)
        min_inter.append(np.percentile(pairwise_distances(points, others), 5) if len(others) > 0 else np.inf)

    merge_penalty = np.mean([min(1.0, intra / inter) for intra, inter in zip(max_intra, min_inter) if inter > 1e-6])
    overlap_score = 1.0 - (0.5 * misassigned_basic + 0.5 * merge_penalty)
    return float(np.clip(overlap_score, 0.0, 1.0))

def compute_centroid_separation(*, X, labels) -> float:
    unique = np.unique(labels[labels != -1])
    if len(unique) < 2: return 0.0
    
    centroids = np.vstack([X[labels == l].mean(axis=0) for l in unique])
    dists = pairwise_distances(centroids)
    
    data_range = np.linalg.norm(np.percentile(X, 95, axis=0) - np.percentile(X, 5, axis=0))
    if data_range == 0: return 0.0
    
    avg_dist = np.mean(dists[np.triu_indices(len(unique), k=1)])
    return float(np.clip(avg_dist / data_range, 0, 1))

def compute_unified_score(
    X: np.ndarray, labels: np.ndarray, *,
    stability: float = 0.0, gap: float | None = None,
    n_clusters: int | None = None, strong_overlap_penalty: bool = False
) -> tuple[float, dict[str, dict[str, float]]]:
    labels = np.asarray(labels)
    mask = labels != -1
    noise_ratio = float(np.mean(labels == -1))

    # Отбрасываем откровенный мусор сразу (слишком много шума или нет точек)
    if noise_ratio > 0.5 or np.sum(mask) < 2: 
        return 0.0, {}

    unique_labels, counts = np.unique(labels[mask], return_counts=True)
    if len(unique_labels) < 2: 
        return 0.0, {}

    max_cluster_ratio = np.max(counts) / np.sum(mask)
    balance_penalty = 1.0
    if max_cluster_ratio > 0.95: balance_penalty = 0.1
    elif max_cluster_ratio > 0.85: balance_penalty = 0.5
        
    min_cluster_ratio = np.min(counts) / np.sum(mask)
    micro_penalty = 1.0 if min_cluster_ratio > 0.01 else 0.5

    if n_clusters is None: 
        n_clusters = len(unique_labels)

    try: sil = float(silhouette_score(X[mask], labels[mask]))
    except: sil = 0.0

    overlap_score = compute_overlap_penalty(X=X, labels=labels)

    try:
        ch_raw = float(calinski_harabasz_score(X[mask], labels[mask]))
        ch = np.log1p(ch_raw) / (np.log1p(ch_raw) + 5.0)
    except: ch = 0.0

    gap_score = float(np.tanh(gap / 3)) if gap is not None and not np.isnan(gap) else 0.0
    stability = float(np.clip(stability, 0.0, 1.0))
    centroid_sep = compute_centroid_separation(X=X, labels=labels)

    if n_clusters == 2: k_penalty = 1.0  
    elif 3 <= n_clusters <= 7: k_penalty = 1.1  
    else: k_penalty = 5 / n_clusters if n_clusters > 7 else 0.1

    # 1. ЖЁСТКИЙ ШТРАФ ЗА ШУМ (прощаем до 5%, далее — экспоненциальное падение)
    # При 5% шума = 1.0 | При 10% шума = 0.6 | При 20% шума = 0.22
    noise_penalty = float(np.exp(-10 * max(0.0, noise_ratio - 0.05)))

    separation_penalty = 1.0 if sil > 0.1 else (max(0.01, float(sil)) / 0.1)
    merge_multiplier = overlap_score if strong_overlap_penalty else np.sqrt(overlap_score)

    # Собираем базовую метрику (игнорируем отрицательный силуэт)
    base_score = (
        0.35 * max(0.0, sil) + 
        0.20 * overlap_score + 
        0.15 * ch + 
        0.15 * centroid_sep + 
        0.10 * stability + 
        0.05 * gap_score
    )

    # 2. НОРМАЛИЗАЦИЯ И РАСТЯЖЕНИЕ
    # Умножаем на 1.6, так как сырая сумма метрик редко пробивает 0.6 даже при идеальной кластеризации
    normalized_base = float(np.clip(base_score * 1.6, 0.0, 1.0))

    # Применяем все штрафы
    final_multiplier = (k_penalty * noise_penalty * separation_penalty * merge_multiplier * balance_penalty * micro_penalty)
    
    # Итоговый скор строго в [0, 1]
    score = float(np.clip(normalized_base * final_multiplier, 0.0, 1.0))

    weights = {
        "formula": {"silhouette": sil, "calinski_harabasz": ch, "stability": stability, "gap_score": gap_score, "overlap_score": overlap_score, "centroid_sep": centroid_sep},
        "multipliers": {"noise_penalty": noise_penalty, "separation_penalty": separation_penalty, "k_penalty": k_penalty, "merge_multiplier": merge_multiplier, "balance_penalty": balance_penalty, "micro_penalty": micro_penalty}
    }
    return score, weights

def detect_metric(X):
    if sparse.issparse(X) or X.shape[1] > 50: return "cosine"
    return "euclidean"


# ==========================================
# 2. ДИНАМИЧЕСКИЕ ПРОСТРАНСТВА ПОИСКА (Supervised)
# ==========================================

def suggest_supervised_params(trial: Trial, model_name: str, task_type: str) -> dict:
    """Генерирует пространство поиска на лету в зависимости от задачи и модели."""
    params = {}
    is_classif = (task_type == "classification") # Замени на TaskType.CLASSIFICATION
    
    if model_name in ["LogisticRegression", "LinearRegression"]:
        if is_classif: params["C"] = trial.suggest_float("C", 1e-3, 1e2, log=True)
        # LinearRegression параметров не имеет
        
    elif model_name == "Ridge":
        params["alpha"] = trial.suggest_float("alpha", 0.01, 100, log=True)
        
    elif model_name == "Lasso":
        params["alpha"] = trial.suggest_float("alpha", 0.001, 1, log=True)
        
    elif model_name == "RandomForest":
        if is_classif:
            params["n_estimators"] = trial.suggest_int("n_estimators", 100, 600)
            params["max_depth"] = trial.suggest_categorical("max_depth", [None, 5, 10, 15, 20])
            params["min_samples_split"] = trial.suggest_int("min_samples_split", 2, 10)
        else:
            params["n_estimators"] = trial.suggest_int("n_estimators", 50, 500)
            params["max_depth"] = trial.suggest_int("max_depth", 5, 30)
            
    elif model_name == "SVC":
        params["C"] = trial.suggest_float("C", 1e-2, 1e2, log=True)
        params["gamma"] = trial.suggest_categorical("gamma", ["scale", "auto"])
        
    elif model_name == "SVR":
        params["C"] = trial.suggest_float("C", 0.1, 5)
        
    elif model_name == "KNN":
        if is_classif:
            params["n_neighbors"] = trial.suggest_int("n_neighbors", 3, 30)
            params["weights"] = trial.suggest_categorical("weights", ["uniform", "distance"])
        else:
            params["n_neighbors"] = trial.suggest_int("n_neighbors", 2, 15)
            
    elif model_name == "GradientBoosting":
        params["n_estimators"] = trial.suggest_int("n_estimators", 100, 500)
        params["learning_rate"] = trial.suggest_float("learning_rate", 0.01, 0.2)
        params["max_depth"] = trial.suggest_int("max_depth", 2, 6) if is_classif else trial.suggest_int("max_depth", 3, 10)
            
    elif model_name == "DecisionTree":
        if is_classif:
            params["max_depth"] = trial.suggest_categorical("max_depth", [None, 5, 10, 15, 20])
            params["min_samples_split"] = trial.suggest_int("min_samples_split", 2, 10)
        else:
            params["max_depth"] = trial.suggest_int("max_depth", 3, 30)
            
    elif model_name == "NaiveBayes":
        pass 
        
    elif model_name == "XGBoost":
        params["n_estimators"] = trial.suggest_int("n_estimators", 100, 800 if is_classif else 500)
        params["max_depth"] = trial.suggest_int("max_depth", 3, 10)
        params["learning_rate"] = trial.suggest_float("learning_rate", 0.01, 0.3)
            
    elif model_name == "LightGBM":
        params["n_estimators"] = trial.suggest_int("n_estimators", 100, 800 if is_classif else 500)
        params["num_leaves"] = trial.suggest_int("num_leaves", 20, 120 if is_classif else 100)
        params["learning_rate"] = trial.suggest_float("learning_rate", 0.01, 0.3)
            
    elif model_name == "CatBoost":
        params["depth"] = trial.suggest_int("depth", 4 if is_classif else 3, 10)
        params["iterations"] = trial.suggest_int("iterations", 200 if is_classif else 100, 800 if is_classif else 500)
        params["learning_rate"] = trial.suggest_float("learning_rate", 0.01, 0.3)
            
    return params


# ==========================================
# 3. ЕДИНЫЙ ДИСПЕТЧЕР ОПТИМИЗАЦИИ
# ==========================================

def optimize_model(
    task_type: str,
    model_name: str,
    logger: Logger,
    X_train: np.ndarray,
    y_train: Optional[np.ndarray] = None, # Нужно для классификации/регрессии
    X_eval: Optional[np.ndarray] = None,  # Нужно для кластеризации
    base_model: Optional[Any] = None,     # Базовый инстанс модели (с class_weights и т.д.)
    n_trials: int = 25,
    random_state: int = 42,
    cv_folds: int = 3,
    scoring: str = "accuracy"
) -> tuple[dict, list[int]]:
    """
    Универсальная функция тюнинга.
    Возвращает: (лучшие_гиперпараметры, список_k_для_кластеризации)
    """
    # Отключаем спам в консоль от Optuna
    optuna.logging.set_verbosity(optuna.logging.WARNING)  # type: ignore
    
    k_trials: list[int] = []

    logger.info(f"{model_name} | Optuna started...")

    def objective(trial):
        # ----------------------------------------
        # ВЕТВЬ 1: КЛАСТЕРИЗАЦИЯ
        # ----------------------------------------
        if task_type == "clustering": # Замени на TaskType.CLUSTERING
            assert X_eval is not None
            n_samples = X_train.shape[0]
            metric = detect_metric(X_train)
            X_clust = normalize(X_train) if metric == "cosine" else X_train
            max_min_samples = int(np.clip(n_samples * 0.05, 5, 200))
            
            try:
                k = None
                if model_name == "KMeans":
                    k = trial.suggest_int("n_clusters", 2, min(15, max(3, int(np.sqrt(n_samples)))))
                    model = KMeans(n_clusters=k, n_init="auto", random_state=random_state)
                elif model_name == "BisectingKMeans":
                    k = trial.suggest_int("n_clusters", 2, min(15, max(3, int(np.sqrt(n_samples)))))
                    model = BisectingKMeans(n_clusters=k, random_state=random_state)
                elif model_name == "GaussianMixture":
                    k = trial.suggest_int("n_components", 2, min(15, max(3, int(np.sqrt(n_samples)))))
                    model = GaussianMixture(n_components=k, covariance_type=trial.suggest_categorical("covariance_type", ["full", "diag"]), random_state=random_state)
                elif model_name == "MeanShift":
                    bw = estimate_bandwidth(X_clust, quantile=0.3)
                    model = MeanShift(bandwidth=trial.suggest_float("bandwidth", max(0.1, bw*0.5), max(1.0, bw*2.0)), bin_seeding=True)
                elif model_name == "Birch":
                    k = trial.suggest_int("n_clusters", 2, min(15, max(3, int(np.sqrt(n_samples)))))
                    model = Birch(threshold=trial.suggest_float("threshold", 0.1, 0.7), n_clusters=k)
                elif model_name == "DBSCAN":
                    base_eps = estimate_eps(cast(np.ndarray, X_clust))
                    model = DBSCAN(eps=trial.suggest_float("eps", base_eps*0.5, base_eps*2.0), min_samples=trial.suggest_int("min_samples", 5, max_min_samples))
                elif model_name == "HDBSCAN":
                    model = hdbscan.HDBSCAN(
                        min_cluster_size=trial.suggest_int("min_cluster_size", max(5, int(n_samples * 0.03)), max(10, int(n_samples * 0.15))),
                        min_samples=trial.suggest_int("min_samples", 5, max(10, max_min_samples // 2)),
                        metric="euclidean" if metric == "cosine" else metric
                    )
                elif model_name == "AgglomerativeClustering":
                    model = AgglomerativeClustering(
                        n_clusters=trial.suggest_int("n_clusters", 2, 8),
                        linkage=trial.suggest_categorical("linkage", ["average", "complete", "single"]) 
                    )
                elif model_name == "OPTICS":
                    model = OPTICS(
                        # ИСПРАВЛЕНИЕ: Нижняя граница теперь 5, а верхняя гарантированно не меньше 10
                        min_samples=trial.suggest_int("min_samples", 5, max(10, max_min_samples)), 
                        xi=trial.suggest_float("xi", 0.01, 0.15),
                        min_cluster_size=trial.suggest_float("min_cluster_size", 0.05, 0.2) 
                    )
                else: return -1.0

                labels = model.fit_predict(X_clust)
                if len(np.unique(labels[labels != -1])) < 2: return -1.0
                if k: k_trials.append(k)

                score, _ = compute_unified_score(X=X_eval, labels=labels, strong_overlap_penalty=True)
                return float(score)
                
            except Exception as e:
                logger.error(f"Optuna error: {str(e)}")
                return -1.0

        # ----------------------------------------
        # ВЕТВЬ 2: SUPERVISED (Классификация/Регрессия)
        # ----------------------------------------
        else:
            try:
                # Получаем гиперпараметры из нашего словаря
                params = suggest_supervised_params(trial, model_name, task_type)
                
                # Клонируем базовую модель (которая уже содержит random_state и class_weights)
                model = cast(Any, clone(base_model))
                model.set_params(**params)
                
                # Запускаем валидацию в защищенном потоковом пуле, чтобы не сломать Windows Joblib
                with parallel_backend("threading", n_jobs=-1):
                    scores = cross_val_score(model, X_train, y_train, cv=cv_folds, scoring=scoring)
                
                # Возвращаем среднюю метрику
                return scores.mean()
            except Exception as e:
                logger.error(f"Optuna error: {str(e)}")
                # Если Optuna предложила несочетаемые параметры, просто отбраковываем trial
                return -float("inf")

    # Запускаем Study
    study = optuna.create_study(direction="maximize")  # type: ignore
    study.optimize(objective, n_trials=n_trials)
    logger.info(f"{model_name} | Optuna finished")
    
    return study.best_params, list(set(k_trials))