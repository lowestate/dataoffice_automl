import pandas as pd
import numpy as np
from logging import Logger
from datetime import datetime, timezone
import warnings
warnings.filterwarnings("ignore")

import hdbscan  # type: ignore
from sklearn.cluster import (
    KMeans,
    DBSCAN,
    AgglomerativeClustering,
    MeanShift,
    OPTICS,
    Birch,
    BisectingKMeans
)
from sklearn.mixture import GaussianMixture
from sklearn.base import clone

from automl.config import ClusteringConfig
from automl.funcs import (
    serialize_file
)
from automl.result_models import ClusteringResult
from automl.cluster.graphics import generate_cluster_plot
from automl.cluster.metrics import (
    normalize_metrics,
    compute_metrics,
    clustering_stability,
    gap_statistic_kmeans
)
from automl.cluster.utils import (
    build_cluster_spaces,
    safe_get_params,
    compute_cluster_stats,
    detect_metric
)
from automl.optuna import optimize_model, compute_unified_score
from typing import Callable, Optional
#from automl.cluster.posthoc_scores import compute_unified_score

def train_all_models(X_spaces, logger, config: ClusteringConfig, random_state: int, progress_callback: Optional[Callable[..., None]] = None):
    results = []
    # Эталон для оценки (то, что мы видим глазами)
    X_eval = X_spaces['manifold'][:, :2]

    for space_name, X_cluster in X_spaces.items():
        if space_name == 'manifold':
            X_train = X_cluster[:, :2]
        else:
            X_train = X_cluster
            
        logger.info(f"Processing space: {space_name}")
        
        models_to_train = ["KMeans", "BisectingKMeans", "GaussianMixture", "DBSCAN", "HDBSCAN", "OPTICS", "MeanShift", "Birch"]
        if X_train.shape[0] <= 15000:
            models_to_train.append("AgglomerativeClustering")

        for name in models_to_train:
            if (space_name == "manifold" and name in ["KMeans", "BisectingKMeans", "GaussianMixture", "MeanShift", "Birch"]) or \
               (name == "DBSCAN" and space_name != "linear"):
                continue

            logger.info(f"{name} ({space_name}) | Tuning...")
            if progress_callback:
                progress_callback("Поиск кластеров", 2, name)
            model_train_start = datetime.now(timezone.utc)
            best_params, k_trials = optimize_model(
                task_type="clustering",
                X_train=X_train, 
                X_eval=X_eval,
                model_name=name, 
                n_trials=config.n_trials,
                random_state=random_state,
                logger=logger
            )
            
            # Initialize model
            match name:
                case "KMeans": model = KMeans(**best_params, n_init="auto", random_state=random_state)
                case "BisectingKMeans": model = BisectingKMeans(**best_params, random_state=random_state)
                case "MeanShift": model = MeanShift(**best_params, bin_seeding=True)
                case "OPTICS": model = OPTICS(**best_params)
                case "Birch": model = Birch(**best_params)
                case "GaussianMixture": model = GaussianMixture(**best_params, random_state=random_state)
                case "AgglomerativeClustering": model = AgglomerativeClustering(**best_params)
                case "DBSCAN": model = DBSCAN(**best_params, metric=detect_metric(X_train))
                case "HDBSCAN": 
                    m = detect_metric(X_train)
                    model = hdbscan.HDBSCAN(**best_params, metric="euclidean" if m == "cosine" else m)

            # FIT & PREDICT
            labels = model.fit_predict(X_train)
            
            if len(np.unique(labels[labels != -1])) < 2:
                continue

            # EVALUATE
            sil, ch, noise_ratio = compute_metrics(X_eval, labels)
            
            # <--- ПРОКИДЫВАЕМ ПАРАМЕТРЫ СТАБИЛЬНОСТИ ИЗ КОНФИГА --->
            stability = clustering_stability(
                model, 
                X_train, 
                n_runs=config.stability_n_runs, 
                sample_frac=config.stability_sample_frac
            )
            
            n_clusters_real = len(np.unique(labels[labels != -1]))
            
            # <--- ПРОКИДЫВАЕМ ПАРАМЕТРЫ GAP ИЗ КОНФИГА --->
            gap = None
            if name == "KMeans" and space_name == "linear":
                gap = gap_statistic_kmeans(
                    X_eval, 
                    n_clusters_real, 
                    random_state=random_state, 
                    n_refs=config.gap_n_refs
                )

            unified_score, weights = compute_unified_score(
                X=X_eval, 
                labels=labels, 
                stability=stability, 
                gap=gap, 
                n_clusters=n_clusters_real,
                strong_overlap_penalty=True
            )
            
            if unified_score < 0.001:
                continue

            # --- ВОЗВРАЩАЕМ РАСЧЕТ k-scores ДЛЯ UI ---
            model_k_scores = []
            if name in ["KMeans", "BisectingKMeans", "AgglomerativeClustering", "GaussianMixture", "Birch"]:
                # Берем уникальные k из триалов Optuna
                for trial_k in sorted(set(k_trials)):
                    try:
                        # Создаем копию модели с новым k
                        test_params = best_params.copy()
                        if name == "GaussianMixture":
                            test_params["n_components"] = trial_k
                        else:
                            test_params["n_clusters"] = trial_k
                            
                        from typing import cast, Any
                        test_model = cast(Any, clone(model)).set_params(**test_params)
                        l_trial = test_model.fit_predict(X_train)
                        
                        # Оцениваем по 2D эталону
                        s_t, c_t, _ = compute_metrics(X_eval, l_trial)
                        model_k_scores.append({
                            "n_clusters": trial_k, 
                            "silhouette_score": s_t, 
                            "calinski_harabasz_score": c_t
                        })
                    except:
                        continue

            # Форматирование метаданных
            formula_keys = ["silhouette", "calinski_harabasz", "stability", "gap_score", "overlap_score", "centroid_sep"]
            multiplier_keys = ["noise_penalty", "separation_penalty", "k_penalty", "merge_multiplier", "balance_penalty", "micro_penalty"]
            extra_for_f = {f"f_{k}": weights.get("formula", {}).get(k) for k in formula_keys} | \
                          {f"m_{k}": weights.get("multipliers", {}).get(k) for k in multiplier_keys}

            model_train_finish = datetime.now(timezone.utc)
            results.append({
                "name": f"{name}_{space_name}", 
                "model": model, 
                "labels": labels,
                "silhouette": sil, 
                "calinski_harabasz": ch, 
                "gap": gap or np.nan,
                "stability": stability, 
                "noise_ratio": noise_ratio, 
                "n_clusters": n_clusters_real,
                "space": space_name, 
                "unified_score": unified_score,
                "n_clusters_scores": model_k_scores, # Тот самый ключ, которого не хватало!
                "train_start": model_train_start.isoformat(),
                "train_finish": model_train_finish.isoformat()
            } | extra_for_f)

    return normalize_metrics(results)

def cluster(
        df: pd.DataFrame,
        scaler: object,
        lencoder: object,
        logger: Logger,
        random_state: int,
        config: ClusteringConfig,
        progress_callback: Optional[Callable[..., None]] = None,
    ) -> dict[str, dict[str, ClusteringResult]]:
    
    logger.info("clustering started")
    if progress_callback:
        progress_callback("Анализ пространства данных", 1)
    
    # 1. Генерируем пространства. 
    # X_spaces['manifold'] — это уже 2D массив, посчитанный в build_cluster_spaces.
    # Это наш "холст", на котором мы будем рисовать ВСЕ модели.
    X_spaces, reducer = build_cluster_spaces(df, random_state, config)
    X_metric = X_spaces["linear"]
    X_viz_base = X_spaces["manifold"] # Используем это для отрисовки

    # 2. Обучаем и тюним все модели
    all_models_raw = train_all_models(X_spaces, logger, config, random_state, progress_callback=progress_callback)
    
    # Сортируем по нашему новому унифицированному скору
    all_models = sorted(
        all_models_raw,
        key=lambda x: x["unified_score"],
        reverse=True
    )[:config.top_n_models]
    
    logger.info(f"top-{config.top_n_models} models: {[model['name'] for model in all_models]}")

    response = {"models": {}}
    
    for i, m in enumerate(all_models, start=1):
        logger.info(f"{m['name']} | final loop processing")
        
        # Берем данные пространства, в котором модель реально обучалась (для статистики)
        labels = m["labels"]
        
        # 3. ГЕНЕРАЦИЯ ГРАФИКА
        # Мы ВСЕГДА передаем X_viz_base (те самые 2D точки из manifold).
        # Это гарантирует, что точка №50 всегда будет на одном и том же месте на всех графиках ТОПа.
        is_noise_algorithm = any(algo in m["name"] for algo in ["DBSCAN", "HDBSCAN", "OPTICS"])
        
        if progress_callback:
            progress_callback("Визуализация кластеров", 4, m["name"])
        clust_plot = generate_cluster_plot(
            X_umap=X_viz_base,
            labels=labels,
            is_hdbscan=is_noise_algorithm
        )
        
        # 4. Сбор статистики и метрик
        if progress_callback:
            progress_callback("Расчет метрик качества", 3, m["name"])
        cluster_stats = compute_cluster_stats(X_metric, labels)
        
        metrics = {k: m[k] for k in ["unified_score", "silhouette", "calinski_harabasz", "gap"] if k in m}
        # Добавляем подробные веса формулы для отладки в UI
        metrics.update({k: v for k, v in m.items() if k.startswith("f_") or k.startswith("m_")})
        
        # 5. Сериализация файлов
        if progress_callback:
            progress_callback("Сохранение моделей", 5, m["name"])
        files = {
            "model": serialize_file(obj=m["model"], filename="model.joblib"),
            "pca": serialize_file(obj=reducer, filename="pca.pkl"),
            "scaler": serialize_file(obj=scaler, filename="scaler.pkl")
        }
        if lencoder is not None:
            files["feature_encoder"] = serialize_file(obj=lencoder, filename="feature_encoder.joblib")

        # Формируем итоговый объект
        from typing import cast, Any
        response["models"][f"топ-{i} модель"] = ClusteringResult(
            model_name=m["name"],
            hyperparams=safe_get_params(m["model"]),
            metrics=metrics,
            n_clusters=m["n_clusters"],
            n_clusters_scores=m["n_clusters_scores"],
            graphics={"clusters": clust_plot},
            centroids=clust_plot["centroids"],
            cluster_stats=cluster_stats,
            files=cast(Any, files),
            metrics_orig=m.get("metrics_orig", {}),
            metrics_norm=m.get("metrics_norm", {}),
            train_start=m["train_start"],
            train_finish=m["train_finish"]
        )

        logger.info(f"{m['name']} | serialized successfully")
        
    return response