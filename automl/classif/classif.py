import pandas as pd
import numpy as np
from logging import Logger
from typing import Any, Protocol, Callable, Optional
from joblib import parallel_backend
from datetime import datetime, timezone
import warnings
warnings.filterwarnings("ignore")

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import PolynomialFeatures, StandardScaler, LabelEncoder, OrdinalEncoder
from sklearn.feature_selection import SelectKBest, f_classif
from sklearn.metrics import accuracy_score, f1_score, roc_auc_score
from sklearn.base import clone

from automl.config import ClassificationConfig
from automl.funcs import (
    serialize_file,
    get_feature_importance,
)
from automl.result_models import ClassifResult
from automl.classif.graphics import get_confusion_matrix
from automl.classif.models_and_hparams import get_models
from automl.optuna import optimize_model


class FitPredictProtocol(Protocol):
    def fit(self, X: Any, y: Any) -> Any: ...
    def predict(self, X: Any) -> Any: ...


def baseline(
        models: dict[str, FitPredictProtocol],
        X_train_scaled: pd.DataFrame,
        X_test_scaled: pd.DataFrame,
        y_train: pd.Series | np.ndarray,
        y_test: pd.Series | np.ndarray,
        is_binary: bool,
        is_imbalanced: bool
    ) -> list[dict[str, Any]]:
    baseline_results = []

    for name, model in models.items():
        # Полиномиальные признаки только для линейной модели
        if name == "LogisticRegression":
            poly = PolynomialFeatures(degree=2, include_bias=False)
            X_train_model = poly.fit_transform(X_train_scaled)
            X_test_model = poly.transform(X_test_scaled)
            feature_names = poly.get_feature_names_out(X_train_scaled.columns)
        else:
            X_train_model = X_train_scaled.values
            X_test_model = X_test_scaled.values
            feature_names = X_train_scaled.columns.to_numpy()

        # Отбор признаков для текущей модели
        selector = SelectKBest(score_func=f_classif, k="all")  # type: ignore
        X_train_selected = selector.fit_transform(X_train_model, y_train)
        X_test_selected = selector.transform(X_test_model)
        selected_features = np.array(feature_names)[selector.get_support()]

        model.fit(X_train_selected, y_train)
        y_pred = model.predict(X_test_selected)

        if is_binary:
            main_metric = f1_score(y_test, y_pred) if is_imbalanced else accuracy_score(y_test, y_pred)
        else:
            main_metric = f1_score(y_test, y_pred, average="macro") if is_imbalanced else accuracy_score(y_test, y_pred)

        baseline_results.append({
            "name": name,
            "model": model,
            "main_metric": main_metric,
            "X_train_selected": X_train_selected,
            "X_test_selected": X_test_selected,
            "selected_features": selected_features
        })
    
    return baseline_results

def classif(
        df: pd.DataFrame,
        target: str,
        logger: Logger,
        config: ClassificationConfig,
        random_state: int,
        lencoder: OrdinalEncoder | None,
        progress_callback: Optional[Callable[..., None]] = None,
    ) -> dict[str, dict[str, ClassifResult]]:
    if progress_callback:
        progress_callback("Оценка базовых моделей", 1)

    X = df.drop(columns=[target])
    y = df[target]

    le = LabelEncoder()
    y = le.fit_transform(y)

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=config.test_size, random_state=random_state, stratify=y
    )
    from typing import cast
    X_train = cast(pd.DataFrame, X_train)
    X_test = cast(pd.DataFrame, X_test)
    y_train = cast(np.ndarray, y_train)
    y_test = cast(np.ndarray, y_test)

    unique_classes = np.unique(y_train)
    n_classes = len(unique_classes)
    is_binary = n_classes == 2
    task_type = "binary" if is_binary else "multiclass"

    numeric_cols = X_train.select_dtypes(include=["number"]).columns.tolist()

    scaler = StandardScaler()
    X_train_scaled = X_train.copy()
    X_train_scaled[numeric_cols] = scaler.fit_transform(X_train[numeric_cols])
    X_test_scaled = X_test.copy()
    X_test_scaled[numeric_cols] = scaler.transform(X_test[numeric_cols])

    counts = np.bincount(y_train)
    imbalance_ratio = counts.min() / counts.max()
    is_imbalanced = bool(imbalance_ratio < config.imbalance_threshold)
    
    if is_imbalanced and config.use_class_weights:
        class_weight = "balanced"
    else:
        class_weight = None

    # ---------- BASELINE и отбор признаков для каждой модели ----------
    baseline_results = baseline(
        models=get_models(class_weight, random_state),
        X_train_scaled=X_train_scaled,
        X_test_scaled=X_test_scaled,
        y_train=y_train,
        y_test=y_test,
        is_binary=is_binary,
        is_imbalanced=is_imbalanced
    )

    # Топ-5 моделей
    top_models = sorted(
        baseline_results,
        key=lambda x: x["main_metric"],
        reverse=True
    )[:config.top_n_models]
    logger.info(f"top-{config.top_n_models} models: {[model['name'] for model in top_models]}")

    scoring = (
        "f1_macro" if (not is_binary and is_imbalanced)
        else "f1" if (is_binary and is_imbalanced)
        else "accuracy"
    )

    response = {
        "models": {}
    }

    for i, m in enumerate(top_models, start=1):
        logger.info(f"{m['name']} | final loop")
        if progress_callback:
            progress_callback("Настройка гиперпараметров", 2, m["name"])
        model_train_start = datetime.now(timezone.utc)

        X_train_selected = m["X_train_selected"]
        X_test_selected = m["X_test_selected"]
        selected_features = m["selected_features"]

        best_params, _ = optimize_model(
            task_type="classitication",     # TaskType.CLASSIFICATION
            model_name=m["name"],
            X_train=X_train_selected,
            y_train=y_train,
            base_model=m["model"],        # Передаем инстанс с class_weight!
            n_trials=config.n_iter,
            random_state=random_state,
            cv_folds=config.cv_folds,
            scoring=scoring,
            logger=logger
        )

        logger.info(f"{m['name']} | hyperparams done")

        best_model = cast(Any, clone(m["model"])).set_params(**best_params)
        with parallel_backend("threading", n_jobs=-1):
            best_model.fit(X_train_selected, y_train)

        y_pred_final = best_model.predict(X_test_selected) # type: ignore
        y_prob_final = best_model.predict_proba(X_test_selected) if hasattr(best_model, "predict_proba") else None  # type: ignore

        if progress_callback:
            progress_callback("Расчет метрик", 3, m["name"])
        metrics = {}
        if is_imbalanced:
            metrics["f1_score"] = float(f1_score(y_test, y_pred_final, average="macro" if not is_binary else "binary"))
        else:
            metrics["accuracy"] = float(accuracy_score(y_test, y_pred_final))

        logger.info(f"{m['name']} | metrics done")

        if y_prob_final is not None:
            if is_binary:
                metrics["roc_auc"] = float(roc_auc_score(y_test, y_prob_final[:, 1]))
            else:
                metrics["roc_auc"] = float(roc_auc_score(y_test, y_prob_final, multi_class="ovr"))
                
        if progress_callback:
            progress_callback("Визуализация результатов", 4, m["name"])
        graphics: dict[str, Any] = {"confusion_matrix": get_confusion_matrix(y_test, y_pred_final)}
        if hasattr(best_model, "feature_importances_"):
            graphics["feature_importance"] = get_feature_importance(
                best_model,
                selected_features,
                logger
            )

        logger.info(f"{m['name']} | graphics done")

        if progress_callback:
            progress_callback("Сохранение моделей", 5, m["name"])
        files = {
            "model": serialize_file(obj=best_model, filename="model.joblib"),
            "target_encoder": serialize_file(obj=le, filename="target_encoder.joblib")
        }
        if lencoder is not None:
            files["feature_encoder"] = serialize_file(obj=lencoder, filename="feature_encoder.joblib")

        logger.info(f"{m['name']} | files done")
        model_train_finish = datetime.now(timezone.utc)

        response["models"][f"топ-{i} модель"] = ClassifResult(
            model_name=m["name"],
            task_type=task_type,
            is_imbalanced=is_imbalanced,
            hyperparams=best_params,
            metrics=metrics,
            selected_features=selected_features.tolist(),
            graphics=cast(Any, graphics),
            files=cast(Any, files),
            train_start=model_train_start.isoformat(),
            train_finish=model_train_finish.isoformat()
        )

        logger.info(f"{m['name']} | serialized")

    return response