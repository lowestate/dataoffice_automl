import pandas as pd
import numpy as np
from typing import Any, Callable, Optional
from logging import Logger
from joblib import parallel_backend
from datetime import datetime, timezone
import warnings
warnings.filterwarnings("ignore")

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, PolynomialFeatures, OrdinalEncoder
from sklearn.feature_selection import SelectKBest, f_regression
from sklearn.metrics import (
    mean_squared_error,
    mean_absolute_error,
    r2_score,
    root_mean_squared_error,
)
from sklearn.base import clone

from automl.config import RegressionConfig
from automl.funcs import (
    serialize_file,
    get_feature_importance
)
from automl.result_models import RegressResult
from automl.regress.models_and_hparams import get_models
from automl.regress.graphics import get_pred_vs_true
from automl.optuna import optimize_model

def baseline(
        X_train: pd.DataFrame,
        X_test: pd.DataFrame,
        y_train: pd.Series | np.ndarray,
        y_test: pd.Series | np.ndarray,
        scaler: StandardScaler,
        numeric_cols: list[str],
        random_state: int
    ) -> list[dict[str, Any]]:
    baseline_results = []
    models = get_models(random_state)

    for name, model in models.items():
        # Полиномиальные признаки только для линейных моделей
        if name in ["LinearRegression", "Ridge", "Lasso"]:
            poly = PolynomialFeatures(degree=2, include_bias=False)
            X_train_model = poly.fit_transform(X_train)
            X_test_model = poly.transform(X_test)
            feature_names = poly.get_feature_names_out(X_train.columns)
        else:
            X_train_model = scaler.fit_transform(X_train[numeric_cols])
            X_test_model = scaler.transform(X_test[numeric_cols])
            feature_names = np.array(X_train[numeric_cols].columns)

        # Отбор признаков для каждой модели отдельно
        selector = SelectKBest(score_func=f_regression, k="all")  # type: ignore
        X_train_selected = selector.fit_transform(X_train_model, y_train)
        X_test_selected = selector.transform(X_test_model)
        selected_features = np.array(feature_names)[selector.get_support()]

        model.fit(X_train_selected, y_train)
        y_pred = model.predict(X_test_selected)

        baseline_results.append({
            "name": name,
            "model": model,
            "X_train_selected": X_train_selected,
            "X_test_selected": X_test_selected,
            "selected_features": selected_features,
            "mae": mean_absolute_error(y_test, y_pred),
            "rmse": root_mean_squared_error(y_test, y_pred),
            "r2": r2_score(y_test, y_pred)
        })
    
    return baseline_results

def regress(
        df: pd.DataFrame,
        target: str,
        logger: Logger,
        config: RegressionConfig,
        random_state: int,
        lencoder: OrdinalEncoder | None,
        progress_callback: Optional[Callable[..., None]] = None,
    ) -> dict[str, dict[str, RegressResult]]:
    logger.info(f"regression started")
    if progress_callback:
        progress_callback("Оценка базовых моделей", 1)
    X = df.drop(columns=[target])
    y = df[target].copy()

    # лог-трансформация таргета
    from typing import cast
    y_numeric = cast(pd.Series, pd.to_numeric(y, errors="coerce"))
    if abs(y_numeric.skew()) > config.skew_threshold:  # type: ignore[operator]
        y_transformed = np.log1p(y_numeric.values)
        transform_type = "log1p"
    else:
        y_transformed = y_numeric.values
        transform_type = None

    logger.info(f"transform y: {transform_type}")

    y_transformed = np.asarray(y_transformed, dtype=float)
    y_binned = pd.qcut(y_transformed, q=10, duplicates="drop")
    X_train, X_test, y_train, y_test = train_test_split(
        X,
        y_transformed,
        test_size=config.test_size,
        random_state=random_state,
        stratify=y_binned,
    )
    X_train = cast(pd.DataFrame, X_train)
    X_test = cast(pd.DataFrame, X_test)
    y_train = cast(np.ndarray, y_train)
    y_test = cast(np.ndarray, y_test)

    numeric_cols = X_train.select_dtypes(include=["number"]).columns.tolist()
    scaler = StandardScaler()

    baseline_results = baseline(
        X_train=X_train,
        X_test=X_test,
        y_train=y_train,
        y_test=y_test,
        scaler=scaler,
        numeric_cols=numeric_cols,
        random_state=random_state
    )

    top_models = sorted(
        baseline_results,
        key=lambda x: x["r2"],
        reverse=True
    )[:config.top_n_models]
    logger.info(f"top-{config.top_n_models} models: {[m['name'] for m in top_models]}")

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
            task_type="regression",     # TaskType.CLASSIFICATION
            model_name=m["name"],
            X_train=X_train_selected,
            y_train=y_train,
            base_model=m["model"],        # Передаем инстанс с class_weight!
            n_trials=config.n_iter,
            random_state=random_state,
            cv_folds=config.cv_folds,
            scoring="r2",
            logger=logger
        )

        logger.info(f"{m['name']} | hyperparams done")
        best_model = cast(Any, clone(m["model"])).set_params(**best_params)
        with parallel_backend("threading", n_jobs=-1):
            best_model.fit(X_train_selected, y_train)
        
        y_pred_final = best_model.predict(X_test_selected) # type: ignore

        if progress_callback:
            progress_callback("Расчет метрик", 3, m["name"])
        from typing import Literal
        metrics: dict[Literal["mse", "mae", "r2"], float] = {
            "mse": float(mean_squared_error(y_test, y_pred_final)),
            "mae": float(mean_absolute_error(y_test, y_pred_final)),
            "r2": float(r2_score(y_test, y_pred_final)),
        }

        logger.info(f"{m['name']} | metrics done")

        if progress_callback:
            progress_callback("Визуализация результатов", 4, m["name"])
        graphics = {}
        
        graphics["pred_vs_true"] = get_pred_vs_true(
            y_test=y_test,
            y_pred_final=y_pred_final
        )

        if hasattr(best_model, "feature_importances_"):
            graphics["feature_importance"] = get_feature_importance(best_model, selected_features, logger)

        logger.info(f"{m['name']} | graphics done")

        if progress_callback:
            progress_callback("Сохранение моделей", 5, m["name"])
        files = {
            "model": serialize_file(
                obj=best_model,
                filename="model.joblib"
            )
        }
        if lencoder is not None:
            files["feature_encoder"] = serialize_file(obj=lencoder, filename="feature_encoder.joblib")

        logger.info(f"{m['name']} | files done")
        model_train_finish = datetime.now(timezone.utc)

        response["models"][f"топ-{i} модель"] = RegressResult(
            model_name=m["name"],
            hyperparams=best_params,
            metrics=metrics,
            target_transform=transform_type,
            selected_features=selected_features.tolist(),
            poly_features=selected_features.tolist() if m["name"] in ["LinearRegression", "Ridge", "Lasso"] else [],
            graphics=cast(Any, graphics),
            files=cast(Any, files),
            train_start=model_train_start.isoformat(),
            train_finish=model_train_finish.isoformat()
        )

        logger.info(f"{m['name']} | serialized")        

    return response