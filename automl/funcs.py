import io
import math
import base64
import numpy as np
import pickle
from typing_extensions import Any, TypedDict, Optional
import pandas as pd
from logging import Logger
from matplotlib.figure import Figure  # type: ignore
import math
from scipy.stats import chi2_contingency


class FeatureImportanceResult(TypedDict):
    features: list[str]
    importances: list[float]


class CorrMatrixResult(TypedDict):
    corr_matrix: list[list[float]]
    labels: list[str]
    data_id: str


def to_base64(obj: Figure | object) -> str:
    """сериализация объекта в base64"""
    res = ''
    buf = io.BytesIO()

    pickle.dump(obj, buf)
    buf.seek(0)
    res = base64.b64encode(buf.read()).decode("utf-8")
    
    return res

# Поля, которые идут напрямую в колонки таблицы models_metadata
# (используем реальные имена колонок из БД)
_DIRECT_DB_FIELDS = {"model_name", "metrics", "graphics", "model_metadata", "train_start", "train_finish"}


def model_result_to_dict(
        result: dict[str, Any],
        place_in_batch: int | None = None,
    ) -> dict[str, Any]:
    """Преобразует результат обучения модели в словарь с ключами == колонки models_metadata."""
    direct_fields: dict[str, Any] = {}
    extra_metadata: dict[str, Any] = {}

    for key, value in result.items():
        if key in _DIRECT_DB_FIELDS:
            direct_fields[key] = value
        elif key in ("hyperparams", "hyperparameters"):
            # Нормализуем к имени колонки в БД
            direct_fields["hyperparams"] = value
        elif key == "files":
            # files хранится внутри model_metadata (колонки files нет в БД)
            extra_metadata["files"] = value
        else:
            extra_metadata[key] = value

    # Объединяем extra_metadata с model_metadata
    existing_meta = direct_fields.get("model_metadata") or {}
    if extra_metadata:
        existing_meta.update(extra_metadata)
    direct_fields["model_metadata"] = existing_meta or None

    if place_in_batch is not None:
        direct_fields["place_in_training_res_batch"] = place_in_batch

    return direct_fields

def get_feature_importance(
        estimator: Any,
        selected_features: np.ndarray,
        logger: Logger
    ) -> Optional[FeatureImportanceResult]:

    importances = getattr(estimator, "feature_importances_", None)

    if importances is None:
        logger.warning(
            f"{type(estimator).__name__} has no attribute feature_importances_"
        )
        return None

    features = np.array(selected_features)

    idx = np.argsort(importances)[::-1]

    importances_sorted = importances[idx]
    features_sorted = features[idx]
    logger.info("feature importance calculated")

    return {
        "features": features_sorted.tolist(),
        "importances": importances_sorted.tolist()
    }

def serialize(v: Any) -> Any:
    # NumPy целые числа -> int
    if isinstance(v, (np.integer, np.int64, np.int32)):  # type: ignore
        return int(v)
    # NumPy числа с плавающей точкой -> float, проверка NaN/Inf
    elif isinstance(v, (np.floating, np.float64, np.float32)):  # type: ignore
        f = float(v)
        if math.isnan(f) or math.isinf(f):
            return None
        return f
    # NumPy булевы значения -> bool
    elif isinstance(v, (np.bool_)):
        return bool(v)
    # NumPy массивы -> списки
    elif isinstance(v, np.ndarray):
        return serialize(v.tolist())
    # dict -> рекурсивно
    elif isinstance(v, dict):
        return {k: serialize(val) for k, val in v.items()}
    # list/tuple/set -> рекурсивно
    elif isinstance(v, (list, tuple, set)):
        return [serialize(val) for val in v]
    # float NaN/Inf обычного типа
    elif isinstance(v, float):
        if math.isnan(v) or math.isinf(v):
            return None
        return v
    # bool, int, str, None оставляем как есть
    elif isinstance(v, (int, str, bool)) or v is None:
        return v
    # datetime -> строка
    elif hasattr(v, "isoformat"):
        try:
            return v.isoformat()
        except Exception:
            return str(v)
    # fallback: все остальное -> строка
    else:
        try:
            return str(v)
        except Exception:
            return repr(v)

def serialize_file(
        *,
        obj: Any,
        filename: str
    ) -> dict[str, str]:
    return {
        "data": to_base64(obj),
        "filename": filename
    }

def cramers_v(x, y):
    confusion = pd.crosstab(x, y)
    chi2 = chi2_contingency(confusion)[0]
    n = confusion.sum().sum()
    r, k = confusion.shape
    return np.sqrt(chi2 / (n * (min(k - 1, r - 1) + 1e-9)))

def correlation_ratio(categories, measurements):
    categories = pd.Categorical(categories)
    cat_codes = categories.codes

    y_avg = np.mean(measurements)
    numerator = 0
    denominator = np.sum((measurements - y_avg) ** 2)

    for cat in np.unique(cat_codes):
        cat_measures = measurements[cat_codes == cat]
        if len(cat_measures) == 0:
            continue
        numerator += len(cat_measures) * (np.mean(cat_measures) - y_avg) ** 2

    return np.sqrt(numerator / (denominator + 1e-9))


def compute_mixed_corr(df: pd.DataFrame):
    cols = df.columns
    n = len(cols)
    corr = np.zeros((n, n))

    for i in range(n):
        for j in range(n):
            from typing import cast
            col1 = cast(pd.Series, df[cols[i]])
            col2 = cast(pd.Series, df[cols[j]])

            if i == j:
                corr[i, j] = 1.0
                continue

            is_num1 = pd.api.types.is_numeric_dtype(col1)
            is_num2 = pd.api.types.is_numeric_dtype(col2)

            try:
                if is_num1 and is_num2:
                    corr[i, j] = col1.corr(col2)

                elif not is_num1 and not is_num2:
                    corr[i, j] = cramers_v(col1, col2)

                else:
                    if is_num1:
                        corr[i, j] = correlation_ratio(col2, col1)
                    else:
                        corr[i, j] = correlation_ratio(col1, col2)

            except:
                corr[i, j] = 0.0

    corr = np.nan_to_num(corr, nan=0.0, posinf=0.0, neginf=0.0)
    return corr, cols.tolist()