import numpy as np
from enum import Enum
from typing import Any, cast
import pandas as pd
from logging import Logger
from typing import Callable
from dataclasses import asdict, is_dataclass
from sklearn.preprocessing import (
    OrdinalEncoder,
    FunctionTransformer,
    OneHotEncoder,
    RobustScaler
)
from sklearn.impute import SimpleImputer
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.decomposition import TruncatedSVD
from sklearn.feature_extraction import FeatureHasher
from scipy import sparse
from scipy.sparse import spmatrix, csr_matrix

from automl.regress.regression import regress
from automl.classif.classif import classif
from automl.cluster.cluster import cluster
from automl.funcs import serialize
from automl.config import BaseTaskConfig


class TaskType(str, Enum):
    REGRESSION = "regression"
    CLASSIFICATION = "classification"
    CLUSTERING = "clustering"


TASK_REGISTRY: dict[TaskType, Callable[..., dict[str, Any]]] = {
    TaskType.REGRESSION: regress,
    TaskType.CLASSIFICATION: classif,
    TaskType.CLUSTERING: cluster,
}


class AutoML:
    def __init__(
        self,
        *,
        logger: Logger,
        random_state: int = 42
    ) -> None:
        self.logger = logger
        self.random_state = random_state

    @staticmethod
    def _hash_rows(X) -> spmatrix:
        """Внутренний метод для хэширования строк (high cardinality)"""
        hasher = FeatureHasher(n_features=128, input_type="string")
        X_str = X.astype(str)
        data = X_str.values if hasattr(X_str, "values") else X_str
        return hasher.transform(data)

    def preprocess(
            self,
            df: pd.DataFrame,
            *,
            task: str,
            cols_to_remove: list[str] | None = None
        ) -> tuple[np.ndarray, dict, pd.DataFrame, Pipeline, OrdinalEncoder | None]:
        
        self.logger.info("preprocessing started")
        cols_to_remove = cols_to_remove or []

        df = df.copy()
        raw_cols = df.columns.tolist()
        raw_rows_n = df.shape[0]

        # --- cleaning ---
        df = df.loc[:, ~df.columns.str.contains("Unnamed", case=False)]
        df = df.loc[:, ~df.columns.str.contains("id", case=False)]
        df = df.drop(columns=cols_to_remove, errors="ignore")
        
        df = df.dropna(axis=1, how="all")
        if task != 'clustering':
            df = df.drop_duplicates()
        df = df.dropna()
        df = df.reset_index(drop=True)

        num_cols = df.select_dtypes(include=[np.number]).columns.tolist()
        cat_cols = df.select_dtypes(include=["object", "category"]).columns.tolist()

        high_card_cols = [c for c in cat_cols if df[c].nunique() > 50]
        low_card_cols = [c for c in cat_cols if c not in high_card_cols]

        # --- pipelines ---
        num_pipe = Pipeline([
            ("imputer", SimpleImputer(strategy="median")),
            ("scaler", RobustScaler())
        ])

        cat_low_pipe = Pipeline([
            ("imputer", SimpleImputer(strategy="most_frequent")),
            ("encoder", OneHotEncoder(handle_unknown="ignore", sparse_output=True))
        ])

        cat_high_pipe = Pipeline([
            ("imputer", SimpleImputer(strategy="most_frequent")),
            ("hasher", FunctionTransformer(self._hash_rows, validate=False))
        ])

        transformers = []
        if num_cols:
            transformers.append(("num", num_pipe, num_cols))
        if low_card_cols:
            transformers.append(("cat_low", cat_low_pipe, low_card_cols))
        if high_card_cols:
            transformers.append(("cat_high", cat_high_pipe, high_card_cols))

        preprocessor = ColumnTransformer(
            transformers=transformers,
            sparse_threshold=0.3,
            remainder="drop"
        )

        X = preprocessor.fit_transform(df)

        if X is None:
            raise ValueError("Preprocessor returned None")

        # --- Automated Dimensionality Guard ---
        n_samples, n_features = X.shape  # type: ignore
        if n_features > 100 or (task == "clustering" and n_features > 50):
            n_comps = min(50, n_features - 1, n_samples - 1)
            if n_comps > 0:
                self.logger.info(f"Auto-compression: {n_features} -> {n_comps}")
                # Используем единый random_state из класса!
                reducer = TruncatedSVD(n_components=n_comps, random_state=self.random_state)
                X = reducer.fit_transform(X)

        # --- Safe Sparse-to-Dense ---
        if sparse.issparse(X):
            X_dense = cast(csr_matrix, X).toarray()
        else:
            X_dense = np.asarray(X)

        # --- correlation mapping & label encoding ---
        df_corr = df.copy()
        mapping = {}
        label_encoder = None

        if cat_cols:
            label_encoder = OrdinalEncoder(handle_unknown='use_encoded_value', unknown_value=-1)
            df_corr_cats = df_corr[cat_cols].fillna("missing").astype(str)
            df_corr[cat_cols] = label_encoder.fit_transform(df_corr_cats)
            
            for i, col in enumerate(cat_cols):
                uniques = label_encoder.categories_[i]
                mapping[col] = {str(u): int(idx) for idx, u in enumerate(uniques)}

        meta = {
            "cols_deleted_n": len(raw_cols) - df.shape[1],
            "cols_deleted": list(set(raw_cols) - set(df.columns)),
            "rows_deleted_n": raw_rows_n - df.shape[0],
            "high_cardinality_cols": high_card_cols,
            "low_cardinality_cols": low_card_cols,
            "numeric_categorical_mapping": mapping,
            "final_features_n": X_dense.shape[1]
        }

        self.logger.info(f"preprocessing finished: {meta}")
        return X_dense, meta, df_corr, num_pipe, label_encoder

    def run(
            self,
            *,
            task_type: TaskType,
            df: pd.DataFrame,
            config: BaseTaskConfig,
            target: str | None,
            scaler: object | None = None,
            lencoder: OrdinalEncoder | None = None,
            preprocessing=True,
            cols_to_remove: list[str] | None = None,
            progress_callback: Callable[[str, int], None] | None = None
        ) -> tuple[dict[str, Any], dict | None]:

        task_func: Callable[..., dict[str, Any]] | None = TASK_REGISTRY.get(task_type)

        if task_func is None:
            raise ValueError("unsupported task")

        if task_type != TaskType.CLUSTERING and target is None:
            raise ValueError("target is required")

        preproc_meta = None

        # Внутренний препроцессинг, если включен
        if preprocessing:
            X, preproc_meta, df_corr, p_scaler, p_lencoder = self.preprocess(
                df=df, 
                task=task_type.value, 
                cols_to_remove=cols_to_remove
            )
            
            # Обновляем переменные для обучения
            scaler = p_scaler
            lencoder = p_lencoder
            
            # Для кластеризации берем матрицу X, для остальных df_corr
            if task_type == TaskType.CLUSTERING:
                df = X  # type: ignore (функция кластеризации справляется с np.ndarray)
            else:
                df = df_corr

        task_kwargs = {
            "df": df,
            "logger": self.logger,
            "lencoder": lencoder,
            "config": config,
            "random_state": self.random_state,
            "progress_callback": progress_callback
        }

        if task_type == TaskType.CLUSTERING:
            task_kwargs["scaler"] = scaler
        else:
            task_kwargs["target"] = target

        result = task_func(**task_kwargs)

        if "models" in result:
            result["models"] = {
                name: serialize(
                    asdict(model) if is_dataclass(model) else model  # type: ignore
                )
                for name, model in result["models"].items()
            }
        
        return result, preproc_meta