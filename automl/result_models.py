from dataclasses import dataclass
from typing import Literal, Optional, Any


@dataclass
class BaseModelResult:
    model_name: str
    hyperparams: dict[str, object]
    metrics: dict[str, float]
    graphics: dict[str, dict[str, Any]]
    files: dict[str, dict[str, str]]
    train_start: str
    train_finish: str


@dataclass
class ClassifResult(BaseModelResult):
    task_type: Literal["binary", "multiclass"]
    is_imbalanced: bool
    metrics: dict[
        Literal["f1_score", "accuracy", "roc_auc"],
        float
    ]
    selected_features: list[str]
    graphics: dict[
        Literal[
            "feature_importance",
            "confusion_matrix"
        ],
        dict[str, Any]
    ]
    files: dict[
        Literal["model", "label_encoder", "feature_encoder"],
        dict[str, str]
    ]


@dataclass
class ClusteringResult(BaseModelResult):
    metrics: dict[str, Any]
    metrics_orig: dict[
        Literal["silhouette", "calinski_harabasz", "gap", "stability"],
        float
    ]
    metrics_norm: dict[
        Literal["silhouette", "calinski_harabasz", "gap", "stability"],
        float
    ]
    n_clusters: int
    n_clusters_scores: list[dict[str, Any]]
    graphics: dict[str, Any]
    centroids: dict[str, list[float]]
    cluster_stats: dict[
        Literal["mean", "median", "mode", "min", "max", "std", "var", "count"],
        float
    ]
    files: dict[
        Literal["model", "pca", "scaler", "feature_encoder"],
        dict[str, str]
    ]
    selected_features: list[str]


@dataclass
class RegressResult(BaseModelResult):
    target_transform: Optional[Literal["log1p"]]
    metrics: dict[
        Literal["mse", "mae", "r2"],
        float
    ]
    selected_features: list[str]
    poly_features: list[str]
    graphics: dict[
        Literal[
            "feature_importance",
            "pred_vs_true"
        ],
        dict[str, Any]
    ]
    files: dict[
        Literal["model", "label_encoder", "feature_encoder"],
        dict[str, str]
    ]