from dataclasses import dataclass

@dataclass
class BaseTaskConfig:
    """Базовый класс с общими параметрами для всех задач"""

    top_n_models: int = 90
    n_trials: int = 25


@dataclass
class ClusteringConfig(BaseTaskConfig):
    """Специфичные параметры для кластеризации"""

    stability_n_runs: int = 8          
    stability_sample_frac: float = 0.8 
    gap_n_refs: int = 5                
    umap_min_dist: float = 0.05


@dataclass
class SupervisedConfig(BaseTaskConfig):
    """Общие параметры для задач с учителем"""

    test_size: float = 0.25
    cv_folds: int = 3
    n_iter: int = 25


@dataclass
class ClassificationConfig(SupervisedConfig):
    """Специфичные параметры для классификации"""

    # Например, можно вынести стратегию балансировки классов
    use_class_weights: bool = True
    imbalance_threshold: float = 0.5


@dataclass
class RegressionConfig(SupervisedConfig):
    """Специфичные параметры для регрессии"""

    # Например, можно вынести порог для лог-трансформации
    skew_threshold: float = 1.0