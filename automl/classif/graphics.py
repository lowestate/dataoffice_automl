import pandas as pd
import numpy as np
from typing_extensions import Optional, TypedDict

class ConfusionMatrixResult(TypedDict):
    matrix: list[list[int]]
    labels: list[str]


def get_confusion_matrix(
        y_test: pd.Series | np.ndarray,
        y_pred_final: pd.Series | np.ndarray,
        labels: Optional[list[str]] = None
    ) -> ConfusionMatrixResult:
    from sklearn.metrics import confusion_matrix

    y_true = y_test.tolist() if isinstance(y_test, pd.Series) else list(y_test)
    y_pred = y_pred_final.tolist() if isinstance(y_pred_final, pd.Series) else list(y_pred_final)

    cm = confusion_matrix(y_true, y_pred)
    if labels is None:
        labels = [str(i) for i in range(cm.shape[0])]

    return {
        "matrix": cm.tolist(),
        "labels": labels
    }