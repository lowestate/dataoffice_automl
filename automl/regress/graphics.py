import pandas as pd
import numpy as np
from typing_extensions import TypedDict


class PredVsTrueResult(TypedDict):
    y_true: list[float]
    y_pred: list[float]


def get_pred_vs_true(
        y_test: pd.Series | np.ndarray,
        y_pred_final: pd.Series | np.ndarray
    ) -> PredVsTrueResult:
    y_true_list = y_test.tolist() if isinstance(y_test, pd.Series) else list(y_test)
    y_pred_list = y_pred_final.tolist() if isinstance(y_pred_final, pd.Series) else list(y_pred_final)

    return {
        "y_true": y_true_list,
        "y_pred": y_pred_list
    }