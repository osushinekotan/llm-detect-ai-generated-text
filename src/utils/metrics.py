from typing import Callable

import joblib
import numpy as np
import pandas as pd
from sklearn.metrics import (
    accuracy_score,
    f1_score,
    log_loss,
    mean_absolute_error,
    mean_absolute_percentage_error,
    mean_squared_error,
    median_absolute_error,
    precision_score,
    r2_score,
    recall_score,
    roc_auc_score,
)


def to_numpy(values: pd.DataFrame | pd.Series | np.ndarray) -> np.ndarray:
    return np.array(values)


def optimize_threshold(
    y_true: pd.DataFrame | pd.Series | np.ndarray,
    y_pred: pd.DataFrame | pd.Series | np.ndarray,
    metrics: Callable,
    n_jobs: int = -1,
    maximize: bool = True,
) -> float:
    search_vals = list(np.arange(0, 1, 0.001))

    def _calc(th: float) -> float:
        score = metrics(y_true, y_pred >= th)
        if not maximize:
            return score * -1
        return score

    results = joblib.Parallel(n_jobs=n_jobs)(joblib.delayed(_calc)(th) for th in search_vals)
    max_idx = np.argmax(results)
    return search_vals[max_idx]


def opt_f1_score(
    y_true: pd.DataFrame | pd.Series | np.ndarray,
    y_pred: pd.DataFrame | pd.Series | np.ndarray,
) -> dict[str, float]:
    th = optimize_threshold(y_true, y_pred, metrics=f1_score)
    return {"th": th, "score": f1_score(y_true, y_pred >= th)}


def opt_acc_score(
    y_true: pd.DataFrame | pd.Series | np.ndarray,
    y_pred: pd.DataFrame | pd.Series | np.ndarray,
) -> dict[str, float]:
    th = optimize_threshold(y_true, y_pred, metrics=accuracy_score)
    return {"th": th, "score": accuracy_score(y_true, y_pred >= th)}


def binary_classification_metrics(
    y_true: pd.DataFrame | pd.Series | np.ndarray,
    y_pred: pd.DataFrame | pd.Series | np.ndarray,
    th: float = 0.5,
) -> dict[str, float]:
    y_pred, y_true = to_numpy(y_pred), to_numpy(y_true)
    none_prob_functions = [
        precision_score,
        recall_score,
    ]
    prob_functions = [
        roc_auc_score,
        log_loss,
        opt_f1_score,
        opt_acc_score,
    ]

    scores = {}
    for f in none_prob_functions:
        scores[f.__name__] = f(y_true, y_pred >= th)
    for f in prob_functions:
        scores[f.__name__] = f(y_true, y_pred)

    return scores


def root_mean_squared_error(
    y_true: pd.DataFrame | pd.Series | np.ndarray,
    y_pred: pd.DataFrame | pd.Series | np.ndarray,
) -> float:
    return mean_squared_error(y_true=y_true, y_pred=y_pred, squared=True)


def regression_metrics(
    y_true: pd.DataFrame | pd.Series | np.ndarray,
    y_pred: pd.DataFrame | pd.Series | np.ndarray,
) -> dict[str, float]:
    y_pred, y_true = to_numpy(y_pred), to_numpy(y_true)
    metricses = {
        "rmse": root_mean_squared_error,
        "mean-ae": mean_absolute_error,
        "meadian-ae": median_absolute_error,
        "r2": r2_score,
        "mape": mean_absolute_percentage_error,
    }
    scores = {}
    for k, func in metricses.items():
        try:
            scores[k] = func(y_true, y_pred)
        except Exception:
            scores[k] = None

    return scores
