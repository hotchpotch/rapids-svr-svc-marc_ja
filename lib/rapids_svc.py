"""
RAPIDS SVC で embedding を学習し、結果を返す
"""

from __future__ import annotations
from cuml.svm import SVC
import numpy as np

DEFAULT_SVC_PARAMS = {
    "C": 3.0,  # Penalty parameter C of the error term.
    "kernel": "rbf",  # Possible options: ‘linear’, ‘poly’, ‘rbf’, ‘sigmoid’.
    "degree": 3,
    "gamma": "scale",  # auto or scale
    "coef0": 0.0,
    "tol": 0.001,  # 0.001 = 1e-3
}


def train_svc(
    X: np.ndarray,
    y: np.ndarray,
    svc_params: dict[str, object] = DEFAULT_SVC_PARAMS,
) -> SVC:
    svc = SVC(**svc_params)
    svc.fit(X, y)
    return svc
