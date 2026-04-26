from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Dict, Optional

import numpy as np
from sklearn.base import clone
from sklearn.metrics import roc_auc_score
from sklearn.model_selection import StratifiedKFold


def partial_auc_score(y_true, y_pred, max_fpr: float = 0.035) -> float:
    return float(roc_auc_score(y_true, y_pred, max_fpr=max_fpr))


def make_stratified_cv(n_splits: int = 5, random_state: int = 42) -> StratifiedKFold:
    return StratifiedKFold(n_splits=n_splits, shuffle=True, random_state=random_state)


@dataclass
class CVResult:
    oof_pred: np.ndarray
    fold_scores: list[float]
    score_mean: float
    score_std: float


def cross_val_predict_proba(
    estimator,
    X,
    y,
    cv: Optional[StratifiedKFold] = None,
    max_fpr: float = 0.035,
) -> CVResult:
    if cv is None:
        cv = make_stratified_cv()

    oof = np.zeros(len(X), dtype=float)
    fold_scores: list[float] = []

    for tr_idx, va_idx in cv.split(X, y):
        model = clone(estimator)
        model.fit(X.iloc[tr_idx], y[tr_idx])
        pred = model.predict_proba(X.iloc[va_idx])[:, 1]
        oof[va_idx] = pred
        fold_scores.append(partial_auc_score(y[va_idx], pred, max_fpr=max_fpr))

    return CVResult(
        oof_pred=oof,
        fold_scores=fold_scores,
        score_mean=float(np.mean(fold_scores)),
        score_std=float(np.std(fold_scores)),
    )
