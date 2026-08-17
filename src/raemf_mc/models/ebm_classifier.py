from __future__ import annotations

import numpy as np
from interpret.glassbox import ExplainableBoostingClassifier


def fit_ebm_classifier(
    X_train: np.ndarray, y_train: np.ndarray, random_state: int = 0
) -> ExplainableBoostingClassifier:
    """Fit a point-estimate 4-class Explainable Boosting Machine.
    interactions=0 keeps the model to single-feature terms only, appropriate
    given the training set is a few thousand rows, not enough to safely
    estimate pairwise interaction terms without overfitting."""
    model = ExplainableBoostingClassifier(random_state=random_state, interactions=0)
    model.fit(X_train, y_train)
    return model


def predict_scores(model: ExplainableBoostingClassifier, X: np.ndarray) -> np.ndarray:
    """Raw per-class additive log-odds scores (NOT probabilities) — what
    temperature scaling needs to operate on. model.classes_ gives the
    column order for these scores, which is alphabetical order among
    whatever classes were present in y_train (e.g. 'Bear','Bull','Sideway',
    'Stress' if all four appeared), not STATE_NAMES order.

    Real regime label sequences can be heavily imbalanced enough that a
    chronological train split sees only 2 of the 4 regimes. When only 2
    classes are present, sklearn's/interpret's decision_function convention
    collapses to a single 1-D margin — log(p[classes_[1]] / p[classes_[0]])
    — instead of one score column per class. Normalize that back into the
    same 2-D (n_samples, n_classes) additive log-odds layout the multiclass
    case already produces, anchoring classes_[0] at a score of 0: softmax
    of [0, raw] reproduces [p[classes_[0]], p[classes_[1]]] exactly, so
    downstream code (log_softmax/nll_loss-based calibration) can treat the
    binary and multiclass cases identically."""
    raw = np.asarray(model.decision_function(X))
    if raw.ndim == 1:
        raw = np.stack([np.zeros_like(raw), raw], axis=1)
    return raw
