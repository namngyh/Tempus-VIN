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
    column order for these scores, which is alphabetical
    ('Bear','Bull','Sideway','Stress'), not STATE_NAMES order."""
    return np.asarray(model.decision_function(X))
