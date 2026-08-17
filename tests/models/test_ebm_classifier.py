import numpy as np
from raemf_mc.models.ebm_classifier import fit_ebm_classifier, predict_scores


def _synthetic_4class_data(n=200, seed=0):
    rng = np.random.default_rng(seed)
    X = rng.normal(size=(n, 4))
    # class depends mostly on the first feature so the model has real signal
    y = np.where(
        X[:, 0] > 0.5, "Bull",
        np.where(X[:, 0] > 0, "Sideway", np.where(X[:, 0] > -0.5, "Bear", "Stress")),
    )
    return X, y


def test_fit_and_predict_scores_shape():
    X, y = _synthetic_4class_data()
    model = fit_ebm_classifier(X, y, random_state=0)
    scores = predict_scores(model, X[:10])
    assert scores.shape == (10, 4)


def test_classes_order_is_alphabetical_not_state_names_order():
    X, y = _synthetic_4class_data()
    model = fit_ebm_classifier(X, y, random_state=0)
    assert list(model.classes_) == ["Bear", "Bull", "Sideway", "Stress"]


def test_model_learns_real_signal_better_than_majority_baseline():
    X, y = _synthetic_4class_data(n=400)
    split = 300
    model = fit_ebm_classifier(X[:split], y[:split], random_state=0)
    preds = model.predict(X[split:])
    accuracy = (preds == y[split:]).mean()
    majority_class = pd_value_counts_top(y[:split])
    baseline_accuracy = (y[split:] == majority_class).mean()
    assert accuracy > baseline_accuracy


def pd_value_counts_top(y):
    values, counts = np.unique(y, return_counts=True)
    return values[np.argmax(counts)]


def test_predict_scores_normalizes_binary_case_to_two_columns():
    # Real chronological splits of imbalanced regime labels can legitimately
    # contain only 2 of the 4 possible classes. interpret's
    # decision_function collapses to a 1-D margin in that case; predict_scores
    # must still return a (n_samples, n_classes) matrix so downstream
    # log_softmax-based calibration code doesn't need a special case.
    rng = np.random.default_rng(0)
    X = rng.normal(size=(200, 4))
    y = np.where(X[:, 0] > 0, "Bull", "Stress")
    model = fit_ebm_classifier(X, y, random_state=0)
    assert list(model.classes_) == ["Bull", "Stress"]

    scores = predict_scores(model, X[:10])
    assert scores.shape == (10, 2)

    # column 0 (classes_[0]) is anchored at 0, so softmax(scores) must
    # reproduce predict_proba exactly.
    exp_scores = np.exp(scores)
    softmax_probs = exp_scores / exp_scores.sum(axis=1, keepdims=True)
    np.testing.assert_allclose(softmax_probs, model.predict_proba(X[:10]), atol=1e-6)
