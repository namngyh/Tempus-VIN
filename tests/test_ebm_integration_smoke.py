# tests/test_ebm_integration_smoke.py
from pathlib import Path

import numpy as np
import pandas as pd
import torch
import yaml

from raemf_mc.data.loader import load_vnindex_ohlcv
from raemf_mc.features.returns import compute_log_returns
from raemf_mc.features.causal import compute_causal_features
from raemf_mc.validation.chronological_split import chronological_split
from raemf_mc.regime.ms_egarch import (
    MSEGARCHParamLayout,
    default_recursion_init,
    fit_ms_egarch,
)
from raemf_mc.regime.posterior_features import (
    canonical_theta,
    compute_posterior_volatility_features,
    compute_regime_labels,
)
from raemf_mc.models.ebm_classifier import fit_ebm_classifier, predict_scores
from raemf_mc.models.calibration import apply_temperature, fit_temperature
from raemf_mc.bayesian.torch_backend import AdviConfig
from raemf_mc.bayesian.priors import HierarchicalPriorConfig

CONFIG_PATH = Path(__file__).resolve().parents[1] / "configs" / "ebm_smoke.yaml"


def test_ebm_pipeline_end_to_end_on_real_data(tmp_path):
    with CONFIG_PATH.open(encoding="utf-8") as f:
        config = yaml.safe_load(f)

    ohlcv = load_vnindex_ohlcv()
    ohlcv_window = ohlcv.iloc[-config["window_sessions"]:]
    log_returns = compute_log_returns(ohlcv_window)
    causal_features = compute_causal_features(ohlcv_window)

    # causal_features drops a 60-row warmup that log_returns doesn't; align
    # everything to the intersection of both indices.
    common_index = log_returns.index.intersection(causal_features.index)
    log_returns = log_returns.loc[common_index]
    causal_features = causal_features.loc[common_index]

    returns_train, returns_val, returns_test = chronological_split(
        log_returns.to_frame("ret"), train_frac=config["train_frac"], val_frac=config["val_frac"]
    )
    # Centering must use TRAIN-only mean, applied to the whole series.
    train_mean = returns_train["ret"].mean()
    centered_returns = log_returns - train_mean

    layout = MSEGARCHParamLayout()
    advi_config = AdviConfig(**config["advi"])
    prior_config = HierarchicalPriorConfig(**config["prior"])

    train_returns_tensor = torch.tensor(
        centered_returns.loc[returns_train.index].to_numpy(), dtype=torch.float32
    )
    posterior = fit_ms_egarch(
        train_returns_tensor, advi_config, prior_config, seeds=config["seeds"],
        device=torch.device("cpu"), layout=layout,
        fallback_log_path=tmp_path / "fallbacks.json",
    )
    for r in posterior.seed_results:
        assert torch.isfinite(r.mu).all()

    init_log_var, init_log_state_prob = default_recursion_init(layout)

    # Posterior draws for volatility features and the canonical point
    # estimate for labels both run the recursion over the FULL centered
    # series (train+val+test) with a fixed theta from the train-only fit —
    # this is the causal, leakage-free forward-filter extension described
    # in the spec, not a refit.
    gen = torch.Generator().manual_seed(0)
    vol_features = compute_posterior_volatility_features(
        posterior, centered_returns, init_log_var, init_log_state_prob,
        layout=layout, n_draws=config["n_posterior_draws"], generator=gen,
    )
    labels = compute_regime_labels(
        posterior, centered_returns, init_log_var, init_log_state_prob, layout=layout
    )

    all_features = causal_features.join(vol_features, how="inner")
    assert not all_features.isna().any().any()

    X_train = all_features.loc[returns_train.index].to_numpy()
    X_val = all_features.loc[returns_val.index].to_numpy()
    X_test = all_features.loc[returns_test.index].to_numpy()
    y_train = labels.loc[returns_train.index].to_numpy()
    y_val = labels.loc[returns_val.index].to_numpy()
    y_test = labels.loc[returns_test.index].to_numpy()

    model = fit_ebm_classifier(X_train, y_train, random_state=0)
    class_to_idx = {c: i for i, c in enumerate(model.classes_)}

    scores_val = torch.tensor(predict_scores(model, X_val), dtype=torch.float32)
    y_val_idx = torch.tensor([class_to_idx[label] for label in y_val], dtype=torch.long)
    temperature = fit_temperature(scores_val, y_val_idx, n_steps=100)
    assert temperature > 0.0

    scores_test = torch.tensor(predict_scores(model, X_test), dtype=torch.float32)
    y_test_idx = torch.tensor([class_to_idx[label] for label in y_test], dtype=torch.long)
    calibrated_probs = apply_temperature(scores_test, temperature)

    # The model must actually have learned something, not just run.
    predicted_idx = calibrated_probs.argmax(dim=1)
    accuracy = (predicted_idx == y_test_idx).float().mean().item()
    majority_class_idx = int(torch.bincount(y_val_idx).argmax())
    baseline_accuracy = (y_test_idx == majority_class_idx).float().mean().item()
    assert accuracy >= baseline_accuracy

    nll_uncalibrated = torch.nn.functional.nll_loss(
        torch.log_softmax(scores_test, dim=1), y_test_idx
    ).item()
    nll_calibrated = torch.nn.functional.nll_loss(
        torch.log(calibrated_probs), y_test_idx
    ).item()
    assert np.isfinite(nll_calibrated)
    # calibration is fit on val, evaluated on test — not guaranteed to
    # strictly improve NLL on test, but must not be wildly worse.
    assert nll_calibrated < nll_uncalibrated * 2.0
