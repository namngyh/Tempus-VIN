# EBM Classifier + Calibration (Sub-project 2) Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Build a 4-class (Bull/Sideway/Bear/Stress) EBM classifier consuming causal OHLCV features plus MS-EGARCH posterior volatility summaries, with temperature calibration, trained and evaluated with zero leakage on real VN-Index data.

**Architecture:** New causal feature module (independent of MS-EGARCH) + a posterior-summary module wrapping the already-built MS-EGARCH fit/recursion (Sub-project 1) to produce both training features and target labels + a thin `ExplainableBoostingClassifier` wrapper + a PyTorch-based temperature-scaling calibrator. All new modules compose existing, already-tested Sub-project 1 infrastructure — no changes to `regime/ms_egarch.py`'s core recursion.

**Tech Stack:** Python 3.11, PyTorch, `interpret` (ExplainableBoostingClassifier), pandas, numpy — all already installed.

**Spec:** `docs/superpowers/specs/2026-08-17-ebm-classifier-design.md`

## Global Constraints

- EBM stays point-estimate (not Bayesian) — matches the original architecture decision.
- EBM input features must NOT include MS-EGARCH regime probabilities — only posterior volatility summary (`posterior_mean_sigma`, `posterior_sd_sigma`) plus 9 causal OHLCV features (11 columns total).
- Target label is a single deterministic sequence from `canonical_theta` (mean of pooled posterior `mu`s), not from the random draws used for volatility features — these two uses of the posterior must stay separate.
- Centering `returns` for the MS-EGARCH recursion must use the TRAIN partition's mean only, applied to the whole series (train+val+test) — never the whole-series mean, which would leak val/test information into the centering step.
- Split is chronological (70/15/15 train/val/test by position, no shuffling) — this is nowcasting, not h-step-ahead forecasting, so `validation/purged_split.py`'s horizon-based purge does not apply here.
- No tuning on the test set: temperature is fit on validation only; test is evaluated exactly once at the end.
- `ExplainableBoostingClassifier.classes_` is alphabetically ordered (`['Bear','Bull','Sideway','Stress']`), NOT `STATE_NAMES` order (`Bull/Sideway/Bear/Stress`) — every place that maps scores/probabilities to class names must use `model.classes_`, never assume `STATE_NAMES` order.
- Causal features use a 60-session warmup (longest rolling window is `drawdown_from_high_60`); the first 60 rows are dropped explicitly, never imputed.

---

## Task 1: Causal OHLCV features

**Files:**
- Create: `src/raemf_mc/features/causal.py`
- Test: `tests/features/test_causal.py`

**Interfaces:**
- Produces: `WARMUP_SESSIONS = 60`; `compute_causal_features(ohlcv: pd.DataFrame) -> pd.DataFrame` (input: DataFrame indexed by date with columns `open, high, low, close, volume`, e.g. `load_vnindex_ohlcv()`'s output; output: DataFrame with the same date index minus the first 60 rows, columns `ret_1, ret_5, ret_20, realized_vol_5, realized_vol_20, volume_change_1, relative_volume_20, high_low_range, drawdown_from_high_60`).

- [ ] **Step 1: Write the failing test**

```python
# tests/features/test_causal.py
import numpy as np
import pandas as pd
from raemf_mc.features.causal import compute_causal_features, WARMUP_SESSIONS


def _synthetic_ohlcv(n=80):
    dates = pd.date_range("2020-01-01", periods=n, freq="D")
    rng = np.random.default_rng(0)
    close = 100 * np.exp(np.cumsum(rng.normal(0, 0.01, n)))
    open_ = close * (1 + rng.normal(0, 0.002, n))
    high = np.maximum(open_, close) * (1 + np.abs(rng.normal(0, 0.003, n)))
    low = np.minimum(open_, close) * (1 - np.abs(rng.normal(0, 0.003, n)))
    volume = rng.integers(1000, 5000, n).astype(float)
    return pd.DataFrame(
        {"open": open_, "high": high, "low": low, "close": close, "volume": volume},
        index=dates,
    )


def test_warmup_rows_dropped_and_no_nan_remain():
    df = _synthetic_ohlcv(80)
    features = compute_causal_features(df)
    assert len(features) == 80 - WARMUP_SESSIONS
    assert features.index[0] == df.index[WARMUP_SESSIONS]
    assert not features.isna().any().any()


def test_ret_1_matches_manual_log_return():
    df = _synthetic_ohlcv(80)
    features = compute_causal_features(df)
    expected = np.log(df["close"] / df["close"].shift(1))
    pd.testing.assert_series_equal(
        features["ret_1"], expected.iloc[WARMUP_SESSIONS:], check_names=False
    )


def test_ret_5_and_ret_20_use_correct_lookback():
    df = _synthetic_ohlcv(80)
    features = compute_causal_features(df)
    t = 70
    expected_ret_5 = np.log(df["close"].iloc[t] / df["close"].iloc[t - 5])
    expected_ret_20 = np.log(df["close"].iloc[t] / df["close"].iloc[t - 20])
    row = features.loc[df.index[t]]
    assert np.isclose(row["ret_5"], expected_ret_5)
    assert np.isclose(row["ret_20"], expected_ret_20)


def test_realized_vol_5_matches_manual_rolling_std():
    df = _synthetic_ohlcv(80)
    features = compute_causal_features(df)
    ret_1 = np.log(df["close"] / df["close"].shift(1))
    expected = ret_1.rolling(5).std()
    pd.testing.assert_series_equal(
        features["realized_vol_5"], expected.iloc[WARMUP_SESSIONS:], check_names=False
    )


def test_drawdown_from_high_60_is_nonpositive():
    df = _synthetic_ohlcv(80)
    features = compute_causal_features(df)
    assert (features["drawdown_from_high_60"] <= 1e-9).all()


def test_relative_volume_20_matches_manual_ratio():
    df = _synthetic_ohlcv(80)
    features = compute_causal_features(df)
    expected = df["volume"] / df["volume"].rolling(20).mean()
    pd.testing.assert_series_equal(
        features["relative_volume_20"], expected.iloc[WARMUP_SESSIONS:], check_names=False
    )


def test_output_has_exactly_nine_columns():
    df = _synthetic_ohlcv(80)
    features = compute_causal_features(df)
    assert list(features.columns) == [
        "ret_1", "ret_5", "ret_20",
        "realized_vol_5", "realized_vol_20",
        "volume_change_1", "relative_volume_20",
        "high_low_range", "drawdown_from_high_60",
    ]
```

- [ ] **Step 2: Run tests to verify they fail**

Run: `python -m pytest tests/features/test_causal.py -v`
Expected: FAIL with `ModuleNotFoundError: No module named 'raemf_mc.features.causal'`

- [ ] **Step 3: Write implementation**

```python
# src/raemf_mc/features/causal.py
from __future__ import annotations

import numpy as np
import pandas as pd

WARMUP_SESSIONS = 60


def compute_causal_features(ohlcv: pd.DataFrame) -> pd.DataFrame:
    """Causal (no-lookahead) features from OHLCV, one row per session.

    Every feature is a pandas rolling/shift computation ending at the
    current row, so it never uses future data. The first WARMUP_SESSIONS
    rows lack enough history for the longest window (drawdown_from_high_60)
    and are dropped explicitly rather than imputed.
    """
    close = ohlcv["close"]
    volume = ohlcv["volume"]
    high = ohlcv["high"]
    low = ohlcv["low"]

    ret_1 = np.log(close / close.shift(1))
    ret_5 = np.log(close / close.shift(5))
    ret_20 = np.log(close / close.shift(20))
    realized_vol_5 = ret_1.rolling(5).std()
    realized_vol_20 = ret_1.rolling(20).std()
    volume_change_1 = np.log(volume / volume.shift(1))
    relative_volume_20 = volume / volume.rolling(20).mean()
    high_low_range = (high - low) / close
    rolling_max_60 = close.rolling(60).max()
    drawdown_from_high_60 = (close - rolling_max_60) / rolling_max_60

    features = pd.DataFrame(
        {
            "ret_1": ret_1,
            "ret_5": ret_5,
            "ret_20": ret_20,
            "realized_vol_5": realized_vol_5,
            "realized_vol_20": realized_vol_20,
            "volume_change_1": volume_change_1,
            "relative_volume_20": relative_volume_20,
            "high_low_range": high_low_range,
            "drawdown_from_high_60": drawdown_from_high_60,
        },
        index=ohlcv.index,
    )
    return features.iloc[WARMUP_SESSIONS:].copy()
```

```python
# src/raemf_mc/features/__init__.py already exists from sub-project 1 — do not recreate it.
```

- [ ] **Step 4: Run tests to verify they pass**

Run: `python -m pytest tests/features/test_causal.py -v`
Expected: PASS (7 tests)

- [ ] **Step 5: Commit**

```bash
git add src/raemf_mc/features/causal.py tests/features/test_causal.py
git commit -m "feat: add causal OHLCV feature engineering for EBM"
```

---

## Task 2: Chronological train/val/test split

**Files:**
- Create: `src/raemf_mc/validation/chronological_split.py`
- Test: `tests/validation/test_chronological_split.py`

**Interfaces:**
- Produces: `chronological_split(df: pd.DataFrame, train_frac: float = 0.7, val_frac: float = 0.15) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]`.

- [ ] **Step 1: Write the failing test**

```python
# tests/validation/test_chronological_split.py
import pandas as pd
from raemf_mc.validation.chronological_split import chronological_split


def test_split_sizes_and_order_for_100_rows():
    dates = pd.date_range("2020-01-01", periods=100, freq="D")
    df = pd.DataFrame({"x": range(100)}, index=dates)
    train, val, test = chronological_split(df, train_frac=0.7, val_frac=0.15)
    assert len(train) == 70
    assert len(val) == 15
    assert len(test) == 15


def test_split_preserves_chronological_order_with_no_overlap():
    dates = pd.date_range("2020-01-01", periods=100, freq="D")
    df = pd.DataFrame({"x": range(100)}, index=dates)
    train, val, test = chronological_split(df, train_frac=0.7, val_frac=0.15)
    assert train.index[-1] < val.index[0]
    assert val.index[-1] < test.index[0]
    all_idx = train.index.append(val.index).append(test.index)
    assert all_idx.is_monotonic_increasing
    assert len(all_idx) == len(df)


def test_split_uses_default_fractions():
    dates = pd.date_range("2020-01-01", periods=200, freq="D")
    df = pd.DataFrame({"x": range(200)}, index=dates)
    train, val, test = chronological_split(df)
    assert len(train) == 140
    assert len(val) == 30
    assert len(test) == 30
```

- [ ] **Step 2: Run tests to verify they fail**

Run: `python -m pytest tests/validation/test_chronological_split.py -v`
Expected: FAIL with `ModuleNotFoundError: No module named 'raemf_mc.validation.chronological_split'`

- [ ] **Step 3: Write implementation**

```python
# src/raemf_mc/validation/chronological_split.py
from __future__ import annotations

import pandas as pd


def chronological_split(
    df: pd.DataFrame, train_frac: float = 0.7, val_frac: float = 0.15
) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    """Split a time-ordered DataFrame into train/val/test by position,
    preserving chronological order — no shuffling, no purge-by-horizon
    (this is for nowcasting, not h-step-ahead forecasting). test receives
    the remainder after train_frac + val_frac."""
    n = len(df)
    n_train = int(n * train_frac)
    n_val = int(n * val_frac)
    train = df.iloc[:n_train]
    val = df.iloc[n_train : n_train + n_val]
    test = df.iloc[n_train + n_val :]
    return train, val, test
```

- [ ] **Step 4: Run tests to verify they pass**

Run: `python -m pytest tests/validation/test_chronological_split.py -v`
Expected: PASS (3 tests)

- [ ] **Step 5: Commit**

```bash
git add src/raemf_mc/validation/chronological_split.py tests/validation/test_chronological_split.py
git commit -m "feat: add chronological train/val/test split for nowcasting"
```

---

## Task 3: Posterior volatility features

**Files:**
- Create: `src/raemf_mc/regime/posterior_features.py`
- Test: `tests/regime/test_posterior_features.py`

**Interfaces:**
- Consumes (from Sub-project 1, already implemented and tested): `PooledPosterior`, `FitResult` (`src/raemf_mc/bayesian/torch_backend.py`); `MSEGARCHParamLayout`, `MSEGARCHParams`, `run_ms_egarch_recursion(returns: Tensor, params: MSEGARCHParams, init_log_var: Tensor, init_log_state_prob: Tensor) -> dict` (keys include `log_var_bar: Tensor` shape `(T,)`), `sample_ms_egarch_draw(posterior, layout, generator=None) -> MSEGARCHParams`, `unpack_params(theta, layout) -> MSEGARCHParams`, `default_recursion_init(layout, dtype, device) -> tuple[Tensor, Tensor]` (`src/raemf_mc/regime/ms_egarch.py`).
- Produces: `canonical_theta(posterior: PooledPosterior) -> torch.Tensor`; `compute_posterior_volatility_features(posterior, returns: pd.Series, init_log_var: torch.Tensor, init_log_state_prob: torch.Tensor, layout=MSEGARCHParamLayout(), n_draws: int = 50, generator: torch.Generator | None = None) -> pd.DataFrame` (columns `posterior_mean_sigma`, `posterior_sd_sigma`, index = `returns.index`).

- [ ] **Step 1: Write the failing test**

```python
# tests/regime/test_posterior_features.py
import math

import pandas as pd
import torch

from raemf_mc.bayesian.torch_backend import FitResult, PooledPosterior
from raemf_mc.regime.ms_egarch import MSEGARCHParamLayout, default_recursion_init
from raemf_mc.regime.posterior_features import (
    canonical_theta,
    compute_posterior_volatility_features,
)


def _fake_posterior(mus: list[torch.Tensor], layout: MSEGARCHParamLayout) -> PooledPosterior:
    results = [
        FitResult(
            mu=mu, log_sigma=torch.full((layout.total,), -3.0), elbo_trace=[0.0],
            completed_without_divergence=True, fallback_used=False,
            fallback_reason=None, n_retries=0, seed=i,
        )
        for i, mu in enumerate(mus)
    ]
    return PooledPosterior(seed_results=results)


def test_canonical_theta_is_mean_of_seed_mus():
    layout = MSEGARCHParamLayout()
    mu_a = torch.zeros(layout.total)
    mu_b = torch.full((layout.total,), 2.0)
    posterior = _fake_posterior([mu_a, mu_b], layout)
    theta = canonical_theta(posterior)
    assert torch.allclose(theta, torch.full((layout.total,), 1.0))


def test_posterior_volatility_features_shape_and_index():
    layout = MSEGARCHParamLayout()
    torch.manual_seed(0)
    mus = [torch.randn(layout.total) * 0.1 for _ in range(3)]
    posterior = _fake_posterior(mus, layout)
    dates = pd.date_range("2020-01-01", periods=15, freq="D")
    returns = pd.Series(torch.randn(15).numpy() * 0.01, index=dates)
    init_log_var, init_log_state_prob = default_recursion_init(layout)
    gen = torch.Generator().manual_seed(1)
    features = compute_posterior_volatility_features(
        posterior, returns, init_log_var, init_log_state_prob,
        layout=layout, n_draws=8, generator=gen,
    )
    assert list(features.columns) == ["posterior_mean_sigma", "posterior_sd_sigma"]
    assert list(features.index) == list(dates)
    assert (features["posterior_mean_sigma"] > 0).all()
    assert (features["posterior_sd_sigma"] >= 0).all()
    assert not features.isna().any().any()


def test_posterior_volatility_features_reproducible_with_fixed_generator():
    layout = MSEGARCHParamLayout()
    torch.manual_seed(2)
    mus = [torch.randn(layout.total) * 0.1 for _ in range(2)]
    posterior = _fake_posterior(mus, layout)
    dates = pd.date_range("2020-01-01", periods=10, freq="D")
    returns = pd.Series(torch.randn(10).numpy() * 0.01, index=dates)
    init_log_var, init_log_state_prob = default_recursion_init(layout)

    gen1 = torch.Generator().manual_seed(42)
    features_1 = compute_posterior_volatility_features(
        posterior, returns, init_log_var, init_log_state_prob,
        layout=layout, n_draws=5, generator=gen1,
    )
    gen2 = torch.Generator().manual_seed(42)
    features_2 = compute_posterior_volatility_features(
        posterior, returns, init_log_var, init_log_state_prob,
        layout=layout, n_draws=5, generator=gen2,
    )
    pd.testing.assert_frame_equal(features_1, features_2)
```

- [ ] **Step 2: Run tests to verify they fail**

Run: `python -m pytest tests/regime/test_posterior_features.py -v`
Expected: FAIL with `ModuleNotFoundError: No module named 'raemf_mc.regime.posterior_features'`

- [ ] **Step 3: Write implementation**

```python
# src/raemf_mc/regime/posterior_features.py
from __future__ import annotations

import pandas as pd
import torch

from raemf_mc.bayesian.torch_backend import PooledPosterior
from raemf_mc.regime.ms_egarch import (
    MSEGARCHParamLayout,
    run_ms_egarch_recursion,
    sample_ms_egarch_draw,
)


def canonical_theta(posterior: PooledPosterior) -> torch.Tensor:
    """Deterministic point-estimate parameter vector: the mean of the
    variational mean (mu) across all pooled seeds. Used to generate a
    single reproducible target-label sequence — distinct from the random
    posterior draws used for the volatility uncertainty features below,
    which need to reflect the posterior's actual spread, not collapse it."""
    mus = torch.stack([r.mu for r in posterior.seed_results])
    return mus.mean(dim=0)


def compute_posterior_volatility_features(
    posterior: PooledPosterior,
    returns: pd.Series,
    init_log_var: torch.Tensor,
    init_log_state_prob: torch.Tensor,
    layout: MSEGARCHParamLayout = MSEGARCHParamLayout(),
    n_draws: int = 50,
    generator: torch.Generator | None = None,
) -> pd.DataFrame:
    """Posterior-mean and posterior-sd of the collapsed sigma_bar_t at every
    session, estimated from n_draws independent posterior draws.

    sigma_bar_t = exp(0.5 * log_var_bar[t]) is the level-space-collapsed
    expected volatility at t under the filtered state distribution — the
    natural "posterior sigma_t" quantity, already computed by
    run_ms_egarch_recursion via Gray's collapsing (Sub-project 1).
    """
    returns_tensor = torch.tensor(returns.to_numpy(), dtype=torch.float32)
    sigma_draws = torch.zeros(n_draws, len(returns))
    for i in range(n_draws):
        params = sample_ms_egarch_draw(posterior, layout, generator=generator)
        result = run_ms_egarch_recursion(
            returns_tensor, params, init_log_var, init_log_state_prob
        )
        sigma_draws[i] = torch.exp(0.5 * result["log_var_bar"])
    posterior_mean_sigma = sigma_draws.mean(dim=0)
    posterior_sd_sigma = sigma_draws.std(dim=0, unbiased=False)
    return pd.DataFrame(
        {
            "posterior_mean_sigma": posterior_mean_sigma.numpy(),
            "posterior_sd_sigma": posterior_sd_sigma.numpy(),
        },
        index=returns.index,
    )
```

- [ ] **Step 4: Run tests to verify they pass**

Run: `python -m pytest tests/regime/test_posterior_features.py -v`
Expected: PASS (3 tests)

- [ ] **Step 5: Commit**

```bash
git add src/raemf_mc/regime/posterior_features.py tests/regime/test_posterior_features.py
git commit -m "feat: add posterior-mean/sd volatility features from MS-EGARCH"
```

---

## Task 4: Regime target labels

**Files:**
- Modify: `src/raemf_mc/regime/posterior_features.py` (append)
- Test: `tests/regime/test_posterior_features.py` (append)

**Interfaces:**
- Consumes: `canonical_theta` (Task 3, this file); `unpack_params`, `MSEGARCHParamLayout`, `run_ms_egarch_recursion` (`ms_egarch.py`); `align_states`, `apply_alignment`, `STATE_NAMES` (`src/raemf_mc/regime/state_alignment.py`, Sub-project 1).
- Produces: `compute_regime_labels(posterior: PooledPosterior, returns: pd.Series, init_log_var: torch.Tensor, init_log_state_prob: torch.Tensor, layout=MSEGARCHParamLayout()) -> pd.Series` (values are strings from `STATE_NAMES`, index = `returns.index`, name `"regime_label"`).

- [ ] **Step 1: Write the failing test**

```python
# append to tests/regime/test_posterior_features.py
from raemf_mc.regime.state_alignment import STATE_NAMES
from raemf_mc.regime.posterior_features import compute_regime_labels


def test_regime_labels_are_valid_state_names_and_reproducible():
    layout = MSEGARCHParamLayout()
    torch.manual_seed(3)
    mus = [torch.randn(layout.total) * 0.05 for _ in range(2)]
    posterior = _fake_posterior(mus, layout)
    dates = pd.date_range("2020-01-01", periods=40, freq="D")
    returns = pd.Series(torch.randn(40).numpy() * 0.01, index=dates)
    init_log_var, init_log_state_prob = default_recursion_init(layout)

    labels_1 = compute_regime_labels(
        posterior, returns, init_log_var, init_log_state_prob, layout=layout
    )
    labels_2 = compute_regime_labels(
        posterior, returns, init_log_var, init_log_state_prob, layout=layout
    )
    assert labels_1.name == "regime_label"
    assert list(labels_1.index) == list(dates)
    assert set(labels_1.unique()).issubset(set(STATE_NAMES))
    pd.testing.assert_series_equal(labels_1, labels_2)  # deterministic, no sampling
```

- [ ] **Step 2: Run test to verify it fails**

Run: `python -m pytest tests/regime/test_posterior_features.py -v`
Expected: FAIL with `ImportError: cannot import name 'compute_regime_labels'`

- [ ] **Step 3: Append implementation**

```python
# append to src/raemf_mc/regime/posterior_features.py

from raemf_mc.regime.ms_egarch import unpack_params
from raemf_mc.regime.state_alignment import STATE_NAMES, align_states, apply_alignment


def compute_regime_labels(
    posterior: PooledPosterior,
    returns: pd.Series,
    init_log_var: torch.Tensor,
    init_log_state_prob: torch.Tensor,
    layout: MSEGARCHParamLayout = MSEGARCHParamLayout(),
) -> pd.Series:
    """Deterministic target label per session: argmax of the train-aligned
    filtered regime probability, using the canonical (posterior-mean)
    parameter point estimate — a single reproducible ground-truth proxy
    sequence, not an ensemble (unlike the volatility features above, which
    deliberately use random draws to capture posterior spread)."""
    theta = canonical_theta(posterior)
    params = unpack_params(theta, layout)
    returns_tensor = torch.tensor(returns.to_numpy(), dtype=torch.float32)
    result = run_ms_egarch_recursion(
        returns_tensor, params, init_log_var, init_log_state_prob
    )
    permutation = align_states(returns_tensor, result["log_filtered_prob"])
    aligned = apply_alignment(result["log_filtered_prob"], permutation)
    label_idx = aligned.argmax(dim=1).tolist()
    labels = [STATE_NAMES[i] for i in label_idx]
    return pd.Series(labels, index=returns.index, name="regime_label")
```

- [ ] **Step 4: Run tests to verify they pass**

Run: `python -m pytest tests/regime/test_posterior_features.py -v`
Expected: PASS (4 tests)

- [ ] **Step 5: Commit**

```bash
git add src/raemf_mc/regime/posterior_features.py tests/regime/test_posterior_features.py
git commit -m "feat: add deterministic regime target-label generation"
```

---

## Task 5: EBM classifier wrapper

**Files:**
- Create: `src/raemf_mc/models/__init__.py`
- Create: `src/raemf_mc/models/ebm_classifier.py`
- Create: `tests/models/__init__.py`
- Test: `tests/models/test_ebm_classifier.py`

**Interfaces:**
- Produces: `fit_ebm_classifier(X_train: np.ndarray, y_train: np.ndarray, random_state: int = 0) -> ExplainableBoostingClassifier`; `predict_scores(model: ExplainableBoostingClassifier, X: np.ndarray) -> np.ndarray` (shape `(n_samples, n_classes)`, raw additive log-odds scores from `model.decision_function`, NOT probabilities).

- [ ] **Step 1: Write the failing test**

```python
# tests/models/test_ebm_classifier.py
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
```

- [ ] **Step 2: Run tests to verify they fail**

Run: `python -m pytest tests/models/test_ebm_classifier.py -v`
Expected: FAIL with `ModuleNotFoundError: No module named 'raemf_mc.models'`

- [ ] **Step 3: Write implementation**

```python
# src/raemf_mc/models/__init__.py
```

```python
# src/raemf_mc/models/ebm_classifier.py
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
```

```python
# tests/models/__init__.py
```

- [ ] **Step 4: Run tests to verify they pass**

Run: `python -m pytest tests/models/test_ebm_classifier.py -v`
Expected: PASS (3 tests)

- [ ] **Step 5: Commit**

```bash
git add src/raemf_mc/models/__init__.py src/raemf_mc/models/ebm_classifier.py tests/models/__init__.py tests/models/test_ebm_classifier.py
git commit -m "feat: add point-estimate EBM classifier wrapper"
```

---

## Task 6: Temperature calibration

**Files:**
- Create: `src/raemf_mc/models/calibration.py`
- Test: `tests/models/test_calibration.py`

**Interfaces:**
- Produces: `fit_temperature(scores_val: torch.Tensor, y_val: torch.Tensor, n_steps: int = 100) -> float` (`scores_val` shape `(n,4)` raw scores, `y_val` shape `(n,)` integer class indices matching `scores_val`'s column order); `apply_temperature(scores: torch.Tensor, temperature: float) -> torch.Tensor` (returns calibrated probabilities, shape `(n,4)`, rows sum to 1).

- [ ] **Step 1: Write the failing test**

```python
# tests/models/test_calibration.py
import torch
from raemf_mc.models.calibration import apply_temperature, fit_temperature


def test_apply_temperature_returns_valid_probabilities():
    scores = torch.tensor([[2.0, 0.5, -1.0, 0.0], [0.1, 0.2, 0.3, 0.4]])
    probs = apply_temperature(scores, temperature=1.5)
    assert probs.shape == (2, 4)
    assert torch.allclose(probs.sum(dim=1), torch.ones(2), atol=1e-6)
    assert (probs > 0).all()


def test_apply_temperature_one_is_a_no_op_softmax():
    scores = torch.tensor([[2.0, 0.5, -1.0, 0.0]])
    probs = apply_temperature(scores, temperature=1.0)
    expected = torch.softmax(scores, dim=1)
    assert torch.allclose(probs, expected, atol=1e-6)


def test_fit_temperature_reduces_nll_on_overconfident_scores():
    # Construct badly overconfident scores (huge magnitude) that rank
    # classes correctly on most rows but are WRONG on a noisy 20% subset —
    # the classic overconfidence scenario: a model that is both sharp and
    # sometimes wrong pays a huge NLL penalty on the wrong rows, which
    # softening (temperature > 1) reduces. (Scores that are always exactly
    # right at the argmax already have near-zero NLL regardless of
    # sharpness, so a noise-free construction wouldn't exercise this at
    # all — the label flip below is what makes the test meaningful.)
    torch.manual_seed(0)
    n = 300
    true_logits = torch.randn(n, 4) * 0.5
    y = true_logits.argmax(dim=1).clone()
    n_flip = n // 5
    flip_idx = torch.randperm(n)[:n_flip]
    y[flip_idx] = (y[flip_idx] + 1) % 4
    overconfident_scores = true_logits * 20.0  # same ranking, way too sharp

    nll_before = torch.nn.functional.nll_loss(
        torch.log_softmax(overconfident_scores, dim=1), y
    )
    temperature = fit_temperature(overconfident_scores, y, n_steps=200)
    calibrated = apply_temperature(overconfident_scores, temperature)
    nll_after = torch.nn.functional.nll_loss(torch.log(calibrated), y)

    assert temperature > 1.0  # should have learned to soften, not sharpen
    assert nll_after < nll_before
```

- [ ] **Step 2: Run tests to verify they fail**

Run: `python -m pytest tests/models/test_calibration.py -v`
Expected: FAIL with `ModuleNotFoundError: No module named 'raemf_mc.models.calibration'`

- [ ] **Step 3: Write implementation**

```python
# src/raemf_mc/models/calibration.py
from __future__ import annotations

import torch


def fit_temperature(
    scores_val: torch.Tensor, y_val: torch.Tensor, n_steps: int = 100
) -> float:
    """Fit a single scalar temperature T > 0 minimizing NLL of
    softmax(scores/T) against y_val (integer class indices matching
    scores_val's column order). Parametrized via softplus to keep T
    positive throughout optimization, initialized near T=1.0
    (softplus(0.5413) ~= 1.0, since log(e-1) ~= 0.5413)."""
    raw_t = torch.nn.Parameter(torch.tensor(0.5413))
    optimizer = torch.optim.Adam([raw_t], lr=0.05)
    for _ in range(n_steps):
        optimizer.zero_grad()
        t = torch.nn.functional.softplus(raw_t)
        log_probs = torch.log_softmax(scores_val / t, dim=1)
        loss = torch.nn.functional.nll_loss(log_probs, y_val)
        loss.backward()
        optimizer.step()
    return float(torch.nn.functional.softplus(raw_t).item())


def apply_temperature(scores: torch.Tensor, temperature: float) -> torch.Tensor:
    """Return calibrated class probabilities: softmax(scores / temperature)."""
    return torch.softmax(scores / temperature, dim=1)
```

- [ ] **Step 4: Run tests to verify they pass**

Run: `python -m pytest tests/models/test_calibration.py -v`
Expected: PASS (3 tests)

- [ ] **Step 5: Commit**

```bash
git add src/raemf_mc/models/calibration.py tests/models/test_calibration.py
git commit -m "feat: add temperature scaling calibration for EBM scores"
```

---

## Task 7: EBM smoke config

**Files:**
- Create: `configs/ebm_smoke.yaml`

**Interfaces:**
- Produces: a YAML file consumed by Task 8's integration test — no config-loader module (mirrors Sub-project 1's `cpu_smoke.yaml` pattern: read directly via `yaml.safe_load` in the test).

- [ ] **Step 1: Write `configs/ebm_smoke.yaml`**

```yaml
# Profile debug nhanh cho EBM: cua so du lieu ngan hon toan bo, it buoc ADVI,
# 1 seed cho MS-EGARCH, it draw cho dac trung bat dinh. Dung de lap nhanh.
device_preference: auto
window_sessions: 1500      # ~6 nam gan nhat, du de co ca 4 regime
train_frac: 0.7
val_frac: 0.15
n_posterior_draws: 20
seeds: [0]
advi:
  n_steps: 300
  learning_rate: 0.05
  warmup_steps: 20
  grad_clip_norm: 10.0
  elbo_ma_window: 20
  early_stop_patience: 100
  min_delta: 0.001
  retry_lr_factor: 0.5
  max_retries: 3
  n_mc_samples: 4
prior:
  hyper_mean_scale: 1.0
  min_effective_observations: 30.0
  min_effective_fraction: 0.05
```

- [ ] **Step 2: Verify it parses as valid YAML**

Run: `python -c "import yaml; print(yaml.safe_load(open('configs/ebm_smoke.yaml')))"`
Expected: prints a dict with keys `device_preference, window_sessions, train_frac, val_frac, n_posterior_draws, seeds, advi, prior` and no exception.

- [ ] **Step 3: Commit**

```bash
git add configs/ebm_smoke.yaml
git commit -m "chore: add ebm_smoke ADVI config profile"
```

---

## Task 8: End-to-end integration test on real data

**Files:**
- Test: `tests/test_ebm_integration_smoke.py`

**Interfaces:**
- Consumes: `load_vnindex_ohlcv` (`src/raemf_mc/data/loader.py`); `compute_log_returns` (`src/raemf_mc/features/returns.py`); `compute_causal_features` (Task 1); `chronological_split` (Task 2); `default_recursion_init`, `fit_ms_egarch`, `MSEGARCHParamLayout` (`src/raemf_mc/regime/ms_egarch.py`); `canonical_theta`, `compute_posterior_volatility_features`, `compute_regime_labels` (Tasks 3-4); `fit_ebm_classifier`, `predict_scores` (Task 5); `fit_temperature`, `apply_temperature` (Task 6); `AdviConfig` (`src/raemf_mc/bayesian/torch_backend.py`); `HierarchicalPriorConfig` (`src/raemf_mc/bayesian/priors.py`); `configs/ebm_smoke.yaml` (Task 7).

This is the task most likely to surface a genuine integration mismatch between pieces that were each tested in isolation with small synthetic inputs — per Sub-project 1's own Task 18, finding and fixing a real bug here (if the implementation reveals one) is expected, normal work, not something to paper over with a loosened assertion.

- [ ] **Step 1: Write the integration test**

```python
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
```

- [ ] **Step 2: Run the test and debug any real integration failure**

Run: `python -m pytest tests/test_ebm_integration_smoke.py -v -s`
Expected: either PASS directly, or a real integration bug to diagnose and fix at its root cause (in whichever earlier task's file actually has the bug) — this is the first test in this plan allowed to fail for a substantive reason.

- [ ] **Step 3: Iterate until it passes**

Run: `python -m pytest tests/test_ebm_integration_smoke.py -v -s`
Expected: PASS

- [ ] **Step 4: Commit**

```bash
git add tests/test_ebm_integration_smoke.py
git commit -m "test: add end-to-end EBM pipeline smoke test on real VN-Index data"
```

---

## Task 9: Full suite verification

**Files:** none (verification only)

- [ ] **Step 1: Run the full test suite**

Run: `python -m pytest -q` (budget 20+ minutes — this now includes both Sub-project 1's real ADVI integration test and this plan's Task 8 real ADVI fit)
Expected: all tests pass, 0 failures

- [ ] **Step 2: Run lint**

Run: `python -m ruff check src tests scripts`
Expected: no errors (fix any reported issues, re-run until clean)

- [ ] **Step 3: Final commit if any lint fixes were needed**

```bash
git add -A
git commit -m "chore: fix lint issues found in full-suite verification"
```

(Skip this commit if step 2 was already clean with no changes.)
