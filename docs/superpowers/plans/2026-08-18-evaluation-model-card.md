# OOS Evaluation + Model Card Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Finish the classification-evaluation work deferred from sub-project
2 (macro F1, per-class recall, NLL), add a walk-forward out-of-sample
harness for the scenario-return/MC layer (CRPS, WIS, Kupiec VaR backtest)
at smoke scale, and write a model card summarizing the full system.

**Architecture:** Four small, independent evaluation modules
(`evaluation/classification_metrics.py`, `evaluation/probabilistic_metrics.py`,
`evaluation/var_backtest.py`, `evaluation/walk_forward.py`) that only READ
existing sub-project 1-3 outputs — nothing upstream changes. The walk-forward
harness is the one orchestration piece: it refits MS-EGARCH + mu_k at each
of N historical cutoffs (reusing `fit_ms_egarch`/`fit_regime_mu`/
`simulate_mc_paths` unchanged) and scores the simulated forecast against
what actually happened next, which real OOS evaluation requires and no
earlier sub-project's integration test did (they all fit-and-evaluate on
the same window).

**Tech Stack:** scikit-learn (already a dependency, for classification
metrics — no need to hand-roll F1/recall/confusion-matrix), numpy/pandas
(CRPS/WIS/Kupiec — pure math, no new dependency), the project's existing
PyTorch ADVI/simulation stack (unchanged, read-only).

**Spec:** docs/superpowers/specs/2026-08-18-evaluation-model-card-design.md

## Global Constraints

- No changes to any sub-project 1-3 file (`regime/ms_egarch.py`,
  `bayesian/priors.py`, `models/*.py`, `scenario/*.py`, `risk/metrics.py`)
  — this sub-project only consumes their existing public functions.
- `model.classes_` is alphabetical, NOT `STATE_NAMES` order — any place
  mapping EBM scores/probabilities back to class names must use
  `model.classes_`, never assume `STATE_NAMES` order (same rule as
  sub-project 2).
- NLL must be computed via `apply_temperature_log_prob` (log-domain) or an
  equivalent log-domain path, never `log(probability)` directly — this is
  the exact underflow bug sub-project 2's fix wave already found and fixed
  once; do not reintroduce it.
- Walk-forward's per-cutoff fit MUST center returns using that cutoff's own
  fit-window mean, never the whole-series mean — same leakage discipline as
  every earlier sub-project.
- `simulate_mc_paths` returns CENTERED daily returns (per its own
  docstring, established in sub-project 3's final-review fix). Any
  real-world comparison (CRPS/WIS/VaR breach against a realized return)
  must add the fit-window's own mean back first — this is the exact drift
  round-trip bug sub-project 3's whole-branch review found and fixed once;
  do not reintroduce it.
- Walk-forward smoke-scale (5-10 cutoffs, lightweight ADVI) is for
  verifying the pipeline is wired correctly — it is explicitly NOT a
  statistically meaningful OOS benchmark (the Kupiec test in particular has
  no real power at this `n`). Every test/docstring that reports numbers
  from it must say so.

---

## Task 1: Classification evaluation metrics

**Files:**
- Create: `src/raemf_mc/evaluation/__init__.py` (empty)
- Create: `src/raemf_mc/evaluation/classification_metrics.py`
- Test: `tests/evaluation/test_classification_metrics.py`

**Interfaces:**
- Produces: `compute_classification_report(y_true: Sequence[str], y_pred: Sequence[str], labels: Sequence[str] = STATE_NAMES) -> dict` (keys: `macro_f1: float`, `recall_by_class: dict[str, float]`, `confusion_matrix: pd.DataFrame` indexed/columned by `labels`); `compute_nll(log_probs: torch.Tensor, y_true_idx: torch.Tensor) -> float`.
- Consumes: `sklearn.metrics.f1_score/recall_score/confusion_matrix`; `STATE_NAMES` (from `raemf_mc.regime.state_alignment`).

- [ ] **Step 1: Write the failing test**

```python
# tests/evaluation/test_classification_metrics.py
import numpy as np
import torch

from raemf_mc.evaluation.classification_metrics import (
    compute_classification_report,
    compute_nll,
)


def test_compute_classification_report_hand_computed():
    # 3 classes, one wrong prediction (Bear predicted as Sideway once).
    y_true = ["Bull", "Bull", "Bear", "Bear", "Sideway"]
    y_pred = ["Bull", "Bull", "Bear", "Sideway", "Sideway"]
    labels = ["Bull", "Sideway", "Bear", "Stress"]

    report = compute_classification_report(y_true, y_pred, labels=labels)

    # Bull: 2/2 correct -> recall 1.0. Sideway: 1/1 correct -> recall 1.0.
    # Bear: 1/2 correct -> recall 0.5. Stress: no true instances -> recall 0.0 (zero_division=0).
    assert report["recall_by_class"]["Bull"] == 1.0
    assert report["recall_by_class"]["Sideway"] == 1.0
    assert abs(report["recall_by_class"]["Bear"] - 0.5) < 1e-9
    assert report["recall_by_class"]["Stress"] == 0.0

    # macro F1 over 4 labels: Bull F1=1.0 (precision=1,recall=1),
    # Sideway: precision=1/2=0.5 (2 predicted, 1 correct), recall=1.0 -> F1=2*0.5*1/(0.5+1)=0.6667
    # Bear: precision=1.0 (1 predicted, 1 correct), recall=0.5 -> F1=2*1*0.5/1.5=0.6667
    # Stress: precision=0 (no predictions), recall=0 -> F1=0
    # macro = (1.0 + 0.6667 + 0.6667 + 0.0) / 4 = 0.5833
    assert abs(report["macro_f1"] - 0.5833) < 1e-3

    cm = report["confusion_matrix"]
    assert list(cm.index) == labels
    assert list(cm.columns) == labels
    assert cm.loc["Bear", "Sideway"] == 1  # the one misclassification
    assert cm.loc["Bull", "Bull"] == 2


def test_compute_nll_uses_log_domain_not_log_of_probability():
    # Two samples, true class index 0 and 1, log-probs already computed
    # (as apply_temperature_log_prob would produce) rather than probabilities.
    log_probs = torch.log_softmax(torch.tensor([[2.0, 0.0], [0.0, 2.0]]), dim=1)
    y_true_idx = torch.tensor([0, 1])
    nll = compute_nll(log_probs, y_true_idx)
    # Both predictions are confident and correct -> low NLL.
    expected = float(-log_probs[0, 0].item() - log_probs[1, 1].item()) / 2
    assert abs(nll - expected) < 1e-6
    assert nll < 0.5  # sanity: confident-correct predictions have low NLL
```

- [ ] **Step 2: Run tests to verify they fail**

Run: `python -m pytest tests/evaluation/test_classification_metrics.py -v`
Expected: FAIL with `ModuleNotFoundError: No module named 'raemf_mc.evaluation'`

- [ ] **Step 3: Implement**

```python
# src/raemf_mc/evaluation/__init__.py
```

```python
# src/raemf_mc/evaluation/classification_metrics.py
from __future__ import annotations

from typing import Sequence

import pandas as pd
import torch
from sklearn.metrics import confusion_matrix, f1_score, recall_score


def compute_classification_report(
    y_true: Sequence[str], y_pred: Sequence[str], labels: Sequence[str]
) -> dict:
    """Macro F1, per-class recall, and a labeled confusion matrix. Reuses
    scikit-learn's implementations rather than hand-rolling them — only the
    labeling/packaging is specific to this project."""
    labels_list = list(labels)
    macro_f1 = float(
        f1_score(y_true, y_pred, labels=labels_list, average="macro", zero_division=0)
    )
    recalls = recall_score(y_true, y_pred, labels=labels_list, average=None, zero_division=0)
    recall_by_class = {label: float(r) for label, r in zip(labels_list, recalls)}
    cm = confusion_matrix(y_true, y_pred, labels=labels_list)
    cm_df = pd.DataFrame(cm, index=labels_list, columns=labels_list)
    return {
        "macro_f1": macro_f1,
        "recall_by_class": recall_by_class,
        "confusion_matrix": cm_df,
    }


def compute_nll(log_probs: torch.Tensor, y_true_idx: torch.Tensor) -> float:
    """Mean negative log-likelihood from LOG-probabilities (e.g. from
    apply_temperature_log_prob), never from log(probability) directly —
    that path underflows to -inf on confidently-wrong predictions, exactly
    the bug sub-project 2 already found and fixed once for this same
    calculation."""
    return float(torch.nn.functional.nll_loss(log_probs, y_true_idx).item())
```

- [ ] **Step 4: Run tests to verify they pass**

Run: `python -m pytest tests/evaluation/test_classification_metrics.py -v`
Expected: PASS

- [ ] **Step 5: Commit**

```bash
git add src/raemf_mc/evaluation/__init__.py src/raemf_mc/evaluation/classification_metrics.py tests/evaluation/test_classification_metrics.py
git commit -m "feat: add classification evaluation metrics (macro F1, per-class recall, NLL)"
```

---

## Task 2: CRPS and WIS

**Files:**
- Create: `src/raemf_mc/evaluation/probabilistic_metrics.py`
- Test: `tests/evaluation/test_probabilistic_metrics.py`

**Interfaces:**
- Produces: `compute_crps(samples: np.ndarray, y: float) -> float`; `compute_wis(samples: np.ndarray, y: float, alpha_levels: tuple[float, ...] = (0.5, 0.2, 0.1, 0.05, 0.02)) -> float`.

- [ ] **Step 1: Write the failing test**

```python
# tests/evaluation/test_probabilistic_metrics.py
import numpy as np

from raemf_mc.evaluation.probabilistic_metrics import compute_crps, compute_wis


def test_compute_crps_hand_computed_two_point_distribution():
    # Empirical distribution: two equally-likely points {1, 3}. Realized y=2.
    # Hand integral of (F(x) - 1{x>=y})^2: 0.25 on [1,2) + 0.25 on [2,3) = 0.5.
    samples = np.array([1.0, 3.0])
    crps = compute_crps(samples, y=2.0)
    assert abs(crps - 0.5) < 1e-9


def test_compute_wis_realized_inside_interval():
    # 101 evenly-spaced points 0..100 so quantiles land on exact integers:
    # quantile(0.25)=25, quantile(0.75)=75 (position=(101-1)*q).
    samples = np.linspace(0, 100, 101)
    wis = compute_wis(samples, y=50.0, alpha_levels=(0.5,))
    # median=50, IS_0.5 = (75-25) + 0 (y inside [25,75]) = 50.
    # WIS = (1/(1+0.5)) * (0.5*|50-50| + (0.5/2)*50) = (1/1.5)*12.5 = 8.3333...
    assert abs(wis - 8.3333) < 1e-3


def test_compute_wis_realized_outside_interval():
    samples = np.linspace(0, 100, 101)
    wis = compute_wis(samples, y=90.0, alpha_levels=(0.5,))
    # l=25, u=75, y=90 > u: IS = (75-25) + (2/0.5)*(90-75) = 50 + 60 = 110.
    # WIS = (1/1.5) * (0.5*|90-50| + 0.25*110) = (1/1.5) * (20 + 27.5) = 31.6667
    assert abs(wis - 31.6667) < 1e-3
```

- [ ] **Step 2: Run tests to verify they fail**

Run: `python -m pytest tests/evaluation/test_probabilistic_metrics.py -v`
Expected: FAIL with `ModuleNotFoundError`

- [ ] **Step 3: Implement**

```python
# src/raemf_mc/evaluation/probabilistic_metrics.py
from __future__ import annotations

import numpy as np


def compute_crps(samples: np.ndarray, y: float) -> float:
    """CRPS of the empirical distribution given by `samples` against a
    realized value `y`, via the exact closed form for an empirical CDF:
    CRPS(F, y) = E|X - y| - 0.5*E|X - X'|, X,X' iid ~ F. Not an
    approximation — this equals the integral definition exactly for a
    discrete empirical distribution. O(M^2) in the sample count; fine at
    M ~ a few hundred to a thousand Monte Carlo paths."""
    term1 = float(np.mean(np.abs(samples - y)))
    diffs = np.abs(samples[:, None] - samples[None, :])
    term2 = 0.5 * float(np.mean(diffs))
    return term1 - term2


def compute_wis(
    samples: np.ndarray,
    y: float,
    alpha_levels: tuple[float, ...] = (0.5, 0.2, 0.1, 0.05, 0.02),
) -> float:
    """Weighted Interval Score (Bracher et al. 2021) of the empirical
    distribution given by `samples` against a realized value `y`.
    `alpha_levels` are TWO-SIDED central-interval levels (0.5 -> the 50%
    interval, i.e. quantiles [0.25, 0.75]) — unrelated to this project's
    one-sided VaR alpha convention; do not conflate the two."""
    median = float(np.median(samples))
    k = len(alpha_levels)
    total = 0.5 * abs(y - median)
    for alpha in alpha_levels:
        lo = float(np.quantile(samples, alpha / 2))
        hi = float(np.quantile(samples, 1 - alpha / 2))
        interval_score = hi - lo
        if y < lo:
            interval_score += (2.0 / alpha) * (lo - y)
        elif y > hi:
            interval_score += (2.0 / alpha) * (y - hi)
        total += (alpha / 2.0) * interval_score
    return total / (k + 0.5)
```

- [ ] **Step 4: Run tests to verify they pass**

Run: `python -m pytest tests/evaluation/test_probabilistic_metrics.py -v`
Expected: PASS

- [ ] **Step 5: Commit**

```bash
git add src/raemf_mc/evaluation/probabilistic_metrics.py tests/evaluation/test_probabilistic_metrics.py
git commit -m "feat: add CRPS and WIS probabilistic forecast metrics"
```

---

## Task 3: Kupiec VaR backtest

**Files:**
- Create: `src/raemf_mc/evaluation/var_backtest.py`
- Test: `tests/evaluation/test_var_backtest.py`

**Interfaces:**
- Produces: `compute_kupiec_lr(n: int, x: int, p: float) -> float`.

- [ ] **Step 1: Write the failing test**

```python
# tests/evaluation/test_var_backtest.py
import math

from raemf_mc.evaluation.var_backtest import compute_kupiec_lr


def test_compute_kupiec_lr_hand_computed():
    # n=10 observations, x=1 violation, p=0.05 (matches VaR_95's expected
    # violation rate). Hand-computed (natural logs):
    # log_L0 = 9*ln(0.95) + 1*ln(0.05) = -3.4573720
    # log_L1 = 9*ln(0.9)  + 1*ln(0.1)  = -3.2508296   (phat = 1/10 = 0.1)
    # LR = -2*(log_L0 - log_L1) = 0.4130848
    lr = compute_kupiec_lr(n=10, x=1, p=0.05)
    assert abs(lr - 0.4130848) < 1e-5


def test_compute_kupiec_lr_zero_violations_edge_case():
    # x=0: log_L1 = 0 exactly (saturated model perfectly fits phat=0).
    # log_L0 = 5*ln(0.95) = -0.2564665
    # LR = -2*(-0.2564665 - 0) = 0.512933
    lr = compute_kupiec_lr(n=5, x=0, p=0.05)
    assert abs(lr - 0.512933) < 1e-5
    assert math.isfinite(lr)


def test_compute_kupiec_lr_all_violations_edge_case():
    # x=n: log_L1 = 0 exactly (saturated model perfectly fits phat=1).
    lr = compute_kupiec_lr(n=5, x=5, p=0.05)
    assert math.isfinite(lr)
    assert lr > 0
```

- [ ] **Step 2: Run tests to verify they fail**

Run: `python -m pytest tests/evaluation/test_var_backtest.py -v`
Expected: FAIL with `ModuleNotFoundError`

- [ ] **Step 3: Implement**

```python
# src/raemf_mc/evaluation/var_backtest.py
from __future__ import annotations

import math


def compute_kupiec_lr(n: int, x: int, p: float) -> float:
    """Kupiec (1995) unconditional-coverage likelihood-ratio statistic for
    a VaR backtest. n: number of observations. x: number of VaR violations
    (realized loss exceeded VaR). p: the VaR's expected violation
    probability (e.g. 0.05 for a 95% VaR). Under H0 (correctly calibrated
    VaR), LR ~ chi-squared(1); compare against 3.841 (95% critical value)
    to test whether the observed violation rate differs significantly from
    p.

    Caveat: with a small n (e.g. the 5-10 walk-forward cutoffs this
    sub-project runs at smoke scale), this test has essentially no
    statistical power — it verifies the FORMULA, not a trustworthy
    calibration conclusion. See configs/eval_smoke.yaml and the walk-forward
    module docstring.
    """
    log_l0 = (n - x) * math.log(1 - p) + x * math.log(p)
    if x == 0 or x == n:
        # The unrestricted (saturated) model with phat = x/n exactly 0 or 1
        # fits perfectly: likelihood 1, log-likelihood 0. Computing
        # x*log(phat) or (n-x)*log(1-phat) directly would be 0*log(0), a
        # floating-point NaN despite the analytic limit being exactly 0.
        log_l1 = 0.0
    else:
        phat = x / n
        log_l1 = (n - x) * math.log(1 - phat) + x * math.log(phat)
    return -2.0 * (log_l0 - log_l1)
```

- [ ] **Step 4: Run tests to verify they pass**

Run: `python -m pytest tests/evaluation/test_var_backtest.py -v`
Expected: PASS

- [ ] **Step 5: Commit**

```bash
git add src/raemf_mc/evaluation/var_backtest.py tests/evaluation/test_var_backtest.py
git commit -m "feat: add Kupiec VaR backtest likelihood-ratio statistic"
```

---

## Task 4: Walk-forward OOS harness

**Files:**
- Create: `src/raemf_mc/evaluation/walk_forward.py`
- Test: `tests/evaluation/test_walk_forward.py`

**Interfaces:**
- Consumes: `fit_ms_egarch`, `default_recursion_init`, `MSEGARCHParamLayout` (`raemf_mc.regime.ms_egarch`); `fit_regime_mu` (`raemf_mc.scenario.mu_fit`); `simulate_mc_paths` (`raemf_mc.scenario.simulate`); `compute_var` (`raemf_mc.risk.metrics`); `compute_crps`, `compute_wis` (Task 2); `compute_kupiec_lr` (Task 3); `AdviConfig` (`raemf_mc.bayesian.torch_backend`); `HierarchicalPriorConfig` (`raemf_mc.bayesian.priors`).
- Produces: `run_walk_forward_evaluation(log_returns: pd.Series, cutoffs: list[int], horizon: int, ms_egarch_advi: AdviConfig, ms_egarch_prior: HierarchicalPriorConfig, mu_advi: AdviConfig, mu_n_draws: int, mu_prior_scale: float, mu_min_effective_observations: float, mu_min_effective_fraction: float, mc_n_paths: int, layout: MSEGARCHParamLayout, seeds: list[int], device: torch.device, fallback_log_dir: Path, var_alphas: tuple[float, ...] = (0.95, 0.99), generator: torch.Generator | None = None) -> tuple[pd.DataFrame, dict[str, float]]` — per-cutoff DataFrame (columns: `cutoff, realized_return, crps, wis, var_95, breached_95, var_99, breached_99`) plus a dict of aggregated Kupiec LR statistics (keys `kupiec_lr_95`, `kupiec_lr_99`).

- [ ] **Step 1: Write the failing test**

```python
# tests/evaluation/test_walk_forward.py
from pathlib import Path

import pandas as pd
import torch

from raemf_mc.bayesian.priors import HierarchicalPriorConfig
from raemf_mc.bayesian.torch_backend import AdviConfig
from raemf_mc.regime.ms_egarch import MSEGARCHParamLayout
from raemf_mc.evaluation.walk_forward import run_walk_forward_evaluation


def test_run_walk_forward_evaluation_shape_and_invariants(tmp_path):
    layout = MSEGARCHParamLayout()
    torch.manual_seed(0)
    # Enough synthetic data for 2 cutoffs with a 5-day horizon and a
    # reasonable fit window each.
    dates = pd.date_range("2020-01-01", periods=120, freq="D")
    log_returns = pd.Series(torch.randn(120).numpy() * 0.01, index=dates)

    advi_config = AdviConfig(n_steps=20, learning_rate=0.05, warmup_steps=5,
                              elbo_ma_window=5, early_stop_patience=20)
    prior_config = HierarchicalPriorConfig()
    fallback_dir = tmp_path / "fallbacks"
    fallback_dir.mkdir()

    results_df, kupiec = run_walk_forward_evaluation(
        log_returns, cutoffs=[80, 100], horizon=5,
        ms_egarch_advi=advi_config, ms_egarch_prior=prior_config,
        mu_advi=advi_config, mu_n_draws=3, mu_prior_scale=0.01,
        mu_min_effective_observations=10.0, mu_min_effective_fraction=0.05,
        mc_n_paths=50, layout=layout, seeds=[0], device=torch.device("cpu"),
        fallback_log_dir=fallback_dir, generator=torch.Generator().manual_seed(1),
    )

    assert list(results_df["cutoff"]) == [80, 100]
    assert results_df["crps"].ge(0).all()
    assert results_df["wis"].ge(0).all()
    for col in ("var_95", "var_99"):
        assert results_df[col].notna().all()
    assert set(kupiec.keys()) == {"kupiec_lr_95", "kupiec_lr_99"}
    for v in kupiec.values():
        assert v == v  # not NaN
        assert v >= 0.0


def test_run_walk_forward_evaluation_rejects_cutoff_too_close_to_end():
    layout = MSEGARCHParamLayout()
    dates = pd.date_range("2020-01-01", periods=30, freq="D")
    log_returns = pd.Series([0.01] * 30, index=dates)
    advi_config = AdviConfig(n_steps=5)
    prior_config = HierarchicalPriorConfig()
    import pytest
    with pytest.raises(ValueError, match="horizon"):
        run_walk_forward_evaluation(
            log_returns, cutoffs=[28], horizon=5,
            ms_egarch_advi=advi_config, ms_egarch_prior=prior_config,
            mu_advi=advi_config, mu_n_draws=2, mu_prior_scale=0.01,
            mu_min_effective_observations=10.0, mu_min_effective_fraction=0.05,
            mc_n_paths=10, layout=layout, seeds=[0], device=torch.device("cpu"),
            fallback_log_dir=Path("."),
        )
```

- [ ] **Step 2: Run tests to verify they fail**

Run: `python -m pytest tests/evaluation/test_walk_forward.py -v`
Expected: FAIL with `ModuleNotFoundError`

- [ ] **Step 3: Implement**

```python
# src/raemf_mc/evaluation/walk_forward.py
from __future__ import annotations

from pathlib import Path

import pandas as pd
import torch

from raemf_mc.bayesian.priors import HierarchicalPriorConfig
from raemf_mc.bayesian.torch_backend import AdviConfig
from raemf_mc.evaluation.probabilistic_metrics import compute_crps, compute_wis
from raemf_mc.evaluation.var_backtest import compute_kupiec_lr
from raemf_mc.regime.ms_egarch import MSEGARCHParamLayout, default_recursion_init, fit_ms_egarch
from raemf_mc.risk.metrics import compute_var
from raemf_mc.scenario.mu_fit import fit_regime_mu
from raemf_mc.scenario.simulate import simulate_mc_paths


def run_walk_forward_evaluation(
    log_returns: pd.Series,
    cutoffs: list[int],
    horizon: int,
    ms_egarch_advi: AdviConfig,
    ms_egarch_prior: HierarchicalPriorConfig,
    mu_advi: AdviConfig,
    mu_n_draws: int,
    mu_prior_scale: float,
    mu_min_effective_observations: float,
    mu_min_effective_fraction: float,
    mc_n_paths: int,
    layout: MSEGARCHParamLayout,
    seeds: list[int],
    device: torch.device,
    fallback_log_dir: Path,
    var_alphas: tuple[float, ...] = (0.95, 0.99),
    generator: torch.Generator | None = None,
) -> tuple[pd.DataFrame, dict[str, float]]:
    """Walk-forward OOS evaluation: at each cutoff, fit MS-EGARCH + mu_k on
    ONLY the data up to that cutoff (never later data -- this is what makes
    it genuinely out-of-sample, unlike every earlier sub-project's
    integration test, which fits and evaluates on the same window),
    simulate `horizon` days forward, and score the simulated distribution
    against the return that actually occurred next in the historical
    record.

    Caveat: at smoke scale (few cutoffs, lightweight ADVI -- see
    configs/eval_smoke.yaml) this verifies the pipeline is wired correctly.
    It is NOT a statistically meaningful OOS benchmark: the aggregated
    Kupiec test in particular needs dozens+ cutoffs to have any power, and
    CRPS/WIS both directly reflect the quality of a smoke-scale (likely
    under-converged) ADVI fit, not a property of the metric itself.
    """
    rows = []
    for i, cutoff in enumerate(cutoffs):
        future_returns = log_returns.iloc[cutoff : cutoff + horizon]
        if len(future_returns) < horizon:
            raise ValueError(
                f"cutoff {cutoff} + horizon {horizon} exceeds available data "
                f"(series has {len(log_returns)} rows)"
            )
        fit_returns = log_returns.iloc[:cutoff]
        train_mean = float(fit_returns.mean())
        centered_fit = fit_returns - train_mean
        returns_tensor = torch.tensor(
            centered_fit.to_numpy(), dtype=torch.float32, device=device
        )

        ms_egarch_posterior = fit_ms_egarch(
            returns_tensor, ms_egarch_advi, ms_egarch_prior, seeds, device, layout,
            fallback_log_path=fallback_log_dir / f"cutoff_{cutoff}_ms_egarch.json",
        )
        init_log_var, init_log_state_prob = default_recursion_init(layout, device=device)
        mu_posterior = fit_regime_mu(
            returns_tensor, ms_egarch_posterior, mu_advi, seeds, device,
            init_log_var, init_log_state_prob, layout=layout,
            n_draws=mu_n_draws, mu_prior_scale=mu_prior_scale,
            min_effective_observations=mu_min_effective_observations,
            min_effective_fraction=mu_min_effective_fraction,
            generator=generator,
            fallback_log_path=fallback_log_dir / f"cutoff_{cutoff}_mu.json",
        )
        daily_paths = simulate_mc_paths(
            ms_egarch_posterior, mu_posterior, returns_tensor,
            init_log_var, init_log_state_prob,
            n_paths=mc_n_paths, horizon=horizon, layout=layout, device=device,
            generator=generator,
            fallback_log_path=fallback_log_dir / f"cutoff_{cutoff}_mc.json",
        )
        # simulate_mc_paths returns CENTERED returns; add the fit window's
        # own mean back (horizon times, once per simulated day) before any
        # real-world comparison -- see this plan's Global Constraints.
        horizon_returns_real = (
            daily_paths.cpu().numpy().sum(axis=1) + horizon * train_mean
        )
        realized_return = float(future_returns.sum())

        row: dict[str, float] = {
            "cutoff": cutoff,
            "realized_return": realized_return,
            "crps": compute_crps(horizon_returns_real, realized_return),
            "wis": compute_wis(horizon_returns_real, realized_return),
        }
        for alpha in var_alphas:
            pct = int(round(alpha * 100))
            var_alpha = compute_var(horizon_returns_real, alpha)
            row[f"var_{pct}"] = var_alpha
            row[f"breached_{pct}"] = realized_return < -var_alpha
        rows.append(row)

    results_df = pd.DataFrame(rows)
    n = len(results_df)
    kupiec: dict[str, float] = {}
    for alpha in var_alphas:
        pct = int(round(alpha * 100))
        x = int(results_df[f"breached_{pct}"].sum())
        kupiec[f"kupiec_lr_{pct}"] = compute_kupiec_lr(n, x, 1.0 - alpha)
    return results_df, kupiec
```

- [ ] **Step 4: Run tests to verify they pass**

Run: `python -m pytest tests/evaluation/test_walk_forward.py -v`
Expected: PASS (budget a few minutes — this test does 2 real, tiny ADVI
fits + Monte Carlo simulation, even at n_steps=20 / n_paths=50 on synthetic
data)

- [ ] **Step 5: Commit**

```bash
git add src/raemf_mc/evaluation/walk_forward.py tests/evaluation/test_walk_forward.py
git commit -m "feat: add walk-forward OOS evaluation harness"
```

---

## Task 5: eval_smoke.yaml config profile

**Files:**
- Create: `configs/eval_smoke.yaml`

**Interfaces:**
- Produces: YAML keys consumed by Task 6 — `window_sessions` (fit window for the LAST cutoff, sizes the whole series pulled from real data), `n_cutoffs`, `horizon`, `ms_egarch_advi`, `ms_egarch_prior`, `mu_advi`, `mu_n_draws`, `mu_prior_scale`, `mu_min_effective_observations`, `mu_min_effective_fraction`, `mc_n_paths`, `seeds`, `var_alphas`.

- [ ] **Step 1: Create the config file**

```yaml
# configs/eval_smoke.yaml
# Profile debug nhanh cho walk-forward OOS: walk-forward nghia la fit lai
# MS-EGARCH + mu_k N LAN (mot lan moi cutoff), nen dung ADVI NHE HON HAN
# cac smoke config truoc (300 buoc) de tong thoi gian con hop ly. KHONG
# nham so lieu nghien cuu cuoi cung - chi verify pipeline dung.
window_sessions: 800
n_cutoffs: 6
horizon: 20
seeds: [0]
ms_egarch_advi:
  n_steps: 80
  learning_rate: 0.05
  warmup_steps: 10
  grad_clip_norm: 10.0
  elbo_ma_window: 10
  early_stop_patience: 40
  min_delta: 0.001
  retry_lr_factor: 0.5
  max_retries: 3
  n_mc_samples: 4
ms_egarch_prior:
  hyper_mean_scale: 1.0
  min_effective_observations: 30.0
  min_effective_fraction: 0.05
mu_advi:
  n_steps: 80
  learning_rate: 0.05
  warmup_steps: 10
  grad_clip_norm: 10.0
  elbo_ma_window: 10
  early_stop_patience: 40
  min_delta: 0.001
  retry_lr_factor: 0.5
  max_retries: 3
  n_mc_samples: 4
mu_n_draws: 20
mu_prior_scale: 0.01
mu_min_effective_observations: 30.0
mu_min_effective_fraction: 0.05
mc_n_paths: 200
var_alphas: [0.95, 0.99]
```

- [ ] **Step 2: Commit**

```bash
git add configs/eval_smoke.yaml
git commit -m "chore: add eval_smoke walk-forward config profile"
```

---

## Task 6: End-to-end integration test on real data

**Files:**
- Test: `tests/test_evaluation_integration_smoke.py`

**Interfaces:**
- Consumes everything from Tasks 1-5, plus `load_vnindex_ohlcv` (`raemf_mc.data.loader`), `compute_log_returns` (`raemf_mc.features.returns`), and sub-project 2's EBM pipeline pieces (`compute_causal_features`, `chronological_split`, `fit_regime_mu`/`fit_ms_egarch`/`compute_regime_labels`/`compute_posterior_volatility_features` reused the same way `tests/test_ebm_integration_smoke.py` already does, `fit_ebm_classifier`, `predict_scores`, `fit_temperature`, `apply_temperature_log_prob`).

This is the task most likely to surface a real integration mismatch, per
this plan's own established convention (sub-project 1 Task 18, sub-project
2 Task 8, sub-project 3 Task 6 all found and fixed real bugs here) — fixing
a genuine bug found while implementing this task is expected, normal work.

- [ ] **Step 1: Write the integration test**

```python
# tests/test_evaluation_integration_smoke.py
from pathlib import Path

import numpy as np
import torch
import yaml

from raemf_mc.bayesian.priors import HierarchicalPriorConfig
from raemf_mc.bayesian.torch_backend import AdviConfig
from raemf_mc.data.loader import load_vnindex_ohlcv
from raemf_mc.evaluation.classification_metrics import (
    compute_classification_report,
    compute_nll,
)
from raemf_mc.evaluation.walk_forward import run_walk_forward_evaluation
from raemf_mc.features.causal import compute_causal_features
from raemf_mc.features.returns import compute_log_returns
from raemf_mc.models.calibration import apply_temperature_log_prob, fit_temperature
from raemf_mc.models.ebm_classifier import fit_ebm_classifier, predict_scores
from raemf_mc.regime.ms_egarch import MSEGARCHParamLayout, default_recursion_init, fit_ms_egarch
from raemf_mc.regime.posterior_features import compute_posterior_volatility_features, compute_regime_labels
from raemf_mc.validation.chronological_split import chronological_split

EVAL_CONFIG_PATH = Path(__file__).resolve().parents[1] / "configs" / "eval_smoke.yaml"
EBM_CONFIG_PATH = Path(__file__).resolve().parents[1] / "configs" / "ebm_smoke.yaml"


def test_part_a_classification_metrics_on_real_ebm_pipeline(tmp_path):
    """Re-run sub-project 2's EBM pipeline on real data (same pattern as
    tests/test_ebm_integration_smoke.py) and evaluate it with Task 1's
    metrics -- this finishes the evaluation sub-project 2's own
    whole-branch review deferred."""
    with EBM_CONFIG_PATH.open(encoding="utf-8") as f:
        config = yaml.safe_load(f)

    ohlcv = load_vnindex_ohlcv()
    ohlcv_window = ohlcv.iloc[-config["window_sessions"] :]
    log_returns = compute_log_returns(ohlcv_window)
    causal_features = compute_causal_features(ohlcv_window)
    common_index = log_returns.index.intersection(causal_features.index)
    log_returns = log_returns.loc[common_index]
    causal_features = causal_features.loc[common_index]

    returns_train, returns_val, returns_test = chronological_split(
        log_returns.to_frame("ret"), train_frac=config["train_frac"], val_frac=config["val_frac"]
    )
    train_mean = returns_train["ret"].mean()
    centered_returns = log_returns - train_mean

    layout = MSEGARCHParamLayout()
    advi_config = AdviConfig(**config["advi"])
    prior_config = HierarchicalPriorConfig(**config["prior"])
    train_tensor = torch.tensor(
        centered_returns.loc[returns_train.index].to_numpy(), dtype=torch.float32
    )
    posterior = fit_ms_egarch(
        train_tensor, advi_config, prior_config, config["seeds"], torch.device("cpu"), layout,
        fallback_log_path=tmp_path / "fallbacks.json",
    )
    init_log_var, init_log_state_prob = default_recursion_init(layout)
    gen = torch.Generator().manual_seed(0)
    vol_features = compute_posterior_volatility_features(
        posterior, centered_returns, init_log_var, init_log_state_prob,
        layout=layout, n_draws=config["n_posterior_draws"], generator=gen,
    )
    labels = compute_regime_labels(
        posterior, centered_returns, init_log_var, init_log_state_prob,
        layout=layout, n_train=len(returns_train),
    )
    all_features = causal_features.join(vol_features, how="inner")

    X_train = all_features.loc[returns_train.index].to_numpy()
    X_test = all_features.loc[returns_test.index].to_numpy()
    y_train = labels.loc[returns_train.index].to_numpy()
    y_val = labels.loc[returns_val.index].to_numpy()
    y_test = labels.loc[returns_test.index].to_numpy()

    model = fit_ebm_classifier(X_train, y_train, random_state=0)
    class_to_idx = {c: i for i, c in enumerate(model.classes_)}

    X_val = all_features.loc[returns_val.index].to_numpy()
    scores_val = torch.tensor(predict_scores(model, X_val), dtype=torch.float32)
    y_val_idx = torch.tensor([class_to_idx[label] for label in y_val], dtype=torch.long)
    temperature = fit_temperature(scores_val, y_val_idx, n_steps=100)

    scores_test = torch.tensor(predict_scores(model, X_test), dtype=torch.float32)
    y_test_idx = torch.tensor([class_to_idx[label] for label in y_test], dtype=torch.long)
    y_pred = [model.classes_[i] for i in scores_test.argmax(dim=1).tolist()]

    report = compute_classification_report(list(y_test), y_pred, labels=list(model.classes_))
    print(f"Classification report (smoke-scale):\n{report}")
    assert 0.0 <= report["macro_f1"] <= 1.0
    for recall in report["recall_by_class"].values():
        assert 0.0 <= recall <= 1.0

    log_probs_calibrated = apply_temperature_log_prob(scores_test, temperature)
    nll = compute_nll(log_probs_calibrated, y_test_idx)
    assert np.isfinite(nll)
    assert nll >= 0.0


def test_part_b_walk_forward_oos_on_real_data(tmp_path):
    with EVAL_CONFIG_PATH.open(encoding="utf-8") as f:
        config = yaml.safe_load(f)

    ohlcv = load_vnindex_ohlcv()
    ohlcv_window = ohlcv.iloc[-config["window_sessions"] :]
    log_returns = compute_log_returns(ohlcv_window)

    n = len(log_returns)
    horizon = config["horizon"]
    n_cutoffs = config["n_cutoffs"]
    # Spread cutoffs evenly over the back half of the window, each leaving
    # at least `horizon` real sessions after it to score against, and at
    # least a reasonable fit window before it.
    min_cutoff = n // 2
    max_cutoff = n - horizon - 1
    cutoffs = sorted(
        {min_cutoff + int(i * (max_cutoff - min_cutoff) / (n_cutoffs - 1)) for i in range(n_cutoffs)}
    )

    layout = MSEGARCHParamLayout()
    ms_egarch_advi = AdviConfig(**config["ms_egarch_advi"])
    ms_egarch_prior = HierarchicalPriorConfig(**config["ms_egarch_prior"])
    mu_advi = AdviConfig(**config["mu_advi"])
    fallback_dir = tmp_path / "fallbacks"
    fallback_dir.mkdir()

    results_df, kupiec = run_walk_forward_evaluation(
        log_returns, cutoffs=cutoffs, horizon=horizon,
        ms_egarch_advi=ms_egarch_advi, ms_egarch_prior=ms_egarch_prior,
        mu_advi=mu_advi, mu_n_draws=config["mu_n_draws"],
        mu_prior_scale=config["mu_prior_scale"],
        mu_min_effective_observations=config["mu_min_effective_observations"],
        mu_min_effective_fraction=config["mu_min_effective_fraction"],
        mc_n_paths=config["mc_n_paths"], layout=layout, seeds=config["seeds"],
        device=torch.device("cpu"), fallback_log_dir=fallback_dir,
        var_alphas=tuple(config["var_alphas"]),
        generator=torch.Generator().manual_seed(0),
    )
    print(f"Walk-forward OOS results (smoke-scale, {len(cutoffs)} cutoffs):\n{results_df}")
    print(f"Kupiec LR statistics (smoke-scale, low power at this n): {kupiec}")

    assert len(results_df) == len(cutoffs)
    assert results_df["crps"].apply(np.isfinite).all()
    assert results_df["wis"].apply(np.isfinite).all()
    assert (results_df["crps"] >= 0.0).all()
    assert (results_df["wis"] >= 0.0).all()
    for v in kupiec.values():
        assert np.isfinite(v)
        assert v >= 0.0
```

- [ ] **Step 2: Run the test and debug any real integration failure**

Run: `python -m pytest tests/test_evaluation_integration_smoke.py -v -s`
Expected: either PASS directly, or a real integration bug to diagnose and
fix at its root cause — this is allowed to fail for a substantive reason,
per the same discipline as every earlier sub-project's equivalent task.
Budget significant real time: Part A repeats one real MS-EGARCH fit (similar
cost to sub-project 2's own integration test, ~15-25 min); Part B does
`n_cutoffs` (6 by default) real MS-EGARCH + mu_k fits at the lighter
`eval_smoke.yaml` settings — likely 30-60+ minutes total. If a single run
doesn't complete within your working session, report `DONE_WITH_CONCERNS`
with what you've verified so far (e.g. a reduced-scale sanity check with
fewer cutoffs/steps) rather than looping or vaguely waiting — the
controller will run the real full-scale version directly, matching the
process already established for every earlier sub-project's equivalent task.

- [ ] **Step 3: Iterate until it passes**

Run: `python -m pytest tests/test_evaluation_integration_smoke.py -v -s`
Expected: PASS

- [ ] **Step 4: Commit**

```bash
git add tests/test_evaluation_integration_smoke.py
git commit -m "test: add end-to-end evaluation pipeline smoke test on real VN-Index data"
```

---

## Task 7: Model card

**Files:**
- Create: `docs/model_card.md`

- [ ] **Step 1: Write the model card**

Write `docs/model_card.md` covering, at minimum, these sections (pull
concrete facts from the existing design docs rather than inventing new
claims):

- **Intended use**: research / decision-support for VN-Index regime and
  risk analysis. Explicitly NOT a production trading system, NOT validated
  for live/automated use.
- **Data**: `data/raw/VNINDEX_Daily.csv`, real date range and session count
  (pull the actual numbers from `docs/data_ingestion_notes.md`), daily
  frequency, cleaning steps applied (reference
  `docs/data_ingestion_notes.md`).
- **Architecture**: one paragraph per layer — MS-EGARCH (4-state regime
  volatility model, ADVI-fit), EBM classifier (point-estimate nowcasting,
  temperature-calibrated), scenario-return + Monte Carlo (per-regime drift
  + forward simulation), risk metrics (VaR/CVaR/max-drawdown) — each
  linking to its design spec under `docs/superpowers/specs/`.
- **Evaluation results**: report the real smoke-scale numbers produced by
  Task 6's integration test (macro F1, per-class recall, NLL from Part A;
  CRPS/WIS/Kupiec LR from Part B) — copy the actual printed output from a
  real test run, not invented numbers. Prefix this whole section with an
  explicit, unmissable caveat that these are smoke-scale pipeline-
  correctness numbers, not research-grade results.
- **Known limitations**: consolidate the limitations already recorded
  across all 4 sub-projects' specs/ledgers — no walk-forward re-fit at
  production scale, single-fit-then-simulate (not fully online), some ADVI
  posterior draws fall outside the model's own stationarity region under
  fast configs, regime labels can be degenerate under under-converged
  fits, CPU-only verified (GPU path implemented but unexercised), no model
  persistence/serialization, no CLI/pipeline orchestration, daily
  frequency only (no intraday).

- [ ] **Step 2: Commit**

```bash
git add docs/model_card.md
git commit -m "docs: add model card summarizing architecture, evaluation, and limitations"
```

---

## Task 8: Full suite verification

**Files:** none (verification only)

- [ ] **Step 1: Run the full test suite**

Run: `python -m pytest -q` (budget significant time — this now includes
every earlier sub-project's real integration test plus this one's two,
one of which runs several real ADVI fits in a loop)
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
