# tests/test_integration_smoke.py
from pathlib import Path

import torch
import yaml

from raemf_mc.data.loader import load_vnindex_ohlcv
from raemf_mc.features.returns import compute_log_returns
from raemf_mc.regime.ms_egarch import (
    MSEGARCHParamLayout,
    default_recursion_init,
    fit_ms_egarch,
    sample_ms_egarch_draw,
    transition_matrix,
)
from raemf_mc.regime.state_alignment import STATE_NAMES, align_states, apply_alignment
from raemf_mc.bayesian.diagnostics import seed_stability_metrics, summarize_elbo_trace
from raemf_mc.bayesian.torch_backend import AdviConfig
from raemf_mc.bayesian.priors import HierarchicalPriorConfig

CONFIG_PATH = Path(__file__).resolve().parents[1] / "configs" / "cpu_smoke.yaml"


def test_full_pipeline_smoke_on_real_data(tmp_path):
    with CONFIG_PATH.open(encoding="utf-8") as f:
        config = yaml.safe_load(f)

    ohlcv = load_vnindex_ohlcv()
    log_returns = compute_log_returns(ohlcv)
    window = log_returns.iloc[-config["window_sessions"] :]
    returns = torch.tensor(window.to_numpy(), dtype=torch.float32)
    returns = returns - returns.mean()

    advi_config = AdviConfig(**config["advi"])
    prior_config = HierarchicalPriorConfig(**config["prior"])
    layout = MSEGARCHParamLayout()

    init_mu = torch.zeros(layout.total)
    posterior = fit_ms_egarch(
        returns, advi_config, prior_config, seeds=config["seeds"],
        device=torch.device("cpu"), layout=layout,
        fallback_log_path=tmp_path / "fallbacks.json",
    )
    assert len(posterior.seed_results) == len(config["seeds"])
    for r in posterior.seed_results:
        assert torch.isfinite(r.mu).all()
        assert torch.isfinite(r.log_sigma).all()

    # --- the fit must actually have done something, not merely not crashed ---
    # ELBO improved from initialization for at least one seed.
    assert any(r.elbo_trace[-1] > r.elbo_trace[0] for r in posterior.seed_results)
    # mu moved away from its init, i.e. the optimizer had a real effect.
    assert any(
        not torch.allclose(r.mu, init_mu, atol=1e-3) for r in posterior.seed_results
    )
    # Fallback visibility is wired up and reports a real number.
    assert 0.0 <= posterior.fallback_fraction <= 1.0

    # Diagnostics are unit-tested in isolation but were never exercised on a
    # real fit's output; make sure they compose with it.
    summary = summarize_elbo_trace(posterior.seed_results[0].elbo_trace)
    assert summary["n_steps"] == len(posterior.seed_results[0].elbo_trace)
    assert summary["best_elbo"] >= summary["final_elbo"]
    stability = seed_stability_metrics(posterior.seed_results)
    assert stability["final_elbo_std"] >= 0.0
    assert stability["max_pairwise_mu_diff"] >= 0.0

    # fixed generator so this slow test is reproducible rather than flaky
    draw = sample_ms_egarch_draw(
        posterior, layout, generator=torch.Generator().manual_seed(0)
    )
    assert draw.omega.shape == (4,)

    # The regime process must not be degenerate: a uniform transition matrix
    # (every diagonal at 1/n) would mean the "regimes" carry no persistence.
    trans = transition_matrix(draw.transition_logits)
    assert torch.allclose(trans.sum(dim=1), torch.ones(4), atol=1e-5)
    diagonal = torch.diagonal(trans)
    assert float(diagonal.max()) > 1.0 / layout.n_states

    from raemf_mc.regime.ms_egarch import run_ms_egarch_recursion

    init_log_var, init_log_state_prob = default_recursion_init(layout)
    result = run_ms_egarch_recursion(returns, draw, init_log_var, init_log_state_prob)
    assert torch.isfinite(result["log_filtered_prob"]).all()
    assert 0.0 <= result["clamp_saturation_fraction"] <= 1.0

    permutation = align_states(returns, result["log_filtered_prob"])
    aligned = apply_alignment(result["log_filtered_prob"], permutation)
    assert aligned.shape == result["log_filtered_prob"].shape
    assert sorted(permutation) == list(range(layout.n_states))
    assert len(STATE_NAMES) == 4
