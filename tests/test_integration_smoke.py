# tests/test_integration_smoke.py
from pathlib import Path

import torch
import yaml

from raemf_mc.data.loader import load_vnindex_ohlcv
from raemf_mc.features.returns import compute_log_returns
from raemf_mc.regime.ms_egarch import MSEGARCHParamLayout, fit_ms_egarch, sample_ms_egarch_draw
from raemf_mc.regime.state_alignment import STATE_NAMES, align_states, apply_alignment
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

    posterior = fit_ms_egarch(
        returns, advi_config, prior_config, seeds=config["seeds"],
        device=torch.device("cpu"), layout=layout,
        fallback_log_path=tmp_path / "fallbacks.json",
    )
    assert len(posterior.seed_results) == len(config["seeds"])
    for r in posterior.seed_results:
        assert torch.isfinite(r.mu).all()
        assert torch.isfinite(r.log_sigma).all()

    draw = sample_ms_egarch_draw(posterior, layout)
    assert draw.omega.shape == (4,)

    from raemf_mc.regime.ms_egarch import run_ms_egarch_recursion

    n = layout.n_states
    init_log_var = torch.zeros(n)
    init_log_state_prob = torch.log(torch.full((n,), 1.0 / n))
    result = run_ms_egarch_recursion(returns, draw, init_log_var, init_log_state_prob)
    assert torch.isfinite(result["log_filtered_prob"]).all()

    permutation = align_states(returns, result["log_filtered_prob"])
    aligned = apply_alignment(result["log_filtered_prob"], permutation)
    assert aligned.shape == result["log_filtered_prob"].shape
    assert len(STATE_NAMES) == 4
