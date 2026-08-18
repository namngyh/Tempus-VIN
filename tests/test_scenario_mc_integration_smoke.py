# tests/test_scenario_mc_integration_smoke.py
from pathlib import Path

import numpy as np
import torch
import yaml

from raemf_mc.bayesian.priors import HierarchicalPriorConfig
from raemf_mc.bayesian.torch_backend import AdviConfig
from raemf_mc.data.loader import load_vnindex_ohlcv
from raemf_mc.features.returns import compute_log_returns
from raemf_mc.regime.ms_egarch import (
    MSEGARCHParamLayout,
    default_recursion_init,
    fit_ms_egarch,
)
from raemf_mc.risk.metrics import summarize_risk
from raemf_mc.scenario.mu_fit import fit_regime_mu
from raemf_mc.scenario.simulate import simulate_mc_paths

CONFIG_PATH = Path(__file__).resolve().parents[1] / "configs" / "mc_smoke.yaml"


def test_scenario_mc_pipeline_end_to_end_on_real_data(tmp_path):
    with CONFIG_PATH.open(encoding="utf-8") as f:
        config = yaml.safe_load(f)

    ohlcv = load_vnindex_ohlcv()
    ohlcv_window = ohlcv.iloc[-config["window_sessions"] :]
    log_returns = compute_log_returns(ohlcv_window)
    # No train/val/test split for this sub-project (forward simulation from
    # "today," not nowcasting or held-out evaluation) -- fit on the whole
    # window, centered by its own mean.
    centered_returns = log_returns - log_returns.mean()

    layout = MSEGARCHParamLayout()
    device = torch.device("cpu")
    returns_tensor = torch.tensor(centered_returns.to_numpy(), dtype=torch.float32)

    ms_egarch_advi = AdviConfig(**config["ms_egarch_advi"])
    ms_egarch_prior = HierarchicalPriorConfig(**config["ms_egarch_prior"])
    ms_egarch_posterior = fit_ms_egarch(
        returns_tensor, ms_egarch_advi, ms_egarch_prior, config["seeds"], device, layout,
        fallback_log_path=tmp_path / "ms_egarch_fallbacks.json",
    )
    for r in ms_egarch_posterior.seed_results:
        assert torch.isfinite(r.mu).all()

    init_log_var, init_log_state_prob = default_recursion_init(layout, device=device)

    mu_advi = AdviConfig(**config["mu_advi"])
    mu_gen = torch.Generator().manual_seed(0)
    mu_posterior = fit_regime_mu(
        returns_tensor, ms_egarch_posterior, mu_advi, config["seeds"], device,
        init_log_var, init_log_state_prob, layout=layout,
        n_draws=config["mu_n_draws"], mu_prior_scale=config["mu_prior_scale"],
        min_effective_observations=config["mu_min_effective_observations"],
        generator=mu_gen, fallback_log_path=tmp_path / "mu_fallbacks.json",
    )
    for r in mu_posterior.seed_results:
        assert torch.isfinite(r.mu).all()

    mc_gen = torch.Generator().manual_seed(1)
    daily_paths = simulate_mc_paths(
        ms_egarch_posterior, mu_posterior, returns_tensor, init_log_var, init_log_state_prob,
        n_paths=config["mc_n_paths"], horizon=config["mc_horizon"], layout=layout,
        device=device, generator=mc_gen,
    )
    assert daily_paths.shape == (config["mc_n_paths"], config["mc_horizon"])
    assert torch.isfinite(daily_paths).all()

    risk_table = summarize_risk(
        daily_paths.cpu().numpy(),
        horizons=tuple(config["mc_report_horizons"]),
        alphas=tuple(config["mc_alphas"]),
    )
    print(f"Risk summary (smoke-scale, {config['mc_n_paths']} paths):\n{risk_table}")

    for h in config["mc_report_horizons"]:
        row = risk_table.loc[h]
        assert row["VaR_99"] >= row["VaR_95"]
        assert row["CVaR_95"] >= row["VaR_95"]
        assert row["CVaR_99"] >= row["VaR_99"]
        assert 0.0 <= row["mean_max_drawdown"] <= 1.0
        assert np.isfinite(row["mean_max_drawdown"])
