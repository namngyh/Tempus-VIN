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
