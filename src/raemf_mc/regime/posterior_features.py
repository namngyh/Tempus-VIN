from __future__ import annotations

import pandas as pd
import torch

from raemf_mc.bayesian.torch_backend import PooledPosterior
from raemf_mc.regime.ms_egarch import (
    MSEGARCHParamLayout,
    run_ms_egarch_recursion,
    sample_ms_egarch_draw,
    unpack_params,
)
from raemf_mc.regime.state_alignment import STATE_NAMES, align_states, apply_alignment


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
    dtype: torch.dtype = torch.float32,
    device: torch.device | None = None,
) -> pd.DataFrame:
    """Posterior-mean and posterior-sd of the collapsed sigma_bar_t at every
    session, estimated from n_draws independent posterior draws.

    sigma_bar_t = exp(0.5 * log_var_bar[t]) is the level-space-collapsed
    expected volatility at t under the filtered state distribution — the
    natural "posterior sigma_t" quantity, already computed by
    run_ms_egarch_recursion via Gray's collapsing (Sub-project 1).

    dtype/device follow the same convention as default_recursion_init: they
    govern only the tensors constructed HERE from the pandas input. `params`
    (from sample_ms_egarch_draw) and init_log_var/init_log_state_prob are the
    caller's responsibility to already have on the matching device, exactly
    as run_ms_egarch_recursion assumes elsewhere. Defaults preserve the
    previous unconditional CPU/float32 behavior.
    """
    returns_tensor = torch.tensor(returns.to_numpy(), dtype=dtype, device=device)
    sigma_draws = torch.zeros(n_draws, len(returns), dtype=dtype, device=device)
    for i in range(n_draws):
        params = sample_ms_egarch_draw(posterior, layout, generator=generator)
        result = run_ms_egarch_recursion(
            returns_tensor, params, init_log_var, init_log_state_prob
        )
        sigma_draws[i] = torch.exp(0.5 * result["log_var_bar"])
    posterior_mean_sigma = sigma_draws.mean(dim=0).cpu()
    posterior_sd_sigma = sigma_draws.std(dim=0, unbiased=False).cpu()
    return pd.DataFrame(
        {
            "posterior_mean_sigma": posterior_mean_sigma.numpy(),
            "posterior_sd_sigma": posterior_sd_sigma.numpy(),
        },
        index=returns.index,
    )


def compute_regime_labels(
    posterior: PooledPosterior,
    returns: pd.Series,
    init_log_var: torch.Tensor,
    init_log_state_prob: torch.Tensor,
    n_train: int,
    layout: MSEGARCHParamLayout = MSEGARCHParamLayout(),
    dtype: torch.dtype = torch.float32,
    device: torch.device | None = None,
) -> pd.Series:
    """Deterministic target label per session: argmax of the train-aligned
    filtered regime probability, using the canonical (posterior-mean)
    parameter point estimate — a single reproducible ground-truth proxy
    sequence, not an ensemble (unlike the volatility features above, which
    deliberately use random draws to capture posterior spread).

    The state->name permutation (align_states) is fit using ONLY the first
    n_train rows, then applied to the full series — never let val/test rows
    influence which raw state is called "Bull" vs "Bear", or the training
    targets themselves become a function of held-out data. apply_alignment is
    a pure column reindex, so applying a train-fit permutation to the whole
    series introduces no leakage; running align_states over the whole series
    does, and demonstrably changed the permutation on real data.
    """
    theta = canonical_theta(posterior)
    params = unpack_params(theta, layout)
    returns_tensor = torch.tensor(returns.to_numpy(), dtype=dtype, device=device)
    result = run_ms_egarch_recursion(
        returns_tensor, params, init_log_var, init_log_state_prob
    )
    permutation = align_states(
        returns_tensor[:n_train], result["log_filtered_prob"][:n_train]
    )
    aligned = apply_alignment(result["log_filtered_prob"], permutation)
    label_idx = aligned.argmax(dim=1).tolist()
    labels = [STATE_NAMES[i] for i in label_idx]
    return pd.Series(labels, index=returns.index, name="regime_label")
