from __future__ import annotations

from dataclasses import dataclass

import torch


@dataclass(frozen=True)
class HierarchicalPriorConfig:
    """Prior specification for the MS-EGARCH parameter block.

    Two layers. The hierarchical layer (`hyper_mean_scale` + the shrinkage
    threshold) controls how tightly the four per-state values are pulled
    toward their shared hyper-mean. The `*_hyper_*` fields are the level
    prior on that hyper-mean itself: without them the hierarchical term is
    translation-invariant (it penalizes only the SPREAD of the four states,
    so adding a constant to all four leaves it untouched) and nothing at all
    constrains where the parameters sit — in particular nothing stops the
    EGARCH persistence `beta` from exceeding 1, which is explosive.
    """

    hyper_mean_scale: float = 1.0

    # Absolute floor on the effective-observation threshold. On its own this
    # never binds: with ~500 (smoke) to ~6264 (full) sessions across 4 states,
    # every regime clears 30 effective observations by a wide margin, so the
    # shrinkage weight pinned at exactly 1.0 and the hierarchical mechanism —
    # the spec's headline feature for the sparse Stress regime — never
    # activated. The fraction below is what actually makes it scale-aware.
    min_effective_observations: float = 30.0
    # A regime holding less than this fraction of total occupancy counts as
    # under-identified and gets real shrinkage, at any sample size.
    min_effective_fraction: float = 0.05

    # Level priors on each parameter family's hyper-mean.
    # omega (log-variance intercept): log-variance of daily log returns sits
    # around log(1e-4) ~ -9, but omega is the intercept of a recursion whose
    # persistence carries most of the level, so keep this genuinely vague.
    omega_hyper_mean_loc: float = 0.0
    omega_hyper_mean_scale: float = 2.0
    # alpha (ARCH/size effect) and gamma (leverage): both are small in
    # magnitude for daily equity indices and are sign-free a priori.
    alpha_hyper_mean_loc: float = 0.0
    alpha_hyper_mean_scale: float = 0.5
    gamma_hyper_mean_loc: float = 0.0
    gamma_hyper_mean_scale: float = 0.5
    # beta (persistence): the one parameter with a hard qualitative boundary.
    # |beta| >= 1 makes the log-variance recursion non-stationary/explosive.
    # Daily equity volatility persistence is empirically ~0.85-0.99, so
    # center at 0.9 with a scale that keeps that whole range inside ~0.7 sd
    # while putting 1.2 at 2 sd. Deliberately weak in absolute terms — with
    # hundreds to thousands of observations the likelihood still dominates;
    # the point is that there is now SOME force against the explosive region,
    # where previously there was literally none.
    beta_hyper_mean_loc: float = 0.9
    beta_hyper_mean_scale: float = 0.15

    # nu_raw, where nu = 2.05 + softplus(nu_raw). The transform is why the
    # obvious Normal(0, 1) is a trap: it puts nu almost entirely below ~4.2,
    # effectively mandating near-infinite kurtosis. Realistic Student-t df for
    # a daily equity index is ~5-15, i.e. nu_raw ~ 2.9-13.0, so center at 5
    # and widen to cover that whole band within ~2 sd.
    nu_raw_loc: float = 5.0
    nu_raw_scale: float = 4.0

    # Transition logits. Rows are softmax([0, logits]) with column 0 as the
    # fixed reference category, so a zero logit vector means a UNIFORM row —
    # i.e. a regime that survives one day with probability 0.25. Centering
    # the prior there fights the entire premise of a regime model: a realistic
    # sticky regime (p_stay ~ 0.95) needs a self-transition logit of ~4, which
    # under the old Normal(0, 1) was 4 prior standard deviations out. Center
    # on a moderately sticky row instead and widen the scale so both sticky
    # and switch-happy rows stay plausible.
    transition_stay_prob: float = 0.85
    transition_logit_scale: float = 2.0

    def shrinkage_threshold(self, n_observations: int) -> float:
        """Effective-observation count below which a state gets shrunk.

        Scales with the window so the mechanism behaves the same on a
        500-session smoke run and a 6264-session full run, with the absolute
        floor guarding tiny windows.
        """
        return max(
            self.min_effective_observations,
            self.min_effective_fraction * float(n_observations),
        )


def state_shrinkage_weight(
    effective_obs: torch.Tensor, min_effective_observations: float
) -> torch.Tensor:
    """Per-state weight in (0, 1]: states with effective_obs far below
    min_effective_observations get a smaller weight (tighter shrinkage
    toward the hyper-mean); states with ample data approach weight 1."""
    return torch.clamp(effective_obs / min_effective_observations, min=0.05, max=1.0)


def hierarchical_normal_log_prob(
    state_params: torch.Tensor,
    hyper_mean: torch.Tensor,
    base_scale: torch.Tensor,
    shrinkage_weight: torch.Tensor,
) -> torch.Tensor:
    """log p(state_params | hyper_mean) under
    Normal(hyper_mean, base_scale * shrinkage_weight). A smaller
    shrinkage_weight tightens the effective prior std, pulling the state
    parameter harder toward the shared hyper-mean."""
    scale = base_scale * shrinkage_weight
    dist = torch.distributions.Normal(hyper_mean, scale)
    # sum(dim=-1) chứ không phải sum(): với đầu vào 1-D `(n_states,)` hai
    # cách cho ra cùng một vô hướng, nhưng với đầu vào có chiều batch dẫn
    # đầu `(B, n_states)` thì sum() sẽ gộp luôn cả batch thành một số —
    # làm hỏng tính độc lập giữa các hàng batch mà ADVI batch dựa vào.
    return dist.log_prob(state_params).sum(dim=-1)
