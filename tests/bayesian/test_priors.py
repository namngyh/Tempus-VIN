import torch
from raemf_mc.bayesian.priors import (
    HierarchicalPriorConfig,
    state_shrinkage_weight,
    hierarchical_normal_log_prob,
)


def test_state_shrinkage_weight_is_lower_for_sparse_states():
    effective_obs = torch.tensor([5.0, 30.0, 100.0])
    weight = state_shrinkage_weight(effective_obs, min_effective_observations=30.0)
    assert weight[0] < weight[1] <= weight[2]
    assert torch.all(weight > 0) and torch.all(weight <= 1.0)


def test_hierarchical_normal_log_prob_penalizes_deviation_from_hyper_mean():
    hyper_mean = torch.tensor(0.0)
    base_scale = torch.tensor(1.0)
    weight = torch.tensor(1.0)
    close = hierarchical_normal_log_prob(torch.tensor([0.1]), hyper_mean, base_scale, weight)
    far = hierarchical_normal_log_prob(torch.tensor([5.0]), hyper_mean, base_scale, weight)
    assert close > far


def test_hierarchical_normal_log_prob_tighter_for_low_shrinkage_weight():
    hyper_mean = torch.tensor(0.0)
    base_scale = torch.tensor(1.0)

    # Far from the mean: a tight (low-weight) prior assigns much lower
    # density than a loose (high-weight) prior — this direction holds
    # under both a correct and an inverted formula, so it alone doesn't
    # prove the shrinkage direction.
    far_deviation = torch.tensor([2.0])
    low_weight_far = hierarchical_normal_log_prob(far_deviation, hyper_mean, base_scale, torch.tensor(0.1))
    high_weight_far = hierarchical_normal_log_prob(far_deviation, hyper_mean, base_scale, torch.tensor(1.0))
    assert low_weight_far < high_weight_far

    # Near the mean: a tight (low-weight) prior concentrates density near
    # the mean, so it must assign HIGHER density here than a loose
    # (high-weight) prior. This is the point that actually distinguishes
    # a correct tightening formula from an inverted (widening) one.
    near_deviation = torch.tensor([0.01])
    low_weight_near = hierarchical_normal_log_prob(near_deviation, hyper_mean, base_scale, torch.tensor(0.1))
    high_weight_near = hierarchical_normal_log_prob(near_deviation, hyper_mean, base_scale, torch.tensor(1.0))
    assert low_weight_near > high_weight_near


def test_hierarchical_prior_config_defaults():
    config = HierarchicalPriorConfig()
    assert config.hyper_mean_scale == 1.0
    assert config.min_effective_observations == 30.0
