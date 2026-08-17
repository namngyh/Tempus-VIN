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
    deviation = torch.tensor([2.0])
    low_weight_logprob = hierarchical_normal_log_prob(deviation, hyper_mean, base_scale, torch.tensor(0.1))
    high_weight_logprob = hierarchical_normal_log_prob(deviation, hyper_mean, base_scale, torch.tensor(1.0))
    # smaller shrinkage_weight -> tighter effective prior std -> lower
    # density further from the hyper-mean at a fixed deviation
    assert low_weight_logprob < high_weight_logprob


def test_hierarchical_prior_config_defaults():
    config = HierarchicalPriorConfig()
    assert config.hyper_mean_scale == 1.0
    assert config.min_effective_observations == 30.0
