import torch
from raemf_mc.regime.state_alignment import STATE_NAMES, align_states, apply_alignment


def test_align_states_orders_by_mean_return_and_volatility():
    # 4 raw states with clearly separated economic character:
    # state 0: high positive mean, low vol -> Bull
    # state 1: near-zero mean, low vol -> Sideway
    # state 2: negative mean, moderate vol -> Bear
    # state 3: very negative mean, high vol -> Stress
    torch.manual_seed(0)
    T = 400
    raw_state_returns = {
        0: torch.randn(T) * 0.003 + 0.0015,
        1: torch.randn(T) * 0.004 + 0.0001,
        2: torch.randn(T) * 0.01 - 0.003,
        3: torch.randn(T) * 0.03 - 0.01,
    }
    # build a returns series and a hard filtered-prob assignment that
    # spends ~T/4 timesteps in each raw state, in a fixed known order
    returns = torch.cat([raw_state_returns[k] for k in range(4)])
    log_filtered_prob = torch.full((4 * T, 4), -30.0)
    for k in range(4):
        log_filtered_prob[k * T : (k + 1) * T, k] = 0.0

    permutation = align_states(returns, log_filtered_prob)
    assert sorted(permutation) == [0, 1, 2, 3]
    assert permutation[STATE_NAMES.index("Bull")] == 0
    assert permutation[STATE_NAMES.index("Sideway")] == 1
    assert permutation[STATE_NAMES.index("Bear")] == 2
    assert permutation[STATE_NAMES.index("Stress")] == 3


def test_apply_alignment_reorders_columns():
    log_filtered_prob = torch.tensor([[0.1, 0.2, 0.3, 0.4]]).log()
    permutation = [2, 0, 3, 1]
    reordered = apply_alignment(log_filtered_prob, permutation)
    expected = torch.tensor([[0.3, 0.1, 0.4, 0.2]]).log()
    assert torch.allclose(reordered, expected, atol=1e-6)


def test_align_states_handles_a_state_with_no_probability_mass():
    """A state that never activates has ~zero total mass; normalizing by it
    must not produce NaN and silently scramble the ordering."""
    torch.manual_seed(2)
    T = 300
    returns = torch.cat([torch.randn(T) * 0.005 + 0.002, torch.randn(T) * 0.02 - 0.005])
    log_filtered_prob = torch.full((2 * T, 4), -80.0)
    log_filtered_prob[:T, 0] = 0.0
    log_filtered_prob[T:, 1] = 0.0
    # states 2 and 3 keep essentially zero mass everywhere

    permutation = align_states(returns, log_filtered_prob)
    assert sorted(permutation) == [0, 1, 2, 3]
    assert permutation.index(0) < permutation.index(1)
