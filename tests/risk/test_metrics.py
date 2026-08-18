import numpy as np

from raemf_mc.risk.metrics import compute_cvar, compute_max_drawdown, compute_var, summarize_risk


def test_compute_var_and_cvar_hand_computed():
    # 101 evenly spaced values from -0.50 to +0.50 (step 0.01) so that
    # quantile(0.05) at 101 points lands EXACTLY on index 5 (no
    # interpolation): position = (101-1)*0.05 = 5.0 exactly.
    returns = np.linspace(-0.50, 0.50, 101)
    var_95 = compute_var(returns, 0.95)
    assert abs(var_95 - 0.45) < 1e-9  # -quantile(0.05) = -(-0.45) = 0.45

    cvar_95 = compute_cvar(returns, 0.95)
    # tail = indices 0..5 = [-0.50,-0.49,-0.48,-0.47,-0.46,-0.45], mean=-0.475
    assert abs(cvar_95 - 0.475) < 1e-9

    assert compute_cvar(returns, 0.95) >= compute_var(returns, 0.95)

    var_99 = compute_var(returns, 0.99)
    assert var_99 >= var_95  # deeper tail = at least as much loss


def test_compute_max_drawdown_hand_computed():
    # Single path: returns that go up, then down enough to create a known
    # drawdown, then flat.
    daily = np.array([[0.10, 0.10, -0.30, 0.0, 0.05]])
    # price: 1 -> 1.10517 -> 1.22140 -> 0.90484 -> 0.90484 -> 0.95123
    # running max hits 1.22140 at t=2, price drops to 0.90484 at t=3
    # drawdown = (1.22140 - 0.90484) / 1.22140 ~= 0.2591
    dd = compute_max_drawdown(daily)
    assert dd.shape == (1,)
    assert abs(dd[0] - 0.2591) < 1e-3


def test_summarize_risk_shape_and_invariants():
    rng = np.random.default_rng(0)
    daily_paths = rng.normal(loc=0.0, scale=0.01, size=(2000, 20))
    table = summarize_risk(daily_paths, horizons=(1, 20), alphas=(0.95, 0.99))
    assert list(table.index) == [1, 20]
    for h in (1, 20):
        row = table.loc[h]
        assert row["VaR_99"] >= row["VaR_95"]
        assert row["CVaR_95"] >= row["VaR_95"]
        assert row["CVaR_99"] >= row["VaR_99"]
        assert 0.0 <= row["mean_max_drawdown"] <= 1.0
        assert 0.0 <= row["median_max_drawdown"] <= 1.0
        assert 0.0 <= row["p95_max_drawdown"] <= 1.0
