import math
import torch
from scipy import stats
from raemf_mc.regime.ms_egarch import (
    MSEGARCHParamLayout,
    unpack_params,
    transition_matrix,
    nu_from_raw,
    expected_abs_standardized_t,
    student_t_log_pdf_with_variance,
)


def test_param_layout_totals_match_spec_budget():
    layout = MSEGARCHParamLayout()
    assert layout.n_states == 4
    assert layout.n_egarch_params == 16
    assert layout.n_transition_params == 12
    assert layout.n_nu_params == 1
    assert layout.total == 29  # 28 core + 1 global nu


def test_unpack_params_round_trip_shapes():
    layout = MSEGARCHParamLayout()
    theta = torch.arange(layout.total, dtype=torch.float32)
    params = unpack_params(theta, layout)
    assert params.omega.shape == (4,)
    assert params.alpha.shape == (4,)
    assert params.beta.shape == (4,)
    assert params.gamma.shape == (4,)
    assert params.transition_logits.shape == (4, 3)
    assert params.nu_raw.shape == ()


def test_transition_matrix_rows_sum_to_one():
    logits = torch.randn(4, 3)
    trans = transition_matrix(logits)
    assert trans.shape == (4, 4)
    row_sums = trans.sum(dim=1)
    assert torch.allclose(row_sums, torch.ones(4), atol=1e-6)
    assert torch.all(trans >= 0)


def test_transition_matrix_zero_logits_gives_uniform_rows():
    logits = torch.zeros(4, 3)
    trans = transition_matrix(logits)
    assert torch.allclose(trans, torch.full((4, 4), 0.25), atol=1e-6)


def test_nu_from_raw_always_above_min():
    raw = torch.tensor([-100.0, 0.0, 100.0])
    nu = nu_from_raw(raw, min_nu=2.05)
    assert torch.all(nu > 2.05)


def test_expected_abs_standardized_t_matches_scipy_reference():
    nu_value = 8.0
    nu = torch.tensor(nu_value)
    computed = expected_abs_standardized_t(nu).item()
    # E|T| for raw Student-t(nu), then rescale to unit variance
    scale_factor = math.sqrt((nu_value - 2) / nu_value)
    raw_t_mean_abs = 2.0 * math.sqrt(nu_value) * math.exp(
        math.lgamma((nu_value + 1) / 2) - math.lgamma(nu_value / 2)
    ) / ((nu_value - 1) * math.sqrt(math.pi))
    expected = raw_t_mean_abs * scale_factor
    assert math.isclose(computed, expected, rel_tol=1e-4)


def test_student_t_log_pdf_with_variance_matches_scipy_for_unit_variance():
    nu_value = 6.0
    variance = torch.tensor(2.5)
    loc = torch.tensor(1.0)
    x = torch.tensor(1.8)
    nu = torch.tensor(nu_value)
    computed = student_t_log_pdf_with_variance(x, loc, variance, nu).item()

    scale_param = math.sqrt(float(variance) * (nu_value - 2) / nu_value)
    expected = stats.t.logpdf(x.item(), df=nu_value, loc=loc.item(), scale=scale_param)
    assert math.isclose(computed, expected, rel_tol=1e-4)
