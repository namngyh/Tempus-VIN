import torch
from raemf_mc.models.calibration import apply_temperature, fit_temperature


def test_apply_temperature_returns_valid_probabilities():
    scores = torch.tensor([[2.0, 0.5, -1.0, 0.0], [0.1, 0.2, 0.3, 0.4]])
    probs = apply_temperature(scores, temperature=1.5)
    assert probs.shape == (2, 4)
    assert torch.allclose(probs.sum(dim=1), torch.ones(2), atol=1e-6)
    assert (probs > 0).all()


def test_apply_temperature_one_is_a_no_op_softmax():
    scores = torch.tensor([[2.0, 0.5, -1.0, 0.0]])
    probs = apply_temperature(scores, temperature=1.0)
    expected = torch.softmax(scores, dim=1)
    assert torch.allclose(probs, expected, atol=1e-6)


def test_fit_temperature_reduces_nll_on_overconfident_scores():
    # Construct badly overconfident scores (huge magnitude) that rank
    # classes correctly on most rows but are WRONG on a noisy 20% subset —
    # the classic overconfidence scenario: a model that is both sharp and
    # sometimes wrong pays a huge NLL penalty on the wrong rows, which
    # softening (temperature > 1) reduces. (Scores that are always exactly
    # right at the argmax already have near-zero NLL regardless of
    # sharpness, so a noise-free construction wouldn't exercise this at
    # all — the label flip below is what makes the test meaningful.)
    torch.manual_seed(0)
    n = 300
    true_logits = torch.randn(n, 4) * 0.5
    y = true_logits.argmax(dim=1).clone()
    n_flip = n // 5
    flip_idx = torch.randperm(n)[:n_flip]
    y[flip_idx] = (y[flip_idx] + 1) % 4
    overconfident_scores = true_logits * 20.0  # same ranking, way too sharp

    nll_before = torch.nn.functional.nll_loss(
        torch.log_softmax(overconfident_scores, dim=1), y
    )
    temperature = fit_temperature(overconfident_scores, y, n_steps=200)
    calibrated = apply_temperature(overconfident_scores, temperature)
    nll_after = torch.nn.functional.nll_loss(torch.log(calibrated), y)

    assert temperature > 1.0  # should have learned to soften, not sharpen
    assert nll_after < nll_before
