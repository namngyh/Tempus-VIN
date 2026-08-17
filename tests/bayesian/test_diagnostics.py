import torch
from raemf_mc.bayesian.torch_backend import FitResult
from raemf_mc.bayesian.diagnostics import summarize_elbo_trace, seed_stability_metrics


def test_summarize_elbo_trace_reports_final_and_best():
    trace = [-10.0, -5.0, -6.0, -4.0, -4.5]
    summary = summarize_elbo_trace(trace)
    assert summary["final_elbo"] == -4.5
    assert summary["best_elbo"] == -4.0
    assert summary["n_steps"] == 5


def test_seed_stability_metrics_zero_for_identical_seeds():
    results = [
        FitResult(mu=torch.zeros(2), log_sigma=torch.zeros(2), elbo_trace=[-1.0, -1.0],
                   converged=True, fallback_used=False, fallback_reason=None, n_retries=0, seed=s)
        for s in range(3)
    ]
    metrics = seed_stability_metrics(results)
    assert metrics["final_elbo_std"] == 0.0
    assert metrics["max_pairwise_mu_diff"] == 0.0


def test_seed_stability_metrics_nonzero_for_divergent_seeds():
    results = [
        FitResult(mu=torch.zeros(2), log_sigma=torch.zeros(2), elbo_trace=[-1.0],
                   converged=True, fallback_used=False, fallback_reason=None, n_retries=0, seed=0),
        FitResult(mu=torch.ones(2) * 5, log_sigma=torch.zeros(2), elbo_trace=[-9.0],
                   converged=True, fallback_used=False, fallback_reason=None, n_retries=0, seed=1),
    ]
    metrics = seed_stability_metrics(results)
    assert metrics["final_elbo_std"] > 0.0
    assert metrics["max_pairwise_mu_diff"] > 0.0
