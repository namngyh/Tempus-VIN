from __future__ import annotations

import json
import math
from dataclasses import dataclass
from pathlib import Path
from typing import Callable

import torch


@dataclass
class AdviConfig:
    n_steps: int = 2000
    learning_rate: float = 0.01
    warmup_steps: int = 100
    grad_clip_norm: float = 10.0
    elbo_ma_window: int = 50
    early_stop_patience: int = 200
    min_delta: float = 1e-3
    retry_lr_factor: float = 0.5
    max_retries: int = 3
    n_mc_samples: int = 4


@dataclass
class FitResult:
    mu: torch.Tensor
    log_sigma: torch.Tensor
    elbo_trace: list[float]
    # NOT a convergence claim. True means only "the attempt ran to completion
    # without hitting a non-finite loss or a raising log_joint" — the ELBO may
    # still be nowhere near a stationary point. Assessing actual convergence is
    # the job of the ELBO trace and the seed-stability diagnostics. This field
    # was called `converged`, which read as a guarantee it never made.
    completed_without_divergence: bool
    fallback_used: bool
    fallback_reason: str | None
    n_retries: int
    seed: int


def assert_no_float16(*tensors: torch.Tensor) -> None:
    for t in tensors:
        if t.dtype == torch.float16:
            raise TypeError(
                "float16 tensors are not allowed in ELBO/log-likelihood/logsumexp computations"
            )


def append_fallback_log(event: dict, path: str | Path = "fallbacks.json") -> None:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    existing = []
    if path.exists():
        existing = json.loads(path.read_text())
    existing.append(event)
    path.write_text(json.dumps(existing, indent=2, default=str))


def _gaussian_entropy(log_sigma: torch.Tensor) -> torch.Tensor:
    # entropy of a diagonal Gaussian: 0.5 * sum(log(2*pi*e) + 2*log_sigma)
    return 0.5 * torch.sum(math.log(2 * math.pi * math.e) + 2 * log_sigma)


def _lr_schedule(step: int, config: AdviConfig) -> float:
    if step < config.warmup_steps and config.warmup_steps > 0:
        return (step + 1) / config.warmup_steps
    remaining = max(config.n_steps - config.warmup_steps, 1)
    progress = min((step - config.warmup_steps) / remaining, 1.0)
    return 0.5 * (1 + math.cos(math.pi * progress))


def _single_attempt(
    log_joint_fn: Callable[[torch.Tensor], torch.Tensor],
    init_mu: torch.Tensor,
    init_log_sigma: torch.Tensor,
    config: AdviConfig,
    seed: int,
    device: torch.device,
    learning_rate: float,
) -> tuple[torch.Tensor, torch.Tensor, list[float], bool, str | None]:
    """Run one ADVI attempt.

    Returns (mu, log_sigma, elbo_trace, completed_without_divergence,
    failure_reason). The returned mu/log_sigma are always guaranteed finite:
    the snapshot is only refreshed when both tensors are entirely finite, so
    a step that has a finite loss but produces NaN gradients (e.g. a 0/0
    inside log_joint) cannot poison the state that a later divergence
    reverts to. Without that guard the caller would report
    `fallback_used=True` with a reason naming a "last finite step" while
    handing back NaNs.
    """
    generator = torch.Generator(device=device).manual_seed(seed)
    mu = init_mu.clone().to(device).requires_grad_(True)
    log_sigma = init_log_sigma.clone().to(device).requires_grad_(True)
    optimizer = torch.optim.Adam([mu, log_sigma], lr=learning_rate)
    scheduler = torch.optim.lr_scheduler.LambdaLR(
        optimizer, lr_lambda=lambda step: _lr_schedule(step, config)
    )

    elbo_trace: list[float] = []
    moving_avg = None
    best_moving_avg = -math.inf
    patience_counter = 0
    last_finite_mu, last_finite_log_sigma = mu.detach().clone(), log_sigma.detach().clone()

    for step in range(config.n_steps):
        optimizer.zero_grad()
        sigma = torch.exp(log_sigma)
        elbo_samples = []
        try:
            for _ in range(config.n_mc_samples):
                eps = torch.randn(mu.shape, generator=generator, device=device)
                theta = mu + sigma * eps
                elbo_samples.append(log_joint_fn(theta))
        except (ValueError, RuntimeError, ArithmeticError) as exc:
            # log_joint can raise instead of returning a non-finite value —
            # torch.distributions argument validation rejects a NaN scale,
            # for instance. That is the same failure as a non-finite loss and
            # must take the same retry/fallback path rather than escaping
            # uncaught and bypassing it entirely.
            return (
                last_finite_mu,
                last_finite_log_sigma,
                elbo_trace,
                False,
                f"log_joint_raised: {type(exc).__name__}: {exc}",
            )
        mean_log_joint = torch.stack(elbo_samples).mean()
        elbo = mean_log_joint + _gaussian_entropy(log_sigma)
        loss = -elbo

        if not torch.isfinite(loss):
            return (
                last_finite_mu,
                last_finite_log_sigma,
                elbo_trace,
                False,
                "elbo_diverged_nan_or_inf",
            )

        loss.backward()
        torch.nn.utils.clip_grad_norm_([mu, log_sigma], config.grad_clip_norm)
        optimizer.step()
        scheduler.step()

        elbo_value = elbo.item()
        elbo_trace.append(elbo_value)
        # Only snapshot genuinely finite state. The loss above can be finite
        # while the gradients are not, in which case optimizer.step() has just
        # written NaNs into mu/log_sigma; storing that as the "last finite"
        # state would make the fallback path return NaNs under a name that
        # promises otherwise.
        if torch.isfinite(mu).all() and torch.isfinite(log_sigma).all():
            last_finite_mu = mu.detach().clone()
            last_finite_log_sigma = log_sigma.detach().clone()

        window = elbo_trace[-config.elbo_ma_window :]
        moving_avg = sum(window) / len(window)
        if moving_avg > best_moving_avg + config.min_delta:
            best_moving_avg = moving_avg
            patience_counter = 0
        else:
            patience_counter += 1
        if patience_counter >= config.early_stop_patience:
            break

    return last_finite_mu, last_finite_log_sigma, elbo_trace, True, None


def fit_mean_field_advi(
    log_joint_fn: Callable[[torch.Tensor], torch.Tensor],
    init_mu: torch.Tensor,
    init_log_sigma: torch.Tensor,
    config: AdviConfig,
    seed: int,
    device: torch.device,
    fallback_log_path: str | Path = "fallbacks.json",
) -> FitResult:
    assert_no_float16(init_mu, init_log_sigma)

    lr = config.learning_rate
    n_retries = 0
    fallback_used = False
    fallback_reason = None
    mu, log_sigma, elbo_trace, completed_without_divergence, failure_reason = _single_attempt(
        log_joint_fn, init_mu, init_log_sigma, config, seed, device, lr
    )

    while not completed_without_divergence and n_retries < config.max_retries:
        n_retries += 1
        lr = lr * config.retry_lr_factor
        event = {
            "seed": seed,
            "retry_number": n_retries,
            "new_learning_rate": lr,
            "reason": failure_reason or "elbo_diverged_nan_or_inf",
        }
        append_fallback_log(event, fallback_log_path)
        mu, log_sigma, elbo_trace, completed_without_divergence, failure_reason = _single_attempt(
            log_joint_fn, init_mu, init_log_sigma, config, seed, device, lr
        )

    if not completed_without_divergence:
        fallback_used = True
        fallback_reason = "exhausted_retries_reverted_to_last_finite_step"
        append_fallback_log(
            {
                "seed": seed,
                "reason": fallback_reason,
                "last_failure": failure_reason,
                "n_retries": n_retries,
            },
            fallback_log_path,
        )

    return FitResult(
        mu=mu,
        log_sigma=log_sigma,
        elbo_trace=elbo_trace,
        completed_without_divergence=completed_without_divergence,
        fallback_used=fallback_used,
        fallback_reason=fallback_reason,
        n_retries=n_retries,
        seed=seed,
    )


@dataclass
class PooledPosterior:
    seed_results: list[FitResult]


def fit_multi_seed_advi(
    log_joint_fn: Callable[[torch.Tensor], torch.Tensor],
    init_mu: torch.Tensor,
    init_log_sigma: torch.Tensor,
    config: AdviConfig,
    seeds: list[int],
    device: torch.device,
    fallback_log_path: str | Path = "fallbacks.json",
) -> PooledPosterior:
    """Fit one mean-field posterior per seed and pool them as an
    equal-weight mixture — never keep only the best-ELBO seed."""
    results = [
        fit_mean_field_advi(
            log_joint_fn, init_mu, init_log_sigma, config, seed, device, fallback_log_path
        )
        for seed in seeds
    ]
    return PooledPosterior(seed_results=results)


def sample_joint_draw(
    posterior: PooledPosterior, generator: torch.Generator | None = None
) -> torch.Tensor:
    """Draw exactly one flat theta vector from the pooled posterior: pick
    one seed uniformly at random, then reparameterize-sample once from
    that seed's mean-field Gaussian. The caller is responsible for
    reusing the returned tensor for an entire Monte Carlo path rather
    than resampling at each horizon step — this function performs no
    caching itself."""
    n_seeds = len(posterior.seed_results)
    idx = int(torch.randint(n_seeds, (1,), generator=generator).item())
    fr = posterior.seed_results[idx]
    eps = torch.randn(fr.mu.shape, generator=generator, device=fr.mu.device)
    return fr.mu + torch.exp(fr.log_sigma) * eps
