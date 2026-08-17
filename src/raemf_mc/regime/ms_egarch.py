from __future__ import annotations

import math
from dataclasses import dataclass

import torch

N_STATES = 4


@dataclass(frozen=True)
class MSEGARCHParamLayout:
    n_states: int = N_STATES

    @property
    def n_egarch_params(self) -> int:
        return 4 * self.n_states

    @property
    def n_transition_params(self) -> int:
        return self.n_states * (self.n_states - 1)

    @property
    def n_nu_params(self) -> int:
        return 1

    @property
    def total(self) -> int:
        return self.n_egarch_params + self.n_transition_params + self.n_nu_params


@dataclass(frozen=True)
class MSEGARCHParams:
    omega: torch.Tensor
    alpha: torch.Tensor
    beta: torch.Tensor
    gamma: torch.Tensor
    transition_logits: torch.Tensor
    nu_raw: torch.Tensor


def unpack_params(
    theta: torch.Tensor, layout: MSEGARCHParamLayout = MSEGARCHParamLayout()
) -> MSEGARCHParams:
    n = layout.n_states
    idx = 0
    omega = theta[idx : idx + n]
    idx += n
    alpha = theta[idx : idx + n]
    idx += n
    beta = theta[idx : idx + n]
    idx += n
    gamma = theta[idx : idx + n]
    idx += n
    transition_logits = theta[idx : idx + n * (n - 1)].reshape(n, n - 1)
    idx += n * (n - 1)
    nu_raw = theta[idx : idx + 1].squeeze(-1)
    return MSEGARCHParams(omega, alpha, beta, gamma, transition_logits, nu_raw)


def transition_matrix(transition_logits: torch.Tensor) -> torch.Tensor:
    """Each row's simplex = softmax of [0, logits_row] — fixes one free
    reference category per row so an (n_states-1)-length logit vector maps
    onto an n_states-simplex (avoids Dirichlet, which is a poor fit for
    ADVI's Gaussian variational family)."""
    n = transition_logits.shape[0]
    zeros = torch.zeros(n, 1, dtype=transition_logits.dtype, device=transition_logits.device)
    full_logits = torch.cat([zeros, transition_logits], dim=1)
    return torch.softmax(full_logits, dim=1)


def nu_from_raw(nu_raw: torch.Tensor, min_nu: float = 2.05) -> torch.Tensor:
    """nu = min_nu + softplus(nu_raw), guaranteed strictly > min_nu.

    softplus(nu_raw) is mathematically positive for every finite nu_raw, but
    for very negative nu_raw it underflows below the float ULP at min_nu
    (e.g. softplus(-100) ~ 3.7e-44, while 2.05's float64 ULP is ~4.4e-16),
    so `min_nu + softplus(nu_raw)` rounds back to exactly min_nu in any
    standard floating-point precision. Clamping to the next representable
    float above min_nu enforces the intended strict inequality without
    perturbing normal-range values, where softplus is many orders above
    the ULP and the clamp is a no-op.
    """
    nu = min_nu + torch.nn.functional.softplus(nu_raw)
    min_nu_t = torch.as_tensor(min_nu, dtype=nu.dtype, device=nu.device)
    floor = torch.nextafter(min_nu_t, torch.full_like(min_nu_t, math.inf))
    return torch.clamp(nu, min=floor)


def expected_abs_standardized_t(nu: torch.Tensor) -> torch.Tensor:
    """E|Z| for Z = a unit-variance-standardized Student-t with df=nu
    (nu > 2), used as the E|z| term in the EGARCH leverage recursion."""
    log_numer = math.log(2.0) + 0.5 * torch.log(nu - 2) + torch.lgamma((nu + 1) / 2)
    log_denom = torch.log(nu - 1) + 0.5 * math.log(math.pi) + torch.lgamma(nu / 2)
    return torch.exp(log_numer - log_denom)


def _student_t_log_pdf(
    x: torch.Tensor, loc: torch.Tensor, scale: torch.Tensor, nu: torch.Tensor
) -> torch.Tensor:
    z = (x - loc) / scale
    log_kernel = -0.5 * (nu + 1) * torch.log1p((z**2) / nu)
    log_norm = (
        torch.lgamma((nu + 1) / 2)
        - torch.lgamma(nu / 2)
        - 0.5 * torch.log(nu * torch.tensor(math.pi))
        - torch.log(scale)
    )
    return log_norm + log_kernel


def student_t_log_pdf_with_variance(
    x: torch.Tensor, loc: torch.Tensor, variance: torch.Tensor, nu: torch.Tensor
) -> torch.Tensor:
    """Student-t log-density parametrized so Var(X) = variance exactly
    (for nu > 2), rather than the raw location-scale parametrization where
    Var(X) = scale^2 * nu / (nu - 2)."""
    scale = torch.sqrt(variance * (nu - 2) / nu)
    return _student_t_log_pdf(x, loc, scale, nu)
