from __future__ import annotations

from pathlib import Path

import pandas as pd
import torch

from raemf_mc.bayesian.torch_backend import PooledPosterior, append_fallback_log
from raemf_mc.regime.ms_egarch import (
    MSEGARCHParamLayout,
    nu_from_raw,
    run_ms_egarch_recursion,
    sample_ms_egarch_draw,
    unpack_params,
)
from raemf_mc.regime.state_alignment import STATE_NAMES, align_states, apply_alignment


def canonical_theta(posterior: PooledPosterior) -> torch.Tensor:
    """Deterministic point-estimate parameter vector: the mean of the
    variational mean (mu) across all pooled seeds. Used to generate a
    single reproducible target-label sequence — distinct from the random
    posterior draws used for the volatility uncertainty features below,
    which need to reflect the posterior's actual spread, not collapse it."""
    mus = torch.stack([r.mu for r in posterior.seed_results])
    return mus.mean(dim=0)


# Nguong bao dong cho `regime_health_report`. Ba nguong nay KHONG phai tieu
# chi chat luong mo hinh -- chung la nguong "ket qua khong dung duoc cho
# muc dich da dinh", chon theo dung cai da do duoc tren du lieu that:
#
# - MAX_OCCUPANCY: mot che do chiem tren 80% occupancy nghia la mo hinh 4
#   trang thai dang hanh xu nhu mo hinh 1 trang thai. Da quan sat 0.827 tren
#   cua so ebm_smoke.
# - MIN_LABEL_CLASSES: bo phan loai khong the huan luyen tren mot lop duy
#   nhat. Day khong phai canh bao ma la dieu kien can cua tang phia sau.
# - NU_FLOOR_MARGIN: nu = 2.05 + softplus(...). Khi nu nam trong khoang
#   margin nay cua san, phan phoi Student-t co kurtosis vo han va moi con so
#   VaR/CVaR sinh ra tu no deu vo nghia (xem comment dau simulate.py ve dieu
#   kien moment cua Nelson 1991). Da quan sat nu = 2.15-2.21 tren ca ba seed.
MAX_OCCUPANCY_SHARE = 0.80
MIN_LABEL_CLASSES = 2
NU_FLOOR_MARGIN = 0.5


def regime_health_report(
    posterior: PooledPosterior,
    returns: pd.Series,
    init_log_var: torch.Tensor,
    init_log_state_prob: torch.Tensor,
    n_train: int,
    layout: MSEGARCHParamLayout = MSEGARCHParamLayout(),
    dtype: torch.dtype = torch.float32,
    device: torch.device | None = None,
) -> dict:
    """Chan doan suc khoe cua mot fit regime: fit nay co dung duoc cho tang
    phan loai phia sau khong.

    Ly do ton tai: cac chan doan da co (`fallback_used`, ELBO trace,
    `clamp_saturation_fraction`) deu bao SACH trong khi fit sinh ra nhan
    hoan toan don lop -- da xac nhan tren du lieu that, ca ba seed deu
    `fallback_used=False` con nhan thi 1049/1049 phien cung mot lop. Do
    chinh la mot silent fallback theo dung nghia AGENTS.md cam: pipeline
    bao thanh cong trong khi ket qua khong dung duoc. Ham nay bien lop loi
    do thanh mot thu do duoc va bao cao duoc.

    Tra ve dict; `healthy=False` khi bat ky nguong nao bi vi pham. Khong tu
    raise -- de nguoi goi quyet dinh dung hay ghi log, dung nhu cach
    `append_fallback_log` da lam o cac tang khac.
    """
    theta = canonical_theta(posterior)
    params = unpack_params(theta, layout)
    returns_tensor = torch.tensor(returns.to_numpy(), dtype=dtype, device=device)
    result = run_ms_egarch_recursion(
        returns_tensor, params, init_log_var, init_log_state_prob
    )
    filtered = torch.exp(result["log_filtered_prob"])
    occupancy = (filtered.sum(dim=0) / filtered.shape[0]).tolist()

    permutation = align_states(
        returns_tensor[:n_train], result["log_filtered_prob"][:n_train]
    )
    aligned = apply_alignment(result["log_filtered_prob"], permutation)
    label_idx = aligned.argmax(dim=1)
    train_labels = label_idx[:n_train]
    label_counts: dict[str, int] = {}
    for k in train_labels.tolist():
        name = STATE_NAMES[k]
        label_counts[name] = label_counts.get(name, 0) + 1

    nu = float(nu_from_raw(params.nu_raw))
    max_occupancy = max(occupancy)
    n_classes = len(label_counts)
    top_label_share = max(label_counts.values()) / max(len(train_labels), 1)

    problems = []
    if max_occupancy > MAX_OCCUPANCY_SHARE:
        problems.append(
            f"mot che do chiem {max_occupancy:.3f} occupancy "
            f"(> {MAX_OCCUPANCY_SHARE}): mo hinh 4 trang thai dang hanh xu "
            f"nhu mo hinh 1 trang thai"
        )
    if n_classes < MIN_LABEL_CLASSES:
        problems.append(
            f"chi co {n_classes} lop nhan tren tap train "
            f"(< {MIN_LABEL_CLASSES}): tang phan loai khong huan luyen duoc"
        )
    if nu < 2.05 + NU_FLOOR_MARGIN:
        problems.append(
            f"nu = {nu:.3f} nam sat san 2.05 (margin {NU_FLOOR_MARGIN}): "
            f"Student-t co kurtosis vo han, moi so VaR/CVaR sinh ra tu fit "
            f"nay deu vo nghia"
        )

    return {
        "healthy": not problems,
        "problems": problems,
        "occupancy": [round(o, 4) for o in occupancy],
        "max_occupancy_share": round(max_occupancy, 4),
        "n_label_classes": n_classes,
        "top_label_share": round(top_label_share, 4),
        "label_counts": label_counts,
        "nu": round(nu, 4),
        "clamp_saturation_fraction": result["clamp_saturation_fraction"],
    }


def compute_posterior_volatility_features(
    posterior: PooledPosterior,
    returns: pd.Series,
    init_log_var: torch.Tensor,
    init_log_state_prob: torch.Tensor,
    layout: MSEGARCHParamLayout = MSEGARCHParamLayout(),
    n_draws: int = 50,
    generator: torch.Generator | None = None,
    dtype: torch.dtype = torch.float32,
    device: torch.device | None = None,
) -> pd.DataFrame:
    """Posterior-mean and posterior-sd of the collapsed sigma_bar_t at every
    session, estimated from n_draws independent posterior draws.

    sigma_bar_t = exp(0.5 * log_var_bar[t]) is the level-space-collapsed
    expected volatility at t under the filtered state distribution — the
    natural "posterior sigma_t" quantity, already computed by
    run_ms_egarch_recursion via Gray's collapsing (Sub-project 1).

    dtype/device follow the same convention as default_recursion_init: they
    govern only the tensors constructed HERE from the pandas input. `params`
    (from sample_ms_egarch_draw) and init_log_var/init_log_state_prob are the
    caller's responsibility to already have on the matching device, exactly
    as run_ms_egarch_recursion assumes elsewhere. Defaults preserve the
    previous unconditional CPU/float32 behavior.
    """
    returns_tensor = torch.tensor(returns.to_numpy(), dtype=dtype, device=device)
    sigma_draws = torch.zeros(n_draws, len(returns), dtype=dtype, device=device)
    for i in range(n_draws):
        params = sample_ms_egarch_draw(posterior, layout, generator=generator)
        result = run_ms_egarch_recursion(
            returns_tensor, params, init_log_var, init_log_state_prob
        )
        sigma_draws[i] = torch.exp(0.5 * result["log_var_bar"])
    posterior_mean_sigma = sigma_draws.mean(dim=0).cpu()
    posterior_sd_sigma = sigma_draws.std(dim=0, unbiased=False).cpu()
    return pd.DataFrame(
        {
            "posterior_mean_sigma": posterior_mean_sigma.numpy(),
            "posterior_sd_sigma": posterior_sd_sigma.numpy(),
        },
        index=returns.index,
    )


def _health_from_result(
    result: dict,
    returns_tensor: torch.Tensor,
    label_idx: list[int],
    n_train: int,
    params,
    layout: MSEGARCHParamLayout,
) -> dict:
    """Cung logic nguong voi `regime_health_report` nhung dung lai ket qua
    recursion da tinh, de `compute_regime_labels` khong phai chay lai bo loc
    lan thu hai chi de bao cao."""
    filtered = torch.exp(result["log_filtered_prob"])
    occupancy = (filtered.sum(dim=0) / filtered.shape[0]).tolist()
    label_counts: dict[str, int] = {}
    for k in label_idx[:n_train]:
        name = STATE_NAMES[k]
        label_counts[name] = label_counts.get(name, 0) + 1
    nu = float(nu_from_raw(params.nu_raw))
    max_occupancy = max(occupancy)
    n_classes = len(label_counts)

    problems = []
    if max_occupancy > MAX_OCCUPANCY_SHARE:
        problems.append(f"occupancy toi da {max_occupancy:.3f} > {MAX_OCCUPANCY_SHARE}")
    if n_classes < MIN_LABEL_CLASSES:
        problems.append(f"chi {n_classes} lop nhan < {MIN_LABEL_CLASSES}")
    if nu < 2.05 + NU_FLOOR_MARGIN:
        problems.append(f"nu = {nu:.3f} sat san 2.05")
    return {
        "healthy": not problems,
        "problems": problems,
        "occupancy": [round(o, 4) for o in occupancy],
        "max_occupancy_share": round(max_occupancy, 4),
        "n_label_classes": n_classes,
        "label_counts": label_counts,
        "nu": round(nu, 4),
    }


def compute_regime_labels(
    posterior: PooledPosterior,
    returns: pd.Series,
    init_log_var: torch.Tensor,
    init_log_state_prob: torch.Tensor,
    n_train: int,
    layout: MSEGARCHParamLayout = MSEGARCHParamLayout(),
    dtype: torch.dtype = torch.float32,
    device: torch.device | None = None,
    health_log_path: str | Path | None = None,
) -> pd.Series:
    """Deterministic target label per session: argmax of the train-aligned
    filtered regime probability, using the canonical (posterior-mean)
    parameter point estimate — a single reproducible ground-truth proxy
    sequence, not an ensemble (unlike the volatility features above, which
    deliberately use random draws to capture posterior spread).

    The state->name permutation (align_states) is fit using ONLY the first
    n_train rows, then applied to the full series — never let val/test rows
    influence which raw state is called "Bull" vs "Bear", or the training
    targets themselves become a function of held-out data. apply_alignment is
    a pure column reindex, so applying a train-fit permutation to the whole
    series introduces no leakage; running align_states over the whole series
    does, and demonstrably changed the permutation on real data.
    """
    theta = canonical_theta(posterior)
    params = unpack_params(theta, layout)
    returns_tensor = torch.tensor(returns.to_numpy(), dtype=dtype, device=device)
    result = run_ms_egarch_recursion(
        returns_tensor, params, init_log_var, init_log_state_prob
    )
    permutation = align_states(
        returns_tensor[:n_train], result["log_filtered_prob"][:n_train]
    )
    aligned = apply_alignment(result["log_filtered_prob"], permutation)
    label_idx = aligned.argmax(dim=1).tolist()
    labels = [STATE_NAMES[i] for i in label_idx]
    labels_series = pd.Series(labels, index=returns.index, name="regime_label")

    if health_log_path is not None:
        # Khong tinh lai recursion: dung lai ket qua vua co o tren.
        report = _health_from_result(result, returns_tensor, label_idx, n_train, params, layout)
        if not report["healthy"]:
            append_fallback_log(
                {"reason": "degenerate_regime_fit", **report}, health_log_path
            )
    return labels_series
