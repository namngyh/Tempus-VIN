from __future__ import annotations

from pathlib import Path

import pandas as pd
import torch

from raemf_mc.bayesian.priors import HierarchicalPriorConfig
from raemf_mc.bayesian.torch_backend import AdviConfig
from raemf_mc.evaluation.probabilistic_metrics import compute_crps, compute_wis
from raemf_mc.evaluation.var_backtest import compute_kupiec_lr
from raemf_mc.regime.ms_egarch import (
    MSEGARCHParamLayout,
    default_recursion_init,
    fit_ms_egarch,
)
from raemf_mc.risk.metrics import compute_var
from raemf_mc.scenario.mu_fit import fit_regime_mu
from raemf_mc.scenario.simulate import simulate_mc_paths


def run_walk_forward_evaluation(
    log_returns: pd.Series,
    cutoffs: list[int],
    horizon: int,
    ms_egarch_advi: AdviConfig,
    ms_egarch_prior: HierarchicalPriorConfig,
    mu_advi: AdviConfig,
    mu_n_draws: int,
    mu_prior_scale: float,
    mu_min_effective_observations: float,
    mu_min_effective_fraction: float,
    mc_n_paths: int,
    layout: MSEGARCHParamLayout,
    seeds: list[int],
    device: torch.device,
    fallback_log_dir: Path,
    var_alphas: tuple[float, ...] = (0.95, 0.99),
    generator: torch.Generator | None = None,
) -> tuple[pd.DataFrame, dict[str, float]]:
    """Đánh giá out-of-sample kiểu walk-forward.

    Tại mỗi mốc, fit MS-EGARCH + mu_k trên CHỈ dữ liệu tới mốc đó (không bao
    giờ dữ liệu sau) — đây là điều làm cho nó thực sự out-of-sample, khác với
    integration test của mọi sub-project trước vốn fit và đánh giá trên cùng
    một cửa sổ. Rồi mô phỏng `horizon` ngày tới và chấm điểm phân phối mô
    phỏng đối chiếu với lợi suất ĐÃ THỰC SỰ xảy ra tiếp theo trong lịch sử.

    Trả về (DataFrame theo từng mốc, dict thống kê Kupiec gộp).

    Ba điểm chống rò rỉ, mỗi điểm đều là một lỗi mà sub-project trước đã mắc
    và phải sửa một lần:

    1. `train_mean` tính trên RIÊNG cửa sổ fit của mốc đó, không phải trung
       bình toàn chuỗi. Dùng trung bình toàn chuỗi là để tương lai rò vào
       quá khứ qua đúng một con số.
    2. `simulate_mc_paths` trả về lợi suất ĐÃ TRỪ TRUNG BÌNH (theo docstring
       của chính nó). Phải cộng `horizon * train_mean` trở lại TRƯỚC mọi so
       sánh với thế giới thật — đây đúng là lỗi khứ hồi drift mà whole-branch
       review của sub-project 3 đã tìm ra.
    3. Mốc cuối cộng horizon phải nằm trong dữ liệu, nếu không thì lợi suất
       "thực tế" bị cắt cụt và CRPS/WIS được tính trên một chân trời ngắn hơn
       chân trời đã mô phỏng — sai lệch âm thầm theo hướng có lợi.

    CẢNH BÁO VỀ Ý NGHĨA THỐNG KÊ: ở quy mô smoke (ít mốc, ADVI rất nhẹ) hàm
    này xác nhận đường ống nối đúng. Nó KHÔNG phải một benchmark OOS có ý
    nghĩa thống kê: kiểm định Kupiec gộp cần hàng chục mốc trở lên mới có
    sức mạnh, còn CRPS/WIS thì phản ánh trực tiếp chất lượng của một fit ADVI
    quy mô smoke (nhiều khả năng chưa hội tụ), không phải tính chất của bản
    thân metric.
    """
    if horizon <= 0:
        raise ValueError(f"horizon phải dương, nhận {horizon}")
    if not cutoffs:
        raise ValueError("cần ít nhất một mốc cutoff")

    fallback_log_dir = Path(fallback_log_dir)
    rows: list[dict[str, float]] = []

    for cutoff in cutoffs:
        future_returns = log_returns.iloc[cutoff : cutoff + horizon]
        if len(future_returns) < horizon:
            raise ValueError(
                f"cutoff {cutoff} + horizon {horizon} vượt quá dữ liệu sẵn có "
                f"(chuỗi có {len(log_returns)} dòng)"
            )
        fit_returns = log_returns.iloc[:cutoff]
        # Trung bình của RIÊNG cửa sổ fit tại mốc này.
        train_mean = float(fit_returns.mean())
        centered_fit = fit_returns - train_mean
        returns_tensor = torch.tensor(
            centered_fit.to_numpy(), dtype=torch.float32, device=device
        )

        ms_egarch_posterior = fit_ms_egarch(
            returns_tensor, ms_egarch_advi, ms_egarch_prior, seeds, device, layout,
            fallback_log_path=fallback_log_dir / f"cutoff_{cutoff}_ms_egarch.json",
        )
        init_log_var, init_log_state_prob = default_recursion_init(
            layout, dtype=returns_tensor.dtype, device=device
        )
        mu_posterior = fit_regime_mu(
            returns_tensor, ms_egarch_posterior, mu_advi, seeds, device,
            init_log_var, init_log_state_prob, layout=layout,
            n_draws=mu_n_draws, mu_prior_scale=mu_prior_scale,
            min_effective_observations=mu_min_effective_observations,
            min_effective_fraction=mu_min_effective_fraction,
            generator=generator,
            fallback_log_path=fallback_log_dir / f"cutoff_{cutoff}_mu.json",
        )
        daily_paths = simulate_mc_paths(
            ms_egarch_posterior, mu_posterior, returns_tensor,
            init_log_var, init_log_state_prob,
            n_paths=mc_n_paths, horizon=horizon, layout=layout, device=device,
            generator=generator,
            fallback_log_path=fallback_log_dir / f"cutoff_{cutoff}_mc.json",
        )
        # Cộng drift trở lại: `horizon` ngày, mỗi ngày một lần `train_mean`.
        horizon_returns_real = (
            daily_paths.cpu().numpy().sum(axis=1) + horizon * train_mean
        )
        realized_return = float(future_returns.sum())

        row: dict[str, float] = {
            "cutoff": cutoff,
            "realized_return": realized_return,
            "crps": compute_crps(horizon_returns_real, realized_return),
            "wis": compute_wis(horizon_returns_real, realized_return),
        }
        for alpha in var_alphas:
            pct = int(round(alpha * 100))
            var_alpha = compute_var(horizon_returns_real, alpha)
            row[f"var_{pct}"] = var_alpha
            # `compute_var` trả về độ lớn khoản lỗ (số dương), nên vi phạm là
            # khi lợi suất thực tế thấp hơn -VaR.
            row[f"breached_{pct}"] = realized_return < -var_alpha
        rows.append(row)

    results_df = pd.DataFrame(rows)
    n = len(results_df)
    kupiec: dict[str, float] = {}
    for alpha in var_alphas:
        pct = int(round(alpha * 100))
        x = int(results_df[f"breached_{pct}"].sum())
        kupiec[f"kupiec_lr_{pct}"] = compute_kupiec_lr(n, x, 1.0 - alpha)
    return results_df, kupiec
