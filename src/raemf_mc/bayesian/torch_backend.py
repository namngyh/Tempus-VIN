from __future__ import annotations

import json
import math
from contextlib import contextmanager
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
    # Ghi toan bo mot buoc ADVI thanh mot CUDA graph roi phat lai. MAC DINH
    # TAT: doi hoi CUDA that, log_joint co supports_batched_theta, va no doi
    # mot vai chi tiet ve hanh vi (xem _single_attempt). Bat len khi va chi
    # khi da doc nhung danh doi do.
    #
    # Do duoc tren RTX 4060 (GeForce/WDDM): overhead phat kernel ~42 us o che
    # do eager va ~1.5 us khi phat lai graph. Mot buoc ADVI day du o T=1500,
    # S=16: CPU 2195 ms, CUDA eager 8852 ms, CUDA graph 341 ms.
    use_cuda_graph: bool = False


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


def _reset_adam_state_in_place(optimizer: torch.optim.Optimizer) -> None:
    """Dua trang thai Adam ve dung diem xuat phat, TAI CHO.

    Khong duoc xoa `optimizer.state` o day. Xoa dict chi bo tham chieu; cac
    tensor moment van ton tai va CUDA graph da ghi cung dia chi cua chinh
    chung luc capture. Ket qua la graph tiep tuc doc/ghi nhung moment da
    duoc ham nong boi cac buoc lam nong, trong khi duong eager khoi hanh tu
    0 -- hai quy dao tach nhau ngay SAU buoc dau tien.

    Da quan sat truc tiep truoc khi sua: elbo[0] cua hai duong trung khit
    (-139.9086) con elbo cuoi lech toi 6.5e2. Chinh cho nay la ly do phep
    doi chieu eager/graph ton tai.
    """
    for state in optimizer.state.values():
        for key in ("exp_avg", "exp_avg_sq", "max_exp_avg_sq", "step"):
            buffer = state.get(key)
            if isinstance(buffer, torch.Tensor):
                buffer.zero_()


@contextmanager
def _distribution_validation_disabled():
    """Tat kiem tra tham so cua torch.distributions trong pham vi khoi lenh.

    Chi dung quanh doan GHI CUDA graph. Ly do: `Distribution.__init__` voi
    validate_args=True chay `support.check(...).all()` roi ep sang bool tren
    HOST -- mot phep dong bo, va moi dong bo deu bat hop phap trong luc ghi
    graph. Da xac nhan bang thi nghiem: cung mot log_prob capture that bai
    voi kiem tra bat, thanh cong khi tat.

    Day la mot DANH DOI THAT, khong phai chi tiet ky thuat. Docstring cua
    `_single_attempt` ghi ro rang no dua vao chinh co che kiem tra nay de
    bat scale NaN va chuyen sang duong retry voi ly do "log_joint_raised".
    Tren duong graph, mot scale NaN thay vao do se cho log_prob NaN -> loss
    NaN -> bi bat boi kiem tra isfinite(loss), tuc VAN vao dung duong retry,
    chi khac nhan ly do ("elbo_diverged_nan_or_inf"). Hanh vi duoc bao toan;
    thu mat la do phan giai cua chan doan.

    Pham vi duoc giu hep nhat co the: chi bao quanh luc ghi, khong bao quanh
    vong lap phat lai, va duong eager khong bi anh huong chut nao.
    """
    previous = torch.distributions.Distribution._validate_args
    torch.distributions.Distribution.set_default_validate_args(False)
    try:
        yield
    finally:
        torch.distributions.Distribution.set_default_validate_args(previous)


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

    `config.use_cuda_graph` ghi mot buoc thanh CUDA graph roi phat lai. VONG
    LAP LA MOT, chung cho ca hai duong -- chi ba cho re nhanh: cach lay eps,
    cach cap nhat learning rate, va viec goi step hay phat lai graph. Day la
    co y: hai vong lap song song se troi khoi nhau, va toan bo logic phan ky
    / anh chup / dung som la phan da qua nhieu vong review.

    Ba danh doi cua duong graph, deu that:

    1. `log_joint` chi co the raise o LUC GHI graph, khong phai luc phat lai.
       Sau khi ghi, do thi la co dinh: mot theta gay NaN se cho loss NaN chu
       khong raise, va bi bat boi kiem tra isfinite y nhu duong eager. Luoi an
       toan van con, chi doi hinh thuc.
    2. Learning rate phai la TENSOR (LambdaLR gan mot so Python, va so do bi
       nuong cung vao graph luc ghi, lam dong bang lich warmup). Duong graph
       tu cap nhat lr tai cho voi DUNG cong thuc ma LambdaLR dung.
    3. Adam phai co capturable=True.
    4. Kiem tra tham so cua torch.distributions bi tat trong luc GHI graph
       (xem `_distribution_validation_disabled`). Mot scale NaN van vao dung
       duong retry, qua kiem tra isfinite(loss) thay vi qua mot exception,
       nen chi khac nhan ly do chu khong khac hanh vi.
    """
    graphed = config.use_cuda_graph
    if graphed:
        # Khong im lang tut xuong duong eager: nguoi goi da yeu cau graph, va
        # chay cham hon 30 lan ma khong bao gi la dung kieu that bai am tham
        # ma du an nay cam.
        if device.type != "cuda":
            raise ValueError(
                f"use_cuda_graph=True nhung device={device} khong phai CUDA"
            )
        if not getattr(log_joint_fn, "supports_batched_theta", False):
            raise ValueError(
                "use_cuda_graph=True doi hoi log_joint co supports_batched_theta "
                "= True (duong tuan tu nhieu MC sample khong ghi graph duoc)"
            )

    generator = torch.Generator(device=device).manual_seed(seed)
    mu = init_mu.clone().to(device).requires_grad_(True)
    log_sigma = init_log_sigma.clone().to(device).requires_grad_(True)

    eps_shape = (config.n_mc_samples, *mu.shape)
    if graphed:
        lr_tensor = torch.tensor(learning_rate, dtype=mu.dtype, device=device)
        optimizer = torch.optim.Adam([mu, log_sigma], lr=lr_tensor, capturable=True)
        scheduler = None
        static_eps = torch.empty(eps_shape, dtype=mu.dtype, device=device)
    else:
        lr_tensor = None
        optimizer = torch.optim.Adam([mu, log_sigma], lr=learning_rate)
        scheduler = torch.optim.lr_scheduler.LambdaLR(
            optimizer, lr_lambda=lambda step: _lr_schedule(step, config)
        )
        static_eps = None

    def _forward_backward() -> tuple[torch.Tensor, torch.Tensor]:
        """Mot buoc: eps -> theta -> ELBO -> backward -> clip -> optimizer.
        Doc `static_eps` (duong graph) hoac tu rut eps (duong eager)."""
        optimizer.zero_grad(set_to_none=False)
        sigma = torch.exp(log_sigma)
        if graphed:
            theta = mu + sigma * static_eps
            mean_log_joint = log_joint_fn(theta).mean()
        elif getattr(log_joint_fn, "supports_batched_theta", False):
            eps = torch.randn(eps_shape, generator=generator, device=device)
            theta = mu + sigma * eps
            mean_log_joint = log_joint_fn(theta).mean()
        else:
            elbo_samples = []
            for _ in range(config.n_mc_samples):
                eps = torch.randn(mu.shape, generator=generator, device=device)
                theta = mu + sigma * eps
                elbo_samples.append(log_joint_fn(theta))
            mean_log_joint = torch.stack(elbo_samples).mean()
        elbo = mean_log_joint + _gaussian_entropy(log_sigma)
        loss = -elbo
        loss.backward()
        torch.nn.utils.clip_grad_norm_([mu, log_sigma], config.grad_clip_norm)
        optimizer.step()
        return elbo, loss

    graph = None
    static_elbo = static_loss = None
    if graphed:
        static_eps.normal_(generator=generator)
        warmup_stream = torch.cuda.Stream()
        warmup_stream.wait_stream(torch.cuda.current_stream())
        with torch.cuda.stream(warmup_stream):
            for _ in range(3):
                _forward_backward()
        torch.cuda.current_stream().wait_stream(warmup_stream)
        # Cac buoc lam nong o tren da lam ban mu/log_sigma/trang thai Adam.
        # Dat lai ve dung diem xuat phat de quy dao khong phu thuoc vao so
        # lan lam nong -- neu khong, ket qua cua duong graph se khac duong
        # eager theo mot cach khong ai giai thich duoc.
        with torch.no_grad():
            mu.copy_(init_mu.to(device))
            log_sigma.copy_(init_log_sigma.to(device))
        _reset_adam_state_in_place(optimizer)
        generator.manual_seed(seed)
        torch.cuda.synchronize()

        graph = torch.cuda.CUDAGraph()
        with _distribution_validation_disabled(), torch.cuda.graph(graph):
            static_elbo, static_loss = _forward_backward()
        torch.cuda.synchronize()
        with torch.no_grad():
            mu.copy_(init_mu.to(device))
            log_sigma.copy_(init_log_sigma.to(device))
        _reset_adam_state_in_place(optimizer)
        generator.manual_seed(seed)

    elbo_trace: list[float] = []
    moving_avg = None
    best_moving_avg = -math.inf
    patience_counter = 0
    last_finite_mu, last_finite_log_sigma = mu.detach().clone(), log_sigma.detach().clone()

    for step in range(config.n_steps):
        if graphed:
            # Cung cong thuc lr ma LambdaLR dung o duong eager, cap nhat tai
            # cho vi gia tri lr da duoc nuong vao graph luc ghi.
            lr_tensor.fill_(learning_rate * _lr_schedule(step, config))
            static_eps.normal_(generator=generator)
            graph.replay()
            elbo, loss = static_elbo, static_loss
        else:
            try:
                elbo, loss = _forward_backward()
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
            scheduler.step()

        if not torch.isfinite(loss):
            return (
                last_finite_mu,
                last_finite_log_sigma,
                elbo_trace,
                False,
                "elbo_diverged_nan_or_inf",
            )

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

    @property
    def n_fallback_seeds(self) -> int:
        """How many pooled seeds ended on the fallback path.

        Pooling stays equal-weight over ALL seeds — dropping the unhealthy
        ones would be cherry-picking, which the project explicitly forbids.
        But a pooled posterior where half the seeds fell back is a very
        different object from one where none did, and that distinction was
        previously invisible to every consumer of the pool. Exposing it lets
        callers log/report it without changing the sampling behaviour.
        """
        return sum(1 for r in self.seed_results if r.fallback_used)

    @property
    def fallback_fraction(self) -> float:
        if not self.seed_results:
            return 0.0
        return self.n_fallback_seeds / len(self.seed_results)

    def fallback_summary(self) -> dict:
        """Loggable snapshot of pool health, including per-seed reasons."""
        return {
            "n_seeds": len(self.seed_results),
            "n_fallback_seeds": self.n_fallback_seeds,
            "fallback_fraction": self.fallback_fraction,
            "fallback_seeds": [
                {"seed": r.seed, "reason": r.fallback_reason, "n_retries": r.n_retries}
                for r in self.seed_results
                if r.fallback_used
            ],
        }


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
    posterior = PooledPosterior(seed_results=results)
    if posterior.n_fallback_seeds:
        # Pool composition stays equal-weight (no cherry-picking), but a pool
        # containing fallback seeds must not look identical to a healthy one.
        append_fallback_log(
            {"reason": "pooled_posterior_contains_fallback_seeds", **posterior.fallback_summary()},
            fallback_log_path,
        )
    return posterior


def sample_joint_draws(
    posterior: PooledPosterior, n_draws: int, generator: torch.Generator | None = None
) -> torch.Tensor:
    """Rút `n_draws` vector theta cùng lúc, trả về `(n_draws, P)`.

    Ngữ nghĩa từng hàng giống hệt `sample_joint_draw`: mỗi hàng chọn ĐỘC LẬP
    một seed theo phân phối đều rồi rút một mẫu reparameterize từ Gaussian
    mean-field của seed đó. Khác biệt duy nhất là số lần gọi RNG — một tensor
    `(n_draws,)` chỉ số cộng một tensor `(n_draws, P)` nhiễu, thay vì
    `n_draws` cặp lời gọi riêng lẻ. Vì vậy chuỗi số ngẫu nhiên KHÔNG trùng
    với việc gọi `sample_joint_draw` lặp `n_draws` lần; tính tái lập theo
    seed vẫn nguyên vẹn (cùng generator -> cùng kết quả).

    Tồn tại để `simulate_mc_paths` rút tham số cho toàn bộ path trong một
    lần, tiền đề cho việc chạy MỘT recursion batch thay vì n_paths recursion
    tuần tự.
    """
    device = posterior.seed_results[0].mu.device
    n_seeds = len(posterior.seed_results)
    idx = torch.randint(n_seeds, (n_draws,), generator=generator, device=device)
    mu_stack = torch.stack([r.mu for r in posterior.seed_results])
    log_sigma_stack = torch.stack([r.log_sigma for r in posterior.seed_results])
    mu_sel = mu_stack[idx]
    log_sigma_sel = log_sigma_stack[idx]
    eps = torch.randn(mu_sel.shape, generator=generator, device=device)
    return mu_sel + torch.exp(log_sigma_sel) * eps


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
    # device= là BẮT BUỘC: torch.randint mặc định cấp phát trên CPU, và một
    # generator CUDA dùng với tensor CPU raise
    # "Expected a 'cpu' device type for generator but found 'cuda'".
    # Đây là lỗi đường GPU thật, đã tái hiện được trên RTX 4060 trước khi sửa
    # (README ghi "đường GPU chưa từng chạy thật" — đây chính là cái nó ẩn).
    device = posterior.seed_results[0].mu.device
    idx = int(torch.randint(n_seeds, (1,), generator=generator, device=device).item())
    fr = posterior.seed_results[idx]
    eps = torch.randn(fr.mu.shape, generator=generator, device=fr.mu.device)
    return fr.mu + torch.exp(fr.log_sigma) * eps
