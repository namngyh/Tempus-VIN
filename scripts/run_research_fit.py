"""Chay fit MS-EGARCH o quy mo nghien cuu va bao cao suc khoe cua ket qua.

Day KHONG phai pipeline day du (chua co tang EBM/Monte Carlo/rui ro o day) --
no tra loi dung mot cau hoi dang chan moi thu phia sau: sau khi cho nu rieng
tung che do va chay ADVI du lau, fit co con suy bien khong?

Moi con so bao cao deu la so do, khong phai muc tieu: `regime_health_report`
tu quyet dinh healthy/khong theo nguong da dat truoc trong
`raemf_mc.regime.posterior_features`.

Dung:
    python scripts/run_research_fit.py --config configs/gpu_research.yaml
    python scripts/run_research_fit.py --config configs/gpu_research.yaml --seeds 0 1
"""
from __future__ import annotations

import argparse
import json
import os
import time
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path

import torch
import yaml

from raemf_mc.bayesian.priors import HierarchicalPriorConfig
from raemf_mc.bayesian.torch_backend import AdviConfig, PooledPosterior
from raemf_mc.data.loader import load_vnindex_ohlcv
from raemf_mc.features.returns import compute_log_returns
from raemf_mc.regime.ms_egarch import (
    MSEGARCHParamLayout,
    default_recursion_init,
    fit_ms_egarch,
    nu_from_raw,
    transition_matrix,
    unpack_params,
)
from raemf_mc.regime.posterior_features import canonical_theta, regime_health_report
from raemf_mc.runtime.hardware import hardware_report, select_device
from raemf_mc.validation.chronological_split import chronological_split


def _fit_one_seed(job: tuple) -> object:
    """Fit MOT seed trong mot tien trinh rieng. Tra ve FitResult.

    Chay song song theo tien trinh chu khong phai gop vao mot chieu batch:
    cac seed doc lap hoan toan, nhung logic retry/fallback cua ADVI la THEO
    TUNG SEED, va gop chung lai doi hoi viet lai phan do bang mat na theo
    hang -- cham dung quy tac "khong silent fallback" cua du an. Song song
    theo tien trinh cho gan het loi ich do ma khong dung vao phan da review:
    da do 4 fit song song chi cham hon 18% so voi chay don, vi tensor qua
    nho de mot tien trinh dung het nhieu nhan.
    """
    returns_numpy, advi_kwargs, prior_kwargs, seed, n_states, log_dir = job
    os.environ["OMP_NUM_THREADS"] = "2"
    os.environ["MKL_NUM_THREADS"] = "2"
    import torch as _torch

    _torch.set_num_threads(2)

    from raemf_mc.bayesian.priors import HierarchicalPriorConfig as _Prior
    from raemf_mc.bayesian.torch_backend import AdviConfig as _Advi
    from raemf_mc.regime.ms_egarch import MSEGARCHParamLayout as _Layout
    from raemf_mc.regime.ms_egarch import fit_ms_egarch as _fit

    layout = _Layout(n_states=n_states)
    tensor = _torch.tensor(returns_numpy, dtype=_torch.float32)
    started = time.perf_counter()
    posterior = _fit(
        tensor, _Advi(**advi_kwargs), _Prior(**prior_kwargs), [seed],
        _torch.device("cpu"), layout,
        fallback_log_path=Path(log_dir) / f"research_fallbacks_seed{seed}.json",
    )
    result = posterior.seed_results[0]
    print(f"    seed {seed} xong sau {(time.perf_counter() - started) / 60:.1f} phut "
          f"({len(result.elbo_trace)} buoc, fallback={result.fallback_used})", flush=True)
    return result


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument("--seeds", type=int, nargs="*", default=None,
                        help="Ghi de danh sach seed trong config.")
    parser.add_argument("--out", type=Path, default=None,
                        help="Ghi bao cao JSON ra duong dan nay.")
    parser.add_argument("--parallel", action="store_true",
                        help="Fit moi seed trong mot tien trinh rieng. Chi ho tro "
                             "device CPU (nhieu tien trinh dung chung mot GPU khong "
                             "cho loi ich tuong ung).")
    args = parser.parse_args()

    with args.config.open(encoding="utf-8") as f:
        config = yaml.safe_load(f)

    preference = config.get("device_preference", "auto")
    device = select_device(preference)
    print("=" * 70)
    for key, value in hardware_report(preference).items():
        print(f"  {key}: {value}")
    print(f"  config: {args.config}")
    print("=" * 70, flush=True)

    ohlcv = load_vnindex_ohlcv()
    log_returns = compute_log_returns(ohlcv)
    window = config.get("window_sessions")
    if window:
        log_returns = log_returns.iloc[-window:]

    train_frac = config.get("train_frac", 0.7)
    val_frac = config.get("val_frac", 0.15)
    returns_train, _, _ = chronological_split(
        log_returns.to_frame("ret"), train_frac=train_frac, val_frac=val_frac
    )
    # Trung tam hoa bang trung binh cua RIENG tap train, ap len ca chuoi --
    # cung ky luat chong ro ri cua moi sub-project truoc.
    centered = log_returns - returns_train["ret"].mean()
    n_train = len(returns_train)

    layout = MSEGARCHParamLayout()
    advi = AdviConfig(**config["advi"])
    prior = HierarchicalPriorConfig(**config["prior"])
    seeds = args.seeds if args.seeds is not None else config["seeds"]

    print(f"T = {len(centered)} phien (train = {n_train})")
    print(f"seeds = {seeds}, n_steps = {advi.n_steps}, "
          f"n_mc_samples = {advi.n_mc_samples}, cuda_graph = {advi.use_cuda_graph}")
    print(flush=True)

    train_tensor = torch.tensor(
        centered.loc[returns_train.index].to_numpy(), dtype=torch.float32
    ).to(device)

    log_dir = args.out.parent if args.out else Path(".")
    log_dir.mkdir(parents=True, exist_ok=True)
    started = time.perf_counter()
    if args.parallel:
        if device.type != "cpu":
            raise ValueError(
                f"--parallel chi ho tro device CPU, nhan {device}. Nhieu tien "
                f"trinh dung chung mot GPU khong cho loi ich tuong ung."
            )
        jobs = [
            (
                centered.loc[returns_train.index].to_numpy(),
                vars(advi).copy(),
                vars(prior).copy(),
                seed,
                layout.n_states,
                str(log_dir),
            )
            for seed in seeds
        ]
        print(f"  chay {len(jobs)} seed song song...", flush=True)
        with ProcessPoolExecutor(max_workers=len(jobs)) as executor:
            results = list(executor.map(_fit_one_seed, jobs))
        posterior = PooledPosterior(seed_results=results)
    else:
        posterior = fit_ms_egarch(
            train_tensor, advi, prior, seeds, device, layout,
            fallback_log_path=log_dir / "research_fallbacks.json",
        )
    elapsed = time.perf_counter() - started
    print(f"fit xong sau {elapsed / 60:.1f} phut "
          f"({elapsed / max(len(seeds), 1) / 60:.1f} phut/seed)")

    for r in posterior.seed_results:
        tail = r.elbo_trace[-1] if r.elbo_trace else float("nan")
        print(f"  seed {r.seed}: elbo_cuoi = {tail:12.2f}  n_buoc = {len(r.elbo_trace):5d}"
              f"  fallback = {r.fallback_used}  {r.fallback_reason or ''}")
    print(flush=True)

    init_log_var, init_log_state_prob = default_recursion_init(
        layout, dtype=train_tensor.dtype, device=device
    )
    report = regime_health_report(
        posterior, centered, init_log_var, init_log_state_prob,
        n_train=n_train, layout=layout, device=device,
    )

    theta = canonical_theta(posterior, layout)
    params = unpack_params(theta, layout)
    nu = nu_from_raw(params.nu_raw)
    trans = transition_matrix(params.transition_logits)

    print("=" * 70)
    print("SUC KHOE FIT")
    print("=" * 70)
    print(f"  healthy               : {report['healthy']}")
    for problem in report["problems"]:
        print(f"    - {problem}")
    print(f"  occupancy             : {report['occupancy']}")
    print(f"  occupancy lon nhat    : {report['max_occupancy_share']}")
    print(f"  so lop nhan (train)   : {report['n_label_classes']}")
    print(f"  phan bo nhan          : {report['label_counts']}")
    print(f"  nu nho nhat           : {report['nu']}")
    print(f"  clamp saturation      : {report['clamp_saturation_fraction']}")
    print()
    print("  Tham so theo che do (da can chinh theo omega/(1-beta) tang dan):")
    print(f"    {'k':>2} {'omega':>9} {'alpha':>8} {'beta':>8} {'gamma':>8} "
          f"{'nu':>8} {'tu chuyen':>10}")
    for k in range(layout.n_states):
        print(f"    {k:>2} {params.omega[k].item():>9.4f} {params.alpha[k].item():>8.4f} "
              f"{params.beta[k].item():>8.4f} {params.gamma[k].item():>8.4f} "
              f"{nu[k].item():>8.3f} {trans[k, k].item():>10.4f}")
    print("=" * 70, flush=True)

    if args.out:
        payload = {
            "config": str(args.config),
            "elapsed_seconds": elapsed,
            "seeds": list(seeds),
            "n_sessions": len(centered),
            "n_train": n_train,
            "health": report,
            "nu_by_state": [round(v, 4) for v in nu.tolist()],
            "self_transition": [round(trans[k, k].item(), 4)
                                for k in range(layout.n_states)],
            "elbo_final": [r.elbo_trace[-1] if r.elbo_trace else None
                           for r in posterior.seed_results],
            "fallback_used": [r.fallback_used for r in posterior.seed_results],
        }
        args.out.parent.mkdir(parents=True, exist_ok=True)
        args.out.write_text(json.dumps(payload, indent=2), encoding="utf-8")
        print(f"bao cao JSON: {args.out}")

    # Ma thoat KHAC 0 khi fit khong khoe: mot lan chay nghien cuu bao "xong"
    # trong khi ket qua khong dung duoc chinh la kieu that bai am tham ma du
    # an nay cam.
    return 0 if report["healthy"] else 1


if __name__ == "__main__":
    raise SystemExit(main())
