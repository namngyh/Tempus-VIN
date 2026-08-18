# Thiết kế: Tầng Bayesian scenario-return + Monte Carlo + VaR/CVaR/drawdown (Sub-project 3 / 4)

Ngày: 2026-08-18
Trạng thái: Đã duyệt bởi người dùng (thiết kế trong chat), chuyển sang giai đoạn viết implementation plan.

## 1. Bối cảnh

Sub-project 1 (MS-EGARCH, đã merge) và Sub-project 2 (EBM classifier, đã merge)
đã xong. MS-EGARCH cung cấp `PooledPosterior` đầy đủ của (ω, α, β, γ theo từng
trạng thái, ma trận chuyển, ν chung), cùng `sample_ms_egarch_draw` trả về đúng
MỘT `MSEGARCHParams` mỗi lần gọi, và `run_ms_egarch_recursion` (forward filter
nhân quả, Gray's collapsing) trả về `log_filtered_prob[t,k]`, `log_var[t,k]`
(phương sai riêng từng trạng thái, chưa gộp), `log_var_bar[t]` (đã gộp).

Quyết định đã chốt sẵn từ sub-project 1 (`docs/ms_egarch_design_decisions.md`,
mục (c), (d)) và xác nhận lại trong buổi brainstorm hôm nay:

- **Không có `c_k`** (hệ số nhân biến động riêng theo regime) — MS-EGARCH đã
  cho σ_k,t riêng theo từng trạng thái, thêm `c_k` sẽ non-identifiable.
- **`ν` dùng chung** một giá trị đã fit từ MS-EGARCH, không tách `ν_k` riêng
  theo regime — tránh thêm tham số chưa có bằng chứng cần thiết.
- **`μ_k`** (drift trung bình riêng theo từng regime) là tham số MỚI, cần ước
  lượng trong sub-project này bằng một fit Bayesian nhẹ (ADVI), **điều kiện
  theo posterior draws thật của MS-EGARCH** (không phải một điểm ước lượng cố
  định) — đúng tinh thần modular/"cut" Bayesian đã chốt ở quyết định (d): fit
  MS-EGARCH không đổi, tầng scenario chỉ điều kiện một chiều theo nó.
- **Horizon**: 1 ngày và 20 ngày (giao dịch). **Mức tin cậy**: 95% và 99%.
- **Quy mô Monte Carlo smoke (CPU)**: 500-1000 path, đủ để kiểm chứng logic
  đúng, không nhắm số liệu VaR chính xác cuối cùng — số liệu chính thức chờ
  chạy full-scale trên máy có GPU sau (giống tinh thần `ebm_smoke.yaml`).

## 2. Mục tiêu sub-project này

- Fit `μ_k` (k=1..4, một theo mỗi trạng thái) bằng ADVI, điều kiện theo N draw
  thật từ MS-EGARCH posterior — có prior yếu, có shrinkage cho trạng thái thưa
  dữ liệu (nhất quán với cơ chế đã xây cho ω/α/β/γ ở sub-project 1).
- Mô phỏng Monte Carlo tiến (forward simulation) từ "hôm nay": mỗi path rút
  một draw MS-EGARCH + một draw μ độc lập, giữ cố định suốt path, mô phỏng
  trạng thái **thực** (sampled, không phải kỳ vọng) và return hàng ngày qua
  20 ngày.
- Tính VaR, CVaR, phân phối max drawdown ở 2 horizon (1, 20 ngày) × 2 mức tin
  cậy (95%, 99%) từ tập hợp path mô phỏng.
- Test đầy đủ: công thức VaR/CVaR/drawdown đối chiếu tính tay trên ví dụ tổng
  hợp nhỏ; fit μ_k kiểm chứng qua trường hợp suy biến (một draw MS-EGARCH duy
  nhất, không nhiễu) khớp một fit thông thường; integration test chạy thật
  trên dữ liệu VN-Index.

Ngoài phạm vi: walk-forward re-fit MS-EGARCH nhiều fold (đã hoãn từ sub-project
2), benchmark OOS chính thức (CRPS/WIS/coverage — thuộc sub-project 4), CLI/
pipeline orchestration đầy đủ, báo cáo README với số liệu cuối cùng (số liệu
smoke-scale hiện tại chỉ để kiểm chứng logic, không phải kết quả nghiên cứu).

## 3. Cấu trúc dự án

```
src/raemf_mc/
  scenario/
    mu_fit.py       # fit mu_k qua ADVI, điều kiện theo N draw MS-EGARCH thật
    simulate.py      # mô phỏng Monte Carlo tiến (state + return path)
  risk/
    metrics.py        # VaR, CVaR, max drawdown từ tập path mô phỏng
tests/
  scenario/test_mu_fit.py
  scenario/test_simulate.py
  risk/test_metrics.py
  test_scenario_mc_integration_smoke.py   # chạy thật trên dữ liệu thật
configs/
  mc_smoke.yaml       # profile debug nhanh (cửa sổ ngắn, ít path/bước ADVI)
```

## 4. Fit μ_k (`scenario/mu_fit.py`)

### 4.1 Vì sao không cần recursion mới

`run_ms_egarch_recursion` giả định return đã center về 0 (`student_t_log_pdf_with_variance(returns[t], torch.zeros_like(variance_t), variance_t, nu)`
— loc cố định = 0). Thêm `μ_k` **không** đòi hỏi sửa hay chạy lại recursion
này: recursion đã trả sẵn `log_filtered_prob[t,k]` VÀ `log_var[t,k]` (phương
sai riêng từng trạng thái, chưa gộp) cho mọi `t`. Với MỘT draw MS-EGARCH cố
định, log-likelihood của `μ` (4 giá trị) là hàm đóng, khả vi trực tiếp trên
hai tensor đã có sẵn đó:

```
ll(mu | draw_i) = sum_t logsumexp_k [
    log_filtered_prob_i[t,k] + student_t_log_pdf_with_variance(
        r_t, mu_k, exp(log_var_i[t,k]), nu_i
    )
]
```

Đây chính là bước "gộp theo state" mà recursion gốc đã làm ở mỗi bước filter
(`log_joint_t = log_pred + loglik_t`), chỉ khác là giờ `loglik_t` có `loc=mu_k`
thay vì `loc=0`. `log_filtered_prob_i` giữ nguyên không đổi — đúng tinh thần
"điều kiện một chiều, không lan ngược" của quyết định (d): μ không ảnh hưởng
ngược lại việc MS-EGARCH đã filter trạng thái như thế nào.

### 4.2 Điều kiện theo N draw thật — trung bình đúng cách (tránh thiên lệch Jensen)

Giống lý do đã ghi ở quyết định (a) (gộp level-space, không phải log-space, để
tránh thiên lệch do bất đẳng thức Jensen), xấp xỉ Monte Carlo của marginal
likelihood `p(r | mu) = E_theta~posterior[p(r | mu, theta)]` phải lấy trung
bình trên KHÔNG GIAN LIKELIHOOD (level), không phải trung bình các log-
likelihood:

```
log p(r | mu) ≈ logsumexp_i( ll(mu | draw_i) ) - log(N)
```

**không phải** `mean_i(ll(mu | draw_i))` (trung bình log trực tiếp là trung
bình nhân của likelihood, luôn thiên lệch xuống so với giá trị đúng).

### 4.3 Tối ưu chi phí: precompute N draw MỘT LẦN, ngoài vòng lặp ADVI

`(log_filtered_prob_i, log_var_i, nu_i)` với `i=1..N` **không phụ thuộc vào
μ** — chạy `sample_ms_egarch_draw` + `run_ms_egarch_recursion` N lần (mặc
định N=50, giống sub-project 2) **một lần duy nhất, trước khi vào vòng lặp
ADVI**, xếp chồng thành tensor `(N, T, 4)`, `detach()` khỏi đồ thị tính đạo
hàm. `log_joint_fn(mu)` bên trong vòng lặp ADVI chỉ còn là phép tính vector
hoá rẻ trên các tensor cố định này — không chạy lại recursion mỗi bước
gradient. Nếu chạy lại N recursion mỗi bước ADVI (300-2000+ bước), chi phí sẽ
tăng gấp hàng trăm/nghìn lần một cách không cần thiết.

### 4.4 Prior cho μ_k — shrinkage cho trạng thái thưa dữ liệu

Trạng thái Stress (hiếm gặp) có nguy cơ μ_Stress được ước lượng từ rất ít
"effective observations" (đo bằng `sum_t filtered_prob[t, Stress]`, trung
bình qua N draw) — cùng rủi ro đã dẫn tới cơ chế hierarchical shrinkage cho
ω/α/β/γ ở sub-project 1. Tái dùng trực tiếp `state_shrinkage_weight` và
`hierarchical_normal_log_prob` đã có sẵn trong `bayesian/priors.py` (hai hàm
này đã tổng quát, không đặc thù cho EGARCH — nhận tensor/float, không đòi hỏi
`HierarchicalPriorConfig` cụ thể):

```python
def build_mu_log_joint(
    centered_returns: torch.Tensor,
    ms_egarch_posterior: PooledPosterior,
    layout: MSEGARCHParamLayout = MSEGARCHParamLayout(),
    n_draws: int = 50,
    mu_prior_scale: float = 0.01,
    min_effective_observations: float = 30.0,
    generator: torch.Generator | None = None,
) -> Callable[[torch.Tensor], torch.Tensor]:
    """theta ở đây LÀ mu (4 giá trị) trực tiếp — không cần unpack/transform,
    mu không bị ràng buộc dấu hay khoảng giá trị."""
```

`mu_prior_scale=0.01` (thang log-return hàng ngày, ~1%) là prior yếu quanh 0
cho hyper-mean ngầm định 0 — không cần hyper-mean riêng như ω/α/β/γ vì không
có rủi ro non-identifiability tương tự (không có tham số thứ hai nào cùng
giải thích scale của μ).

### 4.5 Fit và sample

Tái dùng NGUYÊN VẸN hạ tầng ADVI tổng quát đã có (`fit_multi_seed_advi`,
`sample_joint_draw` trong `bayesian/torch_backend.py` — cả hai đã tổng quát
trên `theta: Tensor`, không đặc thù MS-EGARCH). Không cần viết lại engine,
không cần hàm `sample_mu_draw` riêng — `sample_joint_draw(mu_posterior,
generator)` đã đúng ngữ nghĩa "một draw μ độc lập mỗi lần gọi".

Hàm tiện ích chính: `fit_regime_mu(centered_returns, ms_egarch_posterior,
advi_config, seeds, device, layout=..., n_draws=50, mu_prior_scale=0.01,
min_effective_observations=30.0, generator=None, fallback_log_path=...) ->
PooledPosterior` — gói toàn bộ 4.1-4.4 lại, gọi `fit_multi_seed_advi` với
`init_mu=torch.zeros(4)`, `init_log_sigma` hằng số nhỏ.

## 5. Mô phỏng Monte Carlo (`scenario/simulate.py`)

### 5.1 Điểm xuất phát — trạng thái "hôm nay"

Mỗi Monte Carlo path rút riêng một draw MS-EGARCH (`sample_ms_egarch_draw`) —
**độc lập** với N=50 draw dùng để fit μ ở mục 4 (batch khác, mục đích khác).
Với draw đó, chạy `run_ms_egarch_recursion` một lần trên lịch sử để lấy:
phân phối trạng thái filtered tại phiên cuối (`exp(log_filtered_prob[-1,:])`)
làm phân phối xuất phát; `log_var_bar[-1]` và `z_prev = returns[-1] /
exp(0.5*log_var_bar[-1])` làm điểm khởi động cho đệ quy EGARCH tiếp diễn.

### 5.2 Mô phỏng tiến H ngày (H = max horizon = 20)

Khác với forward filter (trộn xác suất mềm qua mọi trạng thái), mô phỏng
Monte Carlo cần trạng thái **thực** — một trạng thái cụ thể mỗi ngày, rút từ
phân phối categorical:

```
state_0 ~ Categorical(filtered_prob tại phiên cuối lịch sử)
for h in 1..H:
    state_h ~ Categorical(transition_matrix[state_{h-1}, :])
    raw_log_var_h = omega[state_h] + beta[state_h]*log_var_bar_prev
                    + alpha[state_h]*(|z_prev| - E|z|) + gamma[state_h]*z_prev
    log_var_h = clamp(raw_log_var_h, -30, 30)   # cùng ngưỡng đã dùng ở sub-project 1
    eps_h ~ standardized Student-t(nu)            # nu chung, đã fit
    r_h = mu[state_h] + exp(0.5*log_var_h) * eps_h
    z_prev = eps_h        # đã chuẩn hoá theo chính sigma vừa dùng để sinh r_h
    log_var_bar_prev = log_var_h
```

Ghi chú: ở bước mô phỏng, `z_prev` cập nhật trực tiếp bằng `eps_h` (đã chuẩn
hoá theo xây dựng), khác bước filter (nơi `z_prev = returns[t]/sigma_bar_t`
phải chia tường minh vì return quan sát được chưa biết trước sigma nào sinh
ra nó) — cùng một bất biến toán học, chỉ khác chiều tính (sinh vs suy ra).

Hàm chính: `simulate_mc_paths(ms_egarch_posterior, mu_posterior,
centered_returns, init_log_var, init_log_state_prob, n_paths, horizon=20,
layout=..., generator=None) -> np.ndarray` — trả về mảng `(n_paths, horizon)`
log-return hàng ngày mô phỏng. Cắt theo horizon cụ thể (1 hay 20 ngày) và
tổng hợp rủi ro là việc của `risk/metrics.py`, không phải của simulator —
tách trách nhiệm, mô phỏng một lần dùng được cho mọi horizon ≤ 20.

## 6. VaR / CVaR / Max Drawdown (`risk/metrics.py`)

Từ mảng `(n_paths, horizon)`:

- **Cumulative return tại horizon H**: `R_H = sum(daily_returns[:, :H], axis=1)`.
- **VaR_α** (quy ước số dương = mức lỗ): `VaR_alpha = -quantile(R_H, 1-alpha)`.
  Ví dụ α=0.95 → phân vị 5% của `R_H`, đổi dấu.
- **CVaR_α** (Expected Shortfall): `CVaR_alpha = -mean(R_H[R_H <= quantile(R_H, 1-alpha)])`
  — trung bình phần đuôi vượt ngưỡng VaR.
- **Max drawdown mỗi path**: `P_h = exp(cumsum(daily_returns[:, :H]))` (giá
  trị tương đối, `P_0=1`); `drawdown_h = (running_max(P)_h - P_h) /
  running_max(P)_h`; `max_drawdown = max_h(drawdown_h)`. Báo cáo phân phối
  (mean/median/95th percentile) qua các path, không chỉ một số.

Hàm chính: `compute_var(horizon_returns, alpha) -> float`,
`compute_cvar(horizon_returns, alpha) -> float`,
`compute_max_drawdown(daily_returns_2d) -> np.ndarray` (một giá trị mỗi
path), `summarize_risk(daily_paths, horizons=(1,20), alphas=(0.95,0.99)) ->
pd.DataFrame` (bảng tổng hợp: hàng = horizon, cột = VaR/CVaR mỗi mức tin cậy
+ thống kê max drawdown).

## 7. Test bắt buộc

- `test_mu_fit.py`: (i) trường hợp suy biến — N=1 draw MS-EGARCH không nhiễu,
  fit μ khớp một fit ADVI thông thường trên đúng log-likelihood đó (kiểm
  chứng công thức 4.1 đúng khi không có trung bình qua nhiều draw); (ii) xác
  nhận công thức 4.2 dùng `logsumexp - log(N)` chứ không phải `mean` trực
  tiếp (test số trên ví dụ nhỏ có thể tính tay, giống cách đã kiểm chứng
  quyết định (a) ở sub-project 1); (iii) shrinkage prior giảm đúng theo
  effective_observations thấp (tái dùng test pattern đã có cho ω/α/β/γ).
- `test_simulate.py`: (i) trường hợp suy biến K=1 (chỉ một trạng thái, xác
  suất chuyển = 1) khớp một mô phỏng EGARCH đơn-regime thông thường; (ii)
  reproducibility với generator cố định; (iii) shape đúng `(n_paths,
  horizon)`; (iv) trạng thái mô phỏng luôn hợp lệ (0..3).
- `test_metrics.py`: VaR/CVaR/max drawdown đối chiếu tính tay trên một mảng
  return tổng hợp nhỏ đã biết trước kết quả chính xác (không chỉ kiểm tra
  "chạy được" hay dấu).
- `test_scenario_mc_integration_smoke.py`: chạy thật trên dữ liệu VN-Index
  qua `configs/mc_smoke.yaml` — fit MS-EGARCH (tái dùng hạ tầng sub-project
  1), fit μ_k, mô phỏng 500-1000 path, tính VaR/CVaR/drawdown, assert số liệu
  hợp lý (VaR_99 ≥ VaR_95 ≥ 0, CVaR_alpha ≥ VaR_alpha ở cùng alpha, drawdown
  ∈ [0,1], v.v.) — không chỉ "chạy được", theo đúng kỷ luật Task 18/I12.

## 8. Rủi ro và giới hạn đã biết

- Số liệu VaR/CVaR ở quy mô smoke (500-1000 path, cấu hình ADVI nhanh) chỉ để
  kiểm chứng logic đúng — KHÔNG phải số liệu nghiên cứu cuối cùng, giống hạn
  chế đã ghi nhận ở sub-project 2 (nhãn regime có thể suy biến dưới config
  ADVI nhanh, ảnh hưởng trực tiếp tới μ_k và mô phỏng ở đây).
- Chưa kiểm chứng trên máy GPU thật — `mc_smoke.yaml` tiếp tục tinh thần
  CPU-only đã thiết lập.
- μ_k giả định drift hằng số trong mỗi regime suốt horizon mô phỏng (không mô
  hình hoá drift thay đổi trong ngày hay theo thời gian trong regime) — đơn
  giản hoá có chủ đích, nhất quán với việc bỏ `c_k` (quyết định c): tối thiểu
  hoá số tham số scenario-layer khi chưa có bằng chứng cần thêm.
- Chi phí tính: mỗi trong 500-1000 Monte Carlo path chạy riêng một lần
  `run_ms_egarch_recursion` trên toàn bộ lịch sử để xác định điểm xuất phát
  (mục 5.1) — cùng mẫu hình đã dùng ở sub-project 2 (N=50 draw) nhưng quy mô
  lớn hơn 10-20 lần. Ở smoke-scale (`window_sessions` ngắn) vẫn khả thi trên
  CPU; ở full-scale cần đánh giá lại thời gian chạy thật trước khi coi đây là
  thiết kế cuối cùng cho production — nếu quá chậm, hướng tối ưu tự nhiên là
  vector hoá batch nhiều draw cùng lúc thay vì vòng lặp Python tuần tự, cùng
  hướng tối ưu GPU đã ghi nhận là hoãn lại ở sub-project 2.
- Benchmark OOS chính thức (CRPS, WIS, coverage của VaR thực nghiệm) thuộc
  sub-project 4, chưa nằm trong phạm vi này.
