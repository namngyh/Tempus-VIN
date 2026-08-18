# Thiết kế: Đánh giá OOS + Model Card (Sub-project 4 / 4)

Ngày: 2026-08-18
Trạng thái: Đã duyệt bởi người dùng (thiết kế trong chat), chuyển sang giai đoạn viết implementation plan.

## 1. Bối cảnh

Sub-project 1 (MS-EGARCH), 2 (EBM classifier), 3 (scenario-return + Monte
Carlo + VaR/CVaR) đã hoàn tất — tất cả đều xác nhận "chạy đúng, không lỗi,
số liệu hợp lý về mặt số học", nhưng **chưa sub-project nào đo được mô hình
có thực sự đáng tin hay không**. Cụ thể còn thiếu:

- **Sub-project 2's whole-branch review** đã ghi nhận rõ (Important #7,
  chưa fix): macro F1, recall riêng Bear/Stress, NLL trước/sau calibration
  chưa từng được tính thành một module tái dùng được — chỉ có số liệu rời
  rạc trong một test.
- **Sub-project 1 và 3** chưa có benchmark out-of-sample (OOS) nào —
  MS-EGARCH's forecast σ_t và tầng Monte Carlo VaR/CVaR chưa từng được đối
  chiếu với dữ liệu thực tế xảy ra sau đó.
- Quyết định đã chốt từ sub-project 1 (`docs/ms_egarch_design_decisions.md`):
  không so sánh với baseline cũ (không tồn tại) — benchmark OOS đánh giá
  **nội tại** (CRPS, WIS, coverage, recall theo regime), không so sánh
  tương đối.

Quyết định đã chốt trong buổi brainstorm hôm nay:

- Gộp phần "hoàn thành đánh giá EBM còn thiếu" (Phần A) và phần "đánh giá
  xác suất cho scenario-return/MC" (Phần B) vào **một** sub-project — cả
  hai đều là "đo mô hình đã có", không xây kiến trúc mô hình mới.
- Walk-forward OOS (Phần B) ở quy mô smoke dùng **5-10 mốc lịch sử** — đủ
  để xác nhận logic đúng, không nhắm số liệu thống kê cuối cùng.
- Chỉ viết code + test smoke-scale + tài liệu (README, model card) ở đây;
  chạy số liệu thật (nhiều mốc walk-forward hơn, ADVI hội tụ tốt hơn) chờ
  máy có GPU.

## 2. Mục tiêu sub-project này

- **Phần A**: module đánh giá phân loại tái dùng được (macro F1, recall
  riêng Bear/Stress, confusion matrix, NLL trước/sau calibration), áp dụng
  lại lên pipeline EBM đã có của sub-project 2 (không đổi gì upstream).
- **Phần B**: harness walk-forward OOS cho tầng scenario-return/MC — tại
  mỗi mốc lịch sử, fit lại MS-EGARCH + μ_k trên dữ liệu tới mốc đó, mô
  phỏng Monte Carlo H-ngày tới, đối chiếu với dữ liệu **thực tế đã biết**
  sau mốc đó bằng CRPS, WIS, và kiểm định Kupiec (coverage của VaR).
- **Model card**: tài liệu mô tả mô hình theo chuẩn (mục đích dùng, dữ
  liệu, kiến trúc, kết quả đánh giá — ghi rõ smoke-scale, giới hạn đã biết
  xuyên suốt dự án).
- Test đầy đủ: từng công thức (F1/recall/CRPS/WIS/Kupiec) đối chiếu tính
  tay trên ví dụ nhỏ đã biết trước kết quả; integration test chạy thật
  trên dữ liệu VN-Index ở quy mô smoke.

Ngoài phạm vi: chạy walk-forward quy mô đầy đủ (nhiều chục+ mốc, cần cho
Kupiec test có ý nghĩa thống kê thật) — hoãn sang máy GPU; so sánh với
baseline (không tồn tại, theo quyết định gốc); CLI/pipeline orchestration
đầy đủ (đã hoãn từ sub-project 1).

## 3. Cấu trúc dự án

```
src/raemf_mc/evaluation/
  classification_metrics.py   # Phần A
  probabilistic_metrics.py    # Phần B: CRPS, WIS
  var_backtest.py             # Phần B: Kupiec test
  walk_forward.py             # Phần B: điều phối walk-forward
configs/
  eval_smoke.yaml             # cấu hình ADVI RẤT nhẹ riêng cho walk-forward
                               # (N lần refit -> phải nhẹ hơn hẳn các smoke
                               # config trước, xem mục 6)
docs/
  model_card.md
tests/
  evaluation/test_classification_metrics.py
  evaluation/test_probabilistic_metrics.py
  evaluation/test_var_backtest.py
  test_evaluation_integration_smoke.py   # chạy thật trên dữ liệu thật
```

## 4. Phần A — Đánh giá phân loại (`evaluation/classification_metrics.py`)

Tái dùng trực tiếp `sklearn.metrics` (đã có sẵn trong dependencies, đã
kiểm chứng qua cộng đồng — không cần viết lại F1/recall/confusion matrix
từ đầu, chỉ viết wrapper mỏng xử lý đúng quy ước `model.classes_` alphabet
đã biết từ sub-project 2):

- `compute_classification_report(y_true, y_pred, labels=STATE_NAMES) -> dict`:
  bọc `sklearn.metrics.classification_report` + `confusion_matrix`, trả về
  macro F1, recall/precision từng lớp (đặc biệt highlight Bear, Stress —
  mục tiêu chính theo tài liệu gốc), confusion matrix dạng `pd.DataFrame`
  có tên hàng/cột rõ ràng.
- `compute_nll(probs, y_true_idx) -> float`: `-mean(log(probs[range(n),
  y_true_idx]))` — công thức đã dùng ad-hoc trong integration test của
  sub-project 2, nay tách thành hàm dùng chung, dùng lại
  `apply_temperature_log_prob` khi có sẵn log-prob thay vì `log(prob)` để
  tránh lỗi underflow đã sửa ở sub-project 2.

Không đổi gì trong `models/ebm_classifier.py` hay `models/calibration.py`
(sub-project 2, đã merge) — chỉ tiêu thụ output của chúng.

## 5. Phần B — Walk-forward OOS cho scenario-return/MC

### 5.1 Thiết kế walk-forward

```
def run_walk_forward_evaluation(
    ohlcv: pd.DataFrame,
    cutoff_fractions: list[float],   # vd [0.5, 0.6, 0.7, 0.8, 0.9] của window
    horizon: int,
    ms_egarch_advi: AdviConfig, ms_egarch_prior: HierarchicalPriorConfig,
    mu_advi: AdviConfig, mu_n_draws: int, mu_prior_scale: float,
    mu_min_effective_observations: float, mu_min_effective_fraction: float,
    mc_n_paths: int, layout, seeds, device,
    var_alphas: tuple[float,...] = (0.95, 0.99),
) -> pd.DataFrame   # 1 hàng / mốc: crps, wis, realized_return, var_95,
                     # var_99, breached_95 (bool), breached_99 (bool)
```

Tại mỗi mốc `cutoff` (một vị trí index trong `ohlcv`, đủ xa cuối để còn
`horizon` phiên thực tế sau đó để đối chiếu):

1. `fit_window = ohlcv.iloc[:cutoff]` — fit MS-EGARCH + μ_k trên đúng dữ
   liệu **tới mốc đó** (không nhìn thấy tương lai — đây chính là tính chất
   "out-of-sample" thật, khác với sub-project 3's integration test vốn fit
   trên toàn bộ cửa sổ).
2. Mô phỏng Monte Carlo `horizon` ngày tới từ mốc đó (tái dùng nguyên vẹn
   `simulate_mc_paths` từ sub-project 3).
3. `realized_return = log_returns.iloc[cutoff:cutoff+horizon].sum()` — giá
   trị **đã biết thật**, lấy từ dữ liệu lịch sử (không phải tương lai thật,
   nhưng out-of-sample so với bước fit).
4. Tính CRPS, WIS giữa phân phối mô phỏng (mục 5.2) và `realized_return`.
5. Tính VaR_95/99 từ phân phối mô phỏng, so với `realized_return` để xác
   định có "vi phạm" (breach) hay không — tích luỹ qua mọi mốc cho Kupiec
   test (mục 5.3).

**Lưu ý nhân quả về centering**: mỗi mốc phải center bằng đúng mean của
`fit_window` (không phải mean toàn bộ dữ liệu) — cùng kỷ luật leakage đã
áp dụng xuyên suốt dự án.

**Lưu ý quan trọng đã rút ra từ lỗi thật ở sub-project 3**: `simulate_mc_paths`
trả về return ở không gian ĐÃ CENTER (theo đúng hợp đồng đã ghi trong
docstring của nó), còn `realized_return` lấy trực tiếp từ dữ liệu thô
(chưa center). Trước khi tính CRPS/WIS/breach, phải cộng lại đúng mean của
`fit_window` vào phân phối mô phỏng (hoặc trừ mean đó khỏi `realized_return`
— chọn một chiều, nhất quán) để cả hai ở cùng thang đo — nếu không, mọi số
liệu Phần B sẽ lệch hệ thống đúng như lỗi Important #2 đã tìm thấy và sửa
ở review toàn nhánh sub-project 3.

### 5.2 CRPS (`evaluation/probabilistic_metrics.py`)

Từ N draw Monte Carlo `x_1..x_M` (tổng return tích luỹ tại horizon H) và
giá trị thực `y`:

```
CRPS(x, y) = mean(|x_i - y|) - 0.5 * mean(|x_i - x_j|)   (trung bình trên mọi cặp i,j)
```

Công thức dạng đóng chính xác của CRPS cho phân phối kinh nghiệm (empirical
distribution) — không phải xấp xỉ. Với M=500 draw, số hạng đôi là 250.000
phép tính, không đáng kể về hiệu năng nên dùng công thức trực tiếp này
(rõ ràng, dễ verify tính tay) thay vì công thức O(M log M) qua sắp xếp.

### 5.3 WIS (`evaluation/probabilistic_metrics.py`)

Dùng bộ mức tin cậy chuẩn trong tài liệu dự báo xác suất (Bracher et al.
2021, không liên quan tới alpha của VaR — đây là interval **hai phía**,
VaR là **một phía**): `alpha_levels = (0.5, 0.2, 0.1, 0.05, 0.02)` (khoảng
tin cậy 50/80/90/95/98%). Với mỗi mức `alpha`, khoảng `[l, u]` lấy từ
`quantile(alpha/2)` và `quantile(1-alpha/2)` của phân phối mô phỏng:

```
IS_alpha(l, u, y) = (u - l) + (2/alpha)*(l-y)*[y<l] + (2/alpha)*(y-u)*[y>u]
WIS(y) = (1/(K+0.5)) * (0.5*|y - median| + sum_k (alpha_k/2) * IS_alpha_k(l_k, u_k, y))
```

với K=5 (số mức alpha).

### 5.4 Kupiec test (`evaluation/var_backtest.py`)

Với `n` mốc walk-forward, `x` = số mốc mà `realized_return < -VaR_alpha`
(vi phạm), `p = 1 - alpha` (xác suất vi phạm kỳ vọng nếu VaR đúng):

```
LR_uc = -2*[ (n-x)*log(1-p) + x*log(p) - (n-x)*log(1-x/n) - x*log(x/n) ]
```

Xử lý biên `x=0` hoặc `x=n` (log(0)) bằng cách bỏ số hạng tương ứng thay vì
để lỗi toán học. So `LR_uc` với ngưỡng chi-squared(1) ở mức 95% (3.841) —
`LR_uc` vượt ngưỡng nghĩa là tỷ lệ vi phạm thực tế khác biệt có ý nghĩa so
với kỳ vọng.

**Giới hạn phải ghi rõ**: với `n=5-10` mốc smoke-scale, kiểm định Kupiec
gần như không có power thống kê thật (cần vài chục mốc trở lên) — ở quy mô
này, test chỉ xác nhận **công thức tính đúng**, không phải kết luận thống
kê đáng tin về khả năng hiệu chỉnh (calibration) của VaR.

## 6. Cấu hình ADVI riêng cho walk-forward (`configs/eval_smoke.yaml`)

Walk-forward nghĩa là fit lại MS-EGARCH + μ_k **N lần** (một lần mỗi mốc).
Dùng cấu hình ADVI như các smoke config trước (300 bước) sẽ khiến 5-10 mốc
mất hàng giờ. `eval_smoke.yaml` dùng cấu hình **nhẹ hơn hẳn** riêng cho
mục đích này — ví dụ ~50-100 bước ADVI, cửa sổ fit ngắn hơn (vài trăm
phiên thay vì 1500) — đủ để verify pipeline chạy đúng đầu-cuối trong thời
gian hợp lý (~30-40 phút cho toàn bộ 5-10 mốc), không nhắm chất lượng fit
thống kê.

## 7. Model card (`docs/model_card.md`)

Theo cấu trúc chuẩn (Model Cards for Model Reporting):

- **Mục đích sử dụng**: nghiên cứu/hỗ trợ quyết định về regime & rủi ro
  VN-Index; KHÔNG phải hệ thống giao dịch tự động, KHÔNG đã qua kiểm định
  production.
- **Dữ liệu**: nguồn (`VNINDEX_Daily.csv`), phạm vi thời gian, tần suất
  (ngày), các bước làm sạch đã áp dụng.
- **Kiến trúc**: tóm tắt 3 tầng (MS-EGARCH → EBM → scenario-return/MC/risk),
  liên kết tới các design doc chi tiết.
- **Kết quả đánh giá**: số liệu Phần A/B — ghi rõ **SMOKE-SCALE, KHÔNG PHẢI
  KẾT QUẢ NGHIÊN CỨU CUỐI CÙNG**, kèm điều kiện cấu hình đã dùng.
- **Giới hạn đã biết**: tổng hợp toàn bộ giới hạn đã ghi nhận xuyên suốt 4
  sub-project (không walk-forward thật, một số draw ADVI không stationary,
  nhãn regime có thể suy biến ở config nhanh, chưa kiểm chứng GPU, v.v.).

## 8. Test bắt buộc

- `test_classification_metrics.py`: đối chiếu tính tay macro F1/recall
  trên ví dụ nhỏ có nhãn/dự đoán biết trước; xác nhận NLL tính đúng bằng
  log-domain (không dùng `log(prob)` trực tiếp).
- `test_probabilistic_metrics.py`: CRPS đối chiếu công thức đóng trên phân
  phối rời rạc nhỏ (vài điểm, tính tay được); WIS đối chiếu công thức
  interval score trên ví dụ có `y` bên trong/ngoài khoảng tin cậy.
- `test_var_backtest.py`: Kupiec LR đối chiếu tính tay trên bộ (n, x, p)
  nhỏ đã biết trước kết quả; xác nhận xử lý đúng biên `x=0`/`x=n`.
- `test_evaluation_integration_smoke.py`: chạy thật Phần A (trên pipeline
  EBM thật) và Phần B (walk-forward 5-10 mốc trên dữ liệu VN-Index thật,
  dùng `eval_smoke.yaml`) — assert số liệu hợp lý (CRPS/WIS ≥ 0, LR_uc hữu
  hạn, breach count ∈ [0, n]), in ra bảng kết quả rõ ràng.

## 9. Rủi ro và giới hạn đã biết

- Walk-forward smoke-scale (5-10 mốc, ADVI rất nhẹ) chỉ verify logic đúng
  — không phải benchmark thống kê đáng tin, đặc biệt Kupiec test gần như
  không có power ở n này.
- CRPS/WIS phụ thuộc trực tiếp vào chất lượng fit MS-EGARCH/μ_k tại từng
  mốc — nếu ADVI dưới hội tụ (như đã quan sát ở sub-project 2/3 với config
  nhanh), số liệu Phần B sẽ phản ánh đúng một fit chưa tốt, không phải lỗi
  công thức.
- Phần A đánh giá lại đúng test set đã dùng ở sub-project 2 — không phải
  dữ liệu mới, chỉ là hoàn thiện phép đo đã bị hoãn.
- Chưa kiểm chứng trên GPU — `eval_smoke.yaml` tiếp tục tinh thần CPU-only
  đã thiết lập xuyên suốt dự án.
