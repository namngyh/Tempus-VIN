# Thiết kế: EBM classifier + temperature calibration (Sub-project 2 / 5)

Ngày: 2026-08-17
Trạng thái: Đã duyệt bởi người dùng, chuyển sang giai đoạn viết implementation plan.

## 1. Bối cảnh

Sub-project 1 (đã hoàn thành, merge vào `master`, push lên
`https://github.com/namngyh/Tempus-VIN.git`) xây xong: ingest dữ liệu VN-Index
thật (6.264 phiên sạch sau khi sửa lỗi parser), hạ tầng Variational Bayes (ADVI)
dùng chung, và lõi MS-EGARCH 4 trạng thái — cho ra posterior đầy đủ của (ω, α, β,
γ, ma trận chuyển, ν) theo từng regime, cùng forward filter nhân quả sinh
`log_filtered_prob[t,k]` và `log_var_bar[t]` (log-variance đã gộp theo Gray's
collapsing) cho mọi phiên.

Sub-project 2 xây EBM (Explainable Boosting Machine, từ package `interpret` đã
có sẵn trong môi trường — `ExplainableBoostingClassifier`) làm tầng classifier
4 lớp (Bull/Sideway/Bear/Stress), cộng temperature calibration. Theo đúng yêu
cầu gốc: EBM **giữ nguyên point-estimate**, không đưa vào phạm vi Bayesian hóa.

Quyết định đã chốt với người dùng trong buổi brainstorm:

- **Giá trị gia tăng của EBM**: thêm đặc trưng nhân quả từ OHLCV thô (momentum,
  volume, drawdown) mà MS-EGARCH thuần biến động không thấy được — nhắm cải
  thiện tín hiệu nhận diện Bear/Stress so với chỉ dùng riêng biến động.
- **Không dùng xác suất regime của MS-EGARCH làm feature đầu vào cho EBM** (nếu
  dùng, EBM chỉ học lại xấp xỉ chính label của nó — vô nghĩa). Chỉ dùng posterior
  summary của **biến động** (σ_t) làm feature từ MS-EGARCH.
- **Phần cứng**: máy hiện tại chỉ có Intel Iris Xe tích hợp, không có GPU
  NVIDIA/CUDA (đã kiểm tra qua `Win32_VideoController`, không suy đoán). Người
  dùng sẽ chuyển sang máy có GPU NVIDIA sau. Tiếp tục phát triển trên CPU,
  không xây walk-forward re-fit đầy đủ (chi phí ADVI quá lớn trên CPU) — dùng
  một lần fit MS-EGARCH trên train, ghi rõ giới hạn thống kê.

## 2. Mục tiêu sub-project này

- Đặc trưng nhân quả mới từ OHLCV (momentum, realized volatility, volume,
  drawdown) — module riêng, không phụ thuộc MS-EGARCH.
- Đặc trưng posterior summary biến động từ MS-EGARCH (posterior-mean σ_t,
  posterior sd σ_t) sinh nhân quả cho toàn bộ chuỗi từ một lần fit trên train.
- Nhãn mục tiêu (target) 4 lớp từ posterior mean point-estimate của MS-EGARCH
  qua `align_states`/`apply_alignment` đã có sẵn.
- EBM 4 lớp (point-estimate, `ExplainableBoostingClassifier`), fit trên train,
  temperature calibration fit trên validation, đánh giá trung thực trên test
  giữ riêng (không tuning trên test).
- Test đầy đủ: đặc trưng nhân quả không leak tương lai, EBM/calibration không
  đọc test khi fit, integration test chạy thật trên dữ liệu thật.

Ngoài phạm vi: walk-forward re-fit MS-EGARCH nhiều fold, tầng scenario-return,
Monte Carlo, benchmark OOS chính thức (thuộc sub-project 3/4).

## 3. Cấu trúc dự án

```
src/raemf_mc/
  features/
    causal.py            # đặc trưng nhân quả mới từ OHLCV (module mới)
  regime/
    posterior_features.py # posterior-mean/sd sigma_t từ MS-EGARCH (module mới)
  models/
    ebm_classifier.py     # wrapper ExplainableBoostingClassifier + fit/predict
    calibration.py        # temperature scaling
tests/
  features/test_causal.py
  regime/test_posterior_features.py
  models/test_ebm_classifier.py
  models/test_calibration.py
  test_ebm_integration_smoke.py   # chạy thật trên dữ liệu thật, tương tự Task 18 cũ
configs/
  ebm_smoke.yaml          # profile debug nhanh cho EBM (cửa sổ ngắn, ít bước ADVI)
```

## 4. Đặc trưng nhân quả mới (`features/causal.py`)

Tất cả tính bằng rolling window kết thúc tại đúng phiên `t` (pandas `.rolling()`
mặc định đã nhân quả — không cần shift thêm). Mọi hàm nhận `pd.DataFrame` đã
index theo ngày tăng dần (đầu ra của `load_vnindex_ohlcv`).

| Đặc trưng | Công thức | Lý do |
|---|---|---|
| `ret_1`, `ret_5`, `ret_20` | `log(close_t / close_{t-h})`, h=1/5/20 | Momentum nhiều khung thời gian — tín hiệu xu hướng mà biến động thuần không có. |
| `realized_vol_5`, `realized_vol_20` | Std của `ret_1` trên cửa sổ rolling 5/20 phiên | Biến động thực tế đo trực tiếp từ giá, đối chiếu chéo với σ_t của MS-EGARCH — không thay thế mà bổ sung góc nhìn khác. |
| `volume_change_1` | `log(volume_t / volume_{t-1})` | Đột biến khối lượng thường đi trước biến động regime. |
| `relative_volume_20` | `volume_t / rolling_mean(volume, 20)` | Khối lượng bất thường so với nền gần đây — tín hiệu cảnh báo sớm Stress mà một mô hình chỉ nhìn giá không thấy. |
| `high_low_range` | `(high_t - low_t) / close_t` | Biên độ dao động trong phiên, bổ sung cho biến động close-to-close. |
| `drawdown_from_high_60` | `(close_t - rolling_max(close, 60)) / rolling_max(close, 60)` | Khoảng cách tới đỉnh gần nhất — đặc trưng trực tiếp cho Bear/Stress, độc lập với biến động tức thời. |

Cửa sổ dài nhất là 60 phiên → 60 dòng đầu của chuỗi có `NaN` (không đủ lịch
sử), bị loại bỏ tường minh (`dropna()`), ghi rõ số dòng bị loại trong log —
theo đúng kỷ luật "không bịa số liệu", không impute.

Hàm chính: `compute_causal_features(ohlcv: pd.DataFrame) -> pd.DataFrame` trả về
DataFrame index theo ngày (đã bỏ warmup), cột là 7 đặc trưng trên.

## 5. Đặc trưng posterior biến động từ MS-EGARCH (`regime/posterior_features.py`)

`run_ms_egarch_recursion` đã trả về `log_var_bar[t]` — chính là log-variance kỳ
vọng tại thời điểm `t` gộp theo xác suất regime filtered (Gray's collapsing).
Đây là định nghĩa tự nhiên của "posterior-mean σ_t" theo đúng yêu cầu gốc, không
cần định nghĩa thêm khái niệm mới.

Thuật toán: lấy N draw độc lập từ `PooledPosterior` (mặc định N=50, có seed cố
định để tái lập được), với MỖI draw chạy `run_ms_egarch_recursion` một lần trên
TOÀN BỘ chuỗi return (train+val+test nối liền — xem lý do nhân quả ở mục 6),
thu `sigma_bar_t = exp(0.5 * log_var_bar[t])` cho mọi `t`. Sau N draw, tính
`posterior_mean_sigma[t]` và `posterior_sd_sigma[t]` (mean/std qua N draw tại
mỗi `t`) — đúng hai đặc trưng "posterior-mean σ_t" và "posterior sd" (đặc trưng
bất định) mà yêu cầu gốc đã chỉ định.

Hàm chính: `compute_posterior_volatility_features(posterior, returns, init_log_var, init_log_state_prob, n_draws=50, generator=None) -> pd.DataFrame` — trả về 2 cột theo cùng index thời gian với `returns`.

## 6. Nhãn mục tiêu và tính nhân quả — không walk-forward, có ghi giới hạn

**Nhãn (target)**: khác với đặc trưng biến động (lấy trung bình/độ lệch qua N
draw ngẫu nhiên), nhãn cần một chuỗi **xác định, tái lập được**. Dùng
`canonical_theta = trung bình các vector mu của mọi seed trong pooled posterior`
(điểm ước lượng trung tâm, không nhiễu) — chạy `run_ms_egarch_recursion` đúng
một lần với `canonical_theta`, lấy `log_filtered_prob`, qua `align_states`/
`apply_alignment` (đã có sẵn từ sub-project 1), rồi `argmax` mỗi phiên ra nhãn
cứng 1-trong-4 (Bull/Sideway/Bear/Stress). Lý do tách biệt: nhãn cần một chuỗi
sự thật duy nhất để huấn luyện/đánh giá; đặc trưng bất định thì cần phản ánh
đúng độ trải rộng của posterior — dùng chung một cơ chế cho cả hai sẽ làm mất
một trong hai mục đích.

**Tính nhân quả — không leak vào validation/test**: MS-EGARCH được fit bằng
ADVI **chỉ trên phần train** (theo split ở mục 7). Bước center return
(`returns - returns.mean()`, theo đúng pattern đã dùng ở Task 18 sub-project 1)
PHẢI dùng **mean của train only**, rồi trừ đúng giá trị mean cố định đó cho
toàn bộ chuỗi (kể cả val/test) trước khi chạy recursion — nếu dùng mean của
toàn bộ train+val+test để center, bước center đó tự nó đã là một dạng leakage
(mean "nhìn thấy" val/test). Sau khi fit xong,
`canonical_theta` (và các draw dùng cho đặc trưng biến động) là CỐ ĐỊNH — không
tối ưu lại. Chạy forward filter (nhân quả, đã kiểm chứng ở sub-project 1: fit
trên `[0,T]` và trên `[0,T+k]` rồi cắt cho kết quả giống hệt nhau tại
`[0,T]`) trên **toàn bộ chuỗi train+val+test nối liền** với `θ` cố định này.
Vì `θ` không đổi và forward filter chỉ dùng dữ liệu quá khứ tại mỗi bước, đặc
trưng/nhãn tại mọi phiên val/test **không rò rỉ** — `θ` chưa từng "nhìn thấy"
dữ liệu val/test khi được fit.

**Giới hạn cần ghi rõ (không phải walk-forward thật)**: `θ` được fit bằng ADVI
trên TOÀN BỘ cửa sổ train cùng lúc (tối ưu hoá theo batch), nên với một phiên
`t` SỚM trong train, `θ` vẫn mang thông tin từ các phiên MUỘN HƠN trong cùng
cửa sổ train (thông qua batch likelihood). Đây là thiên lệch nhẹ trong phạm vi
train, không phải leakage ra val/test. Walk-forward thật (refit theo từng fold
mở rộng) sẽ loại bỏ hoàn toàn thiên lệch này nhưng cần refit ADVI nhiều lần —
hoãn sang khi có máy GPU thật, ghi chú rõ trong tài liệu, không đánh giá EBM
trên phần train bị ảnh hưởng (accuracy/calibration chỉ báo cáo trên val/test).

## 7. Train/val/test split

Nowcasting (phân loại trạng thái *hiện tại*, không dự báo tương lai), khác với
tầng scenario-return dự báo h-bước — nên không cần `target_end_date_h <
boundary` như `purged_split.py`. Chỉ cần tách theo thứ tự thời gian, không xáo
trộn: **70% train / 15% validation (dùng để fit temperature) / 15% test (giữ
riêng, chỉ đánh giá cuối cùng)**, theo vị trí index sau khi đã bỏ warmup 60
dòng đầu của đặc trưng nhân quả. Tỷ lệ và ranh giới ghi cụ thể (ngày bắt đầu/
kết thúc mỗi phần) vào log khi chạy, không hard-code số dòng cụ thể trong tài
liệu này.

## 8. EBM (`models/ebm_classifier.py`)

Wrapper mỏng quanh `ExplainableBoostingClassifier` (package `interpret`, đã có
sẵn trong môi trường — đã kiểm tra `interpret.glassbox`). Input: ma trận đặc
trưng nối 7 cột từ `causal.py` + 2 cột từ `posterior_features.py` = 9 cột.
`random_state` cố định để tái lập. Không tự ý thêm `interactions` cao (giữ mặc
định thấp, tránh overfit trên ~4-5 nghìn dòng train). Hàm chính:
`fit_ebm_classifier(X_train, y_train, **kwargs) -> ExplainableBoostingClassifier`,
`predict_scores(model, X) -> np.ndarray` (dùng `model.decision_function(X)`,
KHÔNG dùng `predict_proba` trực tiếp — cần raw score cho bước calibration).

Lưu ý quan trọng đã kiểm chứng bằng code thật: `model.classes_` sắp xếp theo
alphabet (`['Bear','Bull','Sideway','Stress']`), KHÁC thứ tự `STATE_NAMES`
(`Bull/Sideway/Bear/Stress`). Mọi chỗ ánh xạ điểm số/xác suất về tên lớp phải
dùng `model.classes_`, không được giả định thứ tự `STATE_NAMES`.

## 9. Temperature calibration (`models/calibration.py`)

Temperature scaling chuẩn cho multiclass: `calibrated_proba = softmax(scores / T)`,
với `scores = model.decision_function(X)` (điểm log-odds thô theo từng lớp,
đã xác nhận `ExplainableBoostingClassifier` có sẵn method này). Học một scalar
`T > 0` duy nhất bằng cách tối thiểu hoá negative log-likelihood trên tập
validation — cài bằng PyTorch (nhất quán với phần còn lại của codebase): `T`
là một tham số `torch.nn.Parameter` (khởi tạo 1.0, tham số hoá qua
`softplus` để đảm bảo dương), tối ưu bằng Adam vài chục bước trên NLL của
validation set, KHÔNG dùng test set.

Hàm chính: `fit_temperature(scores_val: Tensor, y_val: Tensor, n_steps=100) -> float`,
`apply_temperature(scores: Tensor, temperature: float) -> Tensor` (trả về xác
suất đã calibrate qua softmax).

## 10. Đánh giá trung thực (test set)

Trên test set (chưa từng dùng để fit EBM hay temperature): accuracy tổng,
macro F1, recall riêng từng lớp Bear/Stress (mục tiêu chính theo tài liệu gốc),
NLL trước/sau calibration (để xác nhận calibration có cải thiện, không giả
định). Ghi số liệu thật, không so sánh với baseline cũ (không tồn tại, theo
đúng quyết định đã chốt ở sub-project 1) — chỉ báo cáo nội tại.

## 11. Kế hoạch test bắt buộc

- `test_causal.py`: từng công thức đối chiếu tính tay trên chuỗi giá tổng hợp
  nhỏ; xác nhận đúng số dòng warmup bị bỏ (60); xác nhận không NaN sau warmup.
- `test_posterior_features.py`: posterior-mean/sd tính đúng qua N draw đã biết
  trước (dùng posterior giả lập nhỏ); xác nhận nhân quả (giống bài test prefix-
  invariant của sub-project 1: cắt chuỗi sớm hơn phải cho kết quả giống hệt ở
  phần chung).
- `test_ebm_classifier.py`: fit/predict cơ bản trên dữ liệu tổng hợp nhỏ; xác
  nhận `classes_` được xử lý đúng thứ tự (không giả định `STATE_NAMES`).
- `test_calibration.py`: temperature scaling giảm NLL trên một tập validation
  tổng hợp có calibration lệch rõ ràng (tương tự cách kiểm chứng công thức
  shrinkage prior ở sub-project 1 — dùng ví dụ có thể tính tay).
- `test_ebm_integration_smoke.py`: chạy thật trên dữ liệu thật, dùng
  `configs/ebm_smoke.yaml` (cửa sổ ngắn hơn để lặp nhanh khi debug) — fit
  MS-EGARCH trên train, sinh đặc trưng, fit EBM, calibrate, đánh giá trên test,
  assert các con số thật hợp lý (không chỉ "chạy được"), tương tự kỷ luật đã
  áp dụng ở Task 18/I12 của sub-project 1.

## 12. Rủi ro và giới hạn đã biết

- Thiên lệch nhẹ trong train do fit MS-EGARCH theo batch, không phải walk-
  forward thật — đã ghi rõ ở mục 6, chỉ báo cáo metric trên val/test.
- Chưa kiểm chứng trên máy GPU thật — cấu hình `ebm_smoke.yaml` tiếp tục theo
  tinh thần CPU-only của sub-project 1, `gpu_research.yaml` cũ vẫn là stub
  chưa chạy.
- Chưa có benchmark OOS chính thức hay so sánh baseline — thuộc sub-project 4.
