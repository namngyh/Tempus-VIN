# Thiết kế: Nền tảng dự án + Lõi MS-EGARCH (Sub-project 1+2 / 5)

Ngày: 2026-08-17
Trạng thái: Đã duyệt bởi người dùng, chuyển sang giai đoạn viết implementation plan.

## 1. Bối cảnh

Dự án RAEMF-VB-MC (dự báo xác suất chế độ thị trường và phân phối lợi suất VN-Index)
được xây dựng lại hoàn toàn từ đầu — không có repo cũ, không có baseline
"Filtered HMM + EGARCH toàn cục" để giữ lại hay so sánh. Yêu cầu gốc (nâng cấp một
hệ thống có sẵn) được diễn giải lại thành: xây MS-EGARCH (Markov-Switching EGARCH
4 trạng thái, ước lượng toàn phần bằng Variational Bayes/ADVI) làm lõi tầng
regime + risk **ngay từ đầu**, không qua giai đoạn point-estimate trung gian.

Vì phạm vi đầy đủ (7 tầng: đặc trưng nhân quả → regime/risk → EBM → calibration →
Bayesian scenario-return → Monte Carlo → VaR/CVaR) quá lớn cho một spec, công việc
được chia thành 5 sub-project độc lập theo thứ tự:

1. **Nền tảng + Lõi MS-EGARCH** (spec này)
2. EBM classifier + temperature calibration (dùng posterior-summary của MS-EGARCH)
3. Tầng Bayesian scenario-return + Monte Carlo + VaR/CVaR/drawdown
4. Đánh giá & tài liệu (OOS benchmark harness, model card)

Quyết định đã chốt với người dùng: **không xây baseline Filtered HMM + EGARCH cũ
để so sánh** — MS-EGARCH là hệ thống mới hoàn toàn, không cần chứng minh "cải thiện
so với cái cũ" vì cái cũ chưa từng tồn tại. Benchmark OOS ở sub-project 4/5 sẽ đánh
giá nội tại (CRPS, WIS, coverage, recall theo regime...), không so sánh với baseline.

Dữ liệu nguồn duy nhất hiện có: `VNINDEX_Daily.csv` (do người dùng cung cấp),
6.307 dòng OHLCV, 2000-07-28 → 2026-07-13. Chưa dùng biến ngoại sinh (X).

## 2. Mục tiêu sub-project này

- Có một repo Python hoạt động được, có git, có test suite chạy pass.
- Dữ liệu OHLCV được ingest sạch, có kiểm chứng, không bịa số liệu.
- Hạ tầng tối ưu Variational Bayes (ADVI) dùng chung, tái sử dụng được cho các
  sub-project sau.
- Mô hình MS-EGARCH 4 trạng thái ước lượng được posterior qua ADVI trên dữ liệu
  thật, có forward filter nhân quả, có Gray's collapsing đúng, có test leakage và
  test suy biến (K=1 khớp EGARCH đơn-regime).
- Ghi lại 4 quyết định thiết kế mở (a-d nêu trong yêu cầu gốc) kèm lý do vào
  `docs/ms_egarch_design_decisions.md`.

Ngoài phạm vi sub-project này: EBM, tầng scenario-return, Monte Carlo, VaR/CVaR,
CLI/pipeline orchestration đầy đủ, benchmark OOS. Các phần này sẽ có spec riêng.

## 3. Cấu trúc dự án

Dùng trực tiếp Python 3.11 hệ thống hiện có (đã có sẵn torch 2.13+cpu, numpy,
pandas, scipy, pytest, scikit-learn, statsmodels, pyarrow, pyyaml) — không tạo
venv/conda mới để tránh cài lại torch. `ruff` cần kiểm tra/cài thêm nếu thiếu.

```
pyproject.toml
README.md
AGENTS.md
data/
  raw/VNINDEX_Daily.csv        # dữ liệu gốc, không sửa
  processed/                    # output đã làm sạch (parquet), gitignore hoặc commit tuỳ dung lượng
configs/
  cpu_smoke.yaml                 # profile debug nhanh: ít bước ADVI, 1 seed
  gpu_research.yaml              # stub cho máy có CUDA, chưa chạy
docs/
  data_ingestion_notes.md        # log các dòng bị loại khi làm sạch CSV, số liệu thật
  ms_egarch_design_decisions.md  # 4 quyết định mở (a-d), tiếng Việt trung lập
src/raemf_mc/
  __init__.py
  runtime/
    hardware.py      # select_device('auto'), hardware-report
    cpu.py            # thread/core config cho CPU-only
  data/
    ingest.py         # parser thô -> DataFrame sạch, validation gate
    loader.py          # API công khai: load_vnindex_ohlcv(path)
  features/
    returns.py          # log return, không dùng biến ngoại sinh
  validation/
    purged_split.py      # target_end_date_h < boundary, 3 horizon 20/40/60
    leakage_checks.py     # assertion dùng lại cho các module sau
  bayesian/
    torch_backend.py       # optimizer ADVI dùng chung
    priors.py                # hierarchical shrinkage priors
    diagnostics.py            # ELBO trace, seed pooling diagnostics
  regime/
    ms_egarch.py               # đệ quy Gray's collapsing, forward filter, ELBO, fit
    state_alignment.py          # căn Bull/Sideway/Bear/Stress trên train
tests/
  data/test_ingest.py
  validation/test_purged_split.py
  bayesian/test_torch_backend.py
  regime/test_ms_egarch.py
```

`cli.py`/`pipeline.py` orchestration đầy đủ hoãn sang sub-project sau.

## 4. Ingest & làm sạch dữ liệu

### 4.1 Vấn đề định dạng

`VNINDEX_Daily.csv` xuất từ nguồn nào đó có lỗi: giá trị dùng dấu phẩy ngăn cách
hàng nghìn (ví dụ `1,840.69`, `462,997,984`) bị tách thành nhiều cột riêng vì dấu
phẩy trùng ký tự phân cách CSV. Đã kiểm tra thủ công trên toàn bộ 6.307 dòng dữ
liệu (không phải suy đoán):

- Mọi dòng có đúng 13 cột raw (Date + tối đa 11 field số + cột trắng).
- Field giá trị `"1"` đứng một mình luôn là tiền tố hàng nghìn của field giá kế
  tiếp (VN-Index nằm trong khoảng 100-1900 điểm suốt lịch sử dữ liệu, nên tiền tố
  hàng nghìn luôn là `"1"`, không có trường hợp `"2"` trở lên).
- Volume luôn là field cuối cùng của mỗi dòng, có thể bị tách 1-3 lần do vượt
  hàng triệu — nối toàn bộ field còn lại sau khi đã lấy đủ 4 field giá (O,H,L,C).

### 4.2 Thuật toán reconstruct (trong `ingest.py`)

```
parts = các field không rỗng sau cột Date
for mỗi trong 4 slot [Open, High, Low, Close]:
    if parts[idx] == "1":
        rest = parts[idx+1]
        # pad phần nguyên của rest về đủ 3 chữ số trước khi nối:
        # "1"+"829.5" -> "1829.5" (no-op), "1"+"24" -> "1024"
        value = parts[idx] + pad3(phần nguyên của rest) + phần thập phân
        idx += 2
    else:
        value = parts[idx]
        idx += 1
volume = int("".join(parts[idx:]))            # nối toàn bộ field còn lại
```

Bước zero-pad là bắt buộc: nguồn xuất file **cắt bỏ số 0 đứng đầu của nhóm dư
hàng nghìn**, nên `1,024.00` xuất ra thành cặp field `"1"`, `"24"` chứ không phải
`"1"`, `"024.00"`. Xem §4.3 để biết vì sao bỏ qua bước này lại nguy hiểm.

### 4.3 Validation gate — không bịa số liệu

Sau khi parse, mỗi dòng phải thoả bất biến `Low <= Open <= High` và
`Low <= Close <= High`. Con số vi phạm thực tế: **42/6.307 dòng (~0,67%)**.

Ghi chú đính chính (phát hiện ở vòng review toàn nhánh): bản đầu tiên của tài liệu
này ghi **89 dòng** vi phạm và kết luận "khả năng cao dữ liệu nguồn bị hỏng theo
cách không thể tái tạo chắc chắn". Kết luận đó **sai với 47 trong 89 dòng**: đó
không phải lỗi nguồn mà là lỗi của chính thuật toán reconstruct. Cơ chế cụ thể:
nguồn cắt bỏ số 0 đứng đầu của nhóm dư hàng nghìn, nên dòng raw 1458
(`19/1/2007 00:00,1,24,1,24,1,22.97,1,23.05,8,751,729`) có giá trị thật là
Open/High = 1.024,00, Low = 1.022,97, Close = 1.023,05, nhưng phép nối chuỗi trần
`"1" + "24"` cho ra `124`. Sau khi thêm bước zero-pad ở §4.2, 47 dòng này được
khôi phục đúng và chỉ còn **42 dòng thực sự hỏng ở nguồn, không tái tạo được**.

Nguy hiểm của lớp lỗi này: giá trị sai ~10 lần nhưng **nhất quán nội bộ**
(cả 4 giá O/H/L/C cùng sai một hệ số), nên nó **vượt qua toàn bộ bất biến OHLC**
và trôi thẳng vào mô hình. Nó chỉ lộ ra ở tầng chuỗi thời gian: chuỗi cũ có 187
phiên với |log return| > 8% và một phiên nhảy 453% — bất khả thi khi VN-Index có
biên độ dao động ±7%/phiên. Chuỗi đã sửa chỉ còn đúng 1 phiên vượt 8%
(2025-04-03 → 2025-04-08, ~8,2%, một sự kiện thị trường có thật).

Vì vậy pipeline có thêm một cổng kiểm tra ở tầng chuỗi:
`check_implausible_daily_moves()` đếm số phiên có |log return| vượt ngưỡng 0,10
(nới từ biên độ ±7% thật để chừa biên an toàn), log cảnh báo khi có vài phiên và
raise lỗi cứng khi số phiên vượt ngưỡng đủ nhiều để chỉ ra hỏng dữ liệu có hệ
thống. Đây chính là cổng lẽ ra phải bắt được lỗi trên ngay từ đầu.

Chính sách: **loại bỏ các dòng vi phạm, không đoán/sửa giá trị.** Ghi log đầy đủ
(ngày, giá trị raw, lý do) vào `docs/data_ingestion_notes.md`. `ingest.py` raise
lỗi cứng nếu tỷ lệ dòng bị loại vượt ngưỡng bất thường (đề xuất 5%) — vì 0,67%
dưới ngưỡng này nên pipeline tiếp tục chạy nhưng luôn log rõ.

Đã phát hiện thêm: **1 dòng trùng lặp tuyệt đối** (17/12/2024, byte-identical với
dòng ngay trước) → drop bản sao, giữ một bản, log lại.

Ngày parse theo định dạng `d/m/Y H:M` (giờ luôn `00:00`, bỏ qua). Không có lỗi
parse ngày trên toàn bộ 6.307 dòng.

### 4.4 Output

`load_vnindex_ohlcv()` trả về DataFrame index theo ngày tăng dần, cột
`open, high, low, close, volume`, không NaN, không trùng ngày, đã pass validation
gate. Ghi thêm file `data_ingestion_notes.md` liệt kê chính xác các ngày bị loại
và lý do — số liệu thật lấy từ lần chạy `ingest.py` đầu tiên, không phải số ước
lượng trước.

## 5. Purged split & leakage checks

Tái dùng nguyên tắc `target_end_date_h < boundary` cho 3 horizon 20/40/60 phiên —
với mỗi outer fold, mọi hàng dùng để fit (kể cả forward filter của MS-EGARCH) phải
có ngày kết thúc target nhỏ hơn boundary của fold đó. `leakage_checks.py` cung cấp
hàm assertion dùng chung: `assert_no_target_leakage(df, boundary, horizon)`.

**Phạm vi thực tế của sub-project này (ghi rõ để không bị hiểu nhầm là đã xong):**
`purged_split.py` và `leakage_checks.py` được xây và test độc lập, nhưng **chưa hề
được ghép nối với vòng fit MS-EGARCH ở bất kỳ đâu**. Integration test
(`tests/test_integration_smoke.py`) fit trên một cửa sổ dữ liệu liền mạch, không có
purge boundary, không tách train/test, và state alignment trong test đó chạy trên
chính cửa sổ đã fit chứ không phải trên phần train riêng biệt. Nói cách khác,
chuỗi ingest → purged split → fit → alignment-chỉ-trên-train **chưa có test hay
assertion nào kiểm chứng đầu-cuối**. Việc orchestrate toàn bộ pipeline (bao gồm
walk-forward theo fold và tách train/test thật) được **hoãn sang sub-project sau**
theo đúng kế hoạch, không nằm trong phạm vi nhánh này.

## 6. Hạ tầng VB dùng chung (`bayesian/torch_backend.py`)

- Reparameterization gradient cho biến phân Gaussian mean-field (mọi tham số
  latent tham số hoá qua `(mu, log_sigma)`, sample qua `mu + sigma * eps`).
- Adam + linear warm-up + cosine decay learning rate schedule.
- Gradient clipping (norm-based).
- Early-stopping theo moving-average của ELBO (window có thể cấu hình).
- Retry ở learning rate thấp hơn khi phát hiện ELBO NaN/Inf hoặc divergence.
- Fallback về mean-field đơn giản hơn nếu retry vẫn fail — **luôn ghi log** (không
  im lặng), tương tự cơ chế `fallbacks.json` mô tả trong yêu cầu gốc.
- Multi-seed: chạy N seed độc lập, gộp posterior thành **mixture đều theo seed**
  (không chọn seed có ELBO tốt nhất) — tránh cherry-pick.
- Toàn bộ tensor ép tối thiểu `float32`; có assertion chặn cứng bất kỳ tensor
  `float16` nào xuất hiện trong đường tính ELBO/log-likelihood/logsumexp.

Module này không phụ thuộc vào MS-EGARCH cụ thể — là engine tối ưu ADVI tổng quát,
nhận vào một hàm `log_joint(theta) -> Tensor` và trả về posterior variational.

## 7. Lõi toán học MS-EGARCH (`regime/ms_egarch.py`)

### 7.1 Ngân sách tham số

- Ma trận chuyển 4×4: mỗi hàng là một 4-simplex tham số hoá qua softmax của vector
  Gaussian 3 chiều không ràng buộc (không dùng Dirichlet trực tiếp — không tương
  thích tốt với biến phân Gaussian của ADVI) → 4 hàng × 3 = **12 tham số**.
- (ω, α, β, γ) riêng từng trạng thái: 4 trạng thái × 4 tham số = **16 tham số**.
- Tổng: **28 tham số** — khớp đúng con số "~28" nêu trong yêu cầu gốc.
- **ν (bậc tự do Student-t của innovation EGARCH) là một scalar toàn cục dùng
  chung cho cả 4 trạng thái**, không tính vào 28 tham số trên. Suy luận từ chính
  ngân sách "~28" mà yêu cầu gốc đưa ra — nếu ν theo từng trạng thái thì tổng đã
  là 32, không khớp. Giữ ν toàn cục cũng tránh phình tham số trên regime Stress
  vốn có rất ít quan sát trong ~6.306 phiên.
- ν tham số hoá qua softplus + hằng số dịch để đảm bảo ν > 2 (phương sai hữu hạn).

### 7.2 Quyết định (a): Gộp ở level-space, không phải log-space

Đệ quy EGARCH tự nhiên trên log-variance, nhưng Gray's collapsing gốc là một phép
moment-matching: bảo toàn đúng `E[σ²_t | F_{t-1}] = Σ_k P(S_{t-1}=k|F_{t-1})·σ²_{k,t-1}`
theo định lý phương sai toàn phần. Gộp trực tiếp trên log-variance
(`Σ_k P_k · log σ²_{k,t-1}`) không phải một moment thật của phân phối trộn, và do
bất đẳng thức Jensen (log là hàm lõm) sẽ luôn thiên lệch xuống so với log của
trung bình thật — sai lệch càng lớn khi xác suất filtered càng phân tán giữa các
trạng thái có biến động chênh lệch nhau nhiều (đúng lúc cần phân biệt regime rõ
nhất).

Về mặt triển khai, gộp level-space **không cần exponentiate tường minh**:

```
log_var_bar[t-1] = torch.logsumexp(log_P[:, t-1] + log_var[:, k, t-1], dim=state_dim)
```

biểu thức này chính là `log(Σ_k P_k · σ²_{k,t-1})` tính theo cách ổn định số học —
vừa đúng yêu cầu bắt buộc dùng `torch.logsumexp` cho bước forward-filter (mục 3 của
yêu cầu gốc), vừa tránh phải `exp()` tường minh rồi `log()` lại (rủi ro overflow
khi log-variance lớn). Đây là lựa chọn có cơ sở lý thuyết (đúng một moment thật)
và tận dụng đúng nguyên hàm đã bắt buộc phải dùng ở nơi khác trong hệ thống.

Đánh đổi: không có tài liệu học thuật chuẩn mực nào xác nhận lựa chọn này là "đúng
duy nhất" cho trường hợp EGARCH cụ thể (Gray's paper gốc là cho GARCH thường, trên
level-space thuần). Cần test suy biến K=1 để xác nhận không có sai khác khi không
còn tính bất định giữa các trạng thái.

### 7.3 Quyết định (b): Chuẩn hoá z[t-1] bằng σ̄[t-1] đã gộp

`z[t-1] = ε[t-1] / σ̄[t-1]`, với `σ̄[t-1] = exp(0.5 · log_var_bar[t-1])` lấy từ
chính giá trị gộp ở mục 7.2. Lý do: `ε[t-1]` là một số quan sát được duy nhất (đã
thực sự xảy ra), cần một σ tham chiếu duy nhất để chuẩn hoá — và σ̄ gộp là lựa
chọn tự nhiên nhất vì đó chính là giá trị được dùng làm đầu vào cho đệ quy log-
variance của bước kế tiếp, giữ toàn bộ hệ nhất quán nội bộ (không có hai "phiên
bản" biến động khác nhau cùng tồn tại tại một thời điểm). Gray's paper gốc không
có số hạng bất đối xứng nên không có tiền lệ trực tiếp cho lựa chọn này.

### 7.4 Đệ quy đầy đủ

```
log_var_bar[t-1] = logsumexp_k( log P(S_{t-1}=k|F_{t-1}) + log_var[k,t-1] )
sigma_bar[t-1]   = exp(0.5 * log_var_bar[t-1])
z[t-1]           = eps[t-1] / sigma_bar[t-1]
log_var[k,t]     = omega_k + beta_k * log_var_bar[t-1]
                   + alpha_k * (|z[t-1]| - E|z|_nu) + gamma_k * z[t-1]
```

`E|z|_nu` là kỳ vọng trị tuyệt đối của Student-t chuẩn hoá phương sai=1, bậc tự do
ν — công thức closed-form dùng `torch.lgamma`, khả vi hoàn toàn theo ν.

## 8. Forward filter & state alignment

Hamilton filter chuẩn, toàn bộ tính trong log-domain qua `torch.logsumexp`:

```
log_pred[k,t] = logsumexp_j( log_trans[j,k] + log_filt[j,t-1] )
log_lik[k,t]  = log Student-t density(r_t; scale=exp(0.5*log_var[k,t]), df=nu)
log_filt[k,t] = log_pred[k,t] + log_lik[k,t] - logsumexp_k(...)
```

Nhân quả tuyệt đối: `log_filt[:, t]` chỉ phụ thuộc dữ liệu đến thời điểm `t`,
không bao giờ dùng smoothed/backward posterior ở bất kỳ đâu trong fit hay trong
đặc trưng xuất ra cho tầng sau.

`state_alignment.py`: sau khi fit xong trên **train only**, gán 4 trạng thái latent
thành Bull/Sideway/Bear/Stress dựa trên mean return, volatility trung bình,
duration kỳ vọng của từng trạng thái — permutation cố định áp dụng lại cho toàn bộ
dữ liệu (bao gồm cả validation/test), không refit trên test.

Trạng thái triển khai thực tế (đính chính sau review): mean return và volatility
được **chuẩn hoá z-score trên 4 trạng thái** trước khi cộng vào score. Trước đó
hai đại lượng được trừ trực tiếp cho nhau dù khác thang đo tự nhiên (mean return
cỡ 1e-3, volatility cỡ 1e-2), nên trên thực tế score bị volatility chi phối gần
như hoàn toàn. **Tiêu chí duration nêu trong đoạn trên vẫn CHƯA được triển khai**
— nó đòi hỏi truyền `MSEGARCHParams`/ma trận transition vào một hàm hiện chỉ nhận
`returns` và `log_filtered_prob`, là một thay đổi interface thật sự, nên được hoãn
có chủ ý (đã ghi chú ngay trong docstring của `align_states`).

## 9. Hierarchical shrinkage prior

Regime Stress dự kiến có rất ít quan sát trong ~6.264 phiên. Áp prior phân cấp
cho (ω, α, β, γ) của từng trạng thái: mỗi tham số trạng thái có phân phối quanh
một "hyper-mean" toàn cục dùng chung, độ co dãn (shrinkage) về hyper-mean tỷ lệ
nghịch với effective-observation-count ước lượng của trạng thái đó (đếm qua tổng
xác suất filtered của trạng thái trên tập train) — trạng thái ít quan sát bị kéo
về gần tham số trung bình toàn cục hơn, tránh overfit vào vài chục phiên hiếm gặp.

Ba điều chỉnh sau vòng review toàn nhánh (chi tiết lý do nằm trong comment code):

- **Ngưỡng effective-observation phải co giãn theo độ dài cửa sổ.** Ngưỡng tuyệt
  đối cố định (30 quan sát) không bao giờ kích hoạt trên dữ liệu thật: với 500
  (smoke) đến 6.264 (full) phiên chia cho 4 trạng thái, mọi regime đều vượt xa 30,
  nên trọng số shrinkage luôn đúng bằng 1.0 và **cơ chế phân cấp — tính năng chủ
  đạo của mục này — chưa từng hoạt động một lần nào**. Nay ngưỡng là
  `max(min_effective_observations, min_effective_fraction × T)` với
  `min_effective_fraction = 0.05`: regime chiếm dưới 5% occupancy bị co thật, ở
  mọi cỡ mẫu.
- **Cần thêm prior cho chính hyper-mean.** Riêng tầng phân cấp chỉ phạt *độ phân
  tán* giữa 4 trạng thái nên bất biến với phép tịnh tiến: cộng cùng một hằng số
  vào cả 4 giá trị β không tốn gì, tức **không có gì ràng buộc mức của β** và β > 1
  (không dừng, bùng nổ) hoàn toàn tự do. Nay mỗi họ tham số có thêm prior mức trên
  hyper-mean, trong đó β được center tại 0,9 với scale 0,15 — vùng 0,85–0,99 thực
  tế nằm gọn trong ~0,7 sd còn vùng bùng nổ bị phạt.
- **effective_obs được `.detach()`.** Trước đó gradient chảy ngược qua
  effective_obs vào hằng số chuẩn hoá của prior, biến "prior" thành hàm của θ và
  tạo động lực gradient để mô hình **bóp chết occupancy của một regime chỉ nhằm
  làm nhẹ chính phần phạt prior của regime đó**. Occupancy nay được xem là trọng
  số cố định kiểu empirical-Bayes, không thuộc mô hình sinh được lấy đạo hàm.

Hai prior "weakly-informative" ở §7 cũng được chỉnh vì thực chất chúng rất mạnh
sau khi tính đến phép biến đổi: `Normal(0,1)` trên `nu_raw` kéo theo ν gần như
không bao giờ vượt ~4,2 (ép kurtosis gần vô hạn) — nay là `Normal(5, 4)` để ν
thực tế 5–15 nằm trong tầm; `Normal(0,1)` trên transition logits đặt khối lượng
prior quanh **hàng transition đều** (p_stay = 0,25), khiến một regime "dính" thực
tế (p_stay ≈ 0,95) nằm cách ~4 sd — nay center tại p_stay = 0,85 với scale 2,0.

## 10. Quyết định (c), (d) — ghi nhận cho sub-project sau

Ghi vào `docs/ms_egarch_design_decisions.md`, chưa triển khai code trong sub-
project này (thuộc phạm vi sub-project 3):

- **(c) Bỏ `c_k`** ở tầng scenario-return tương lai: MS-EGARCH đã cung cấp σ_k,t
  riêng theo từng regime, giữ thêm `c_k` làm hệ số nhân tự do có nguy cơ non-
  identifiability (hai tham số cùng giải thích một scale, ELBO có sống núi phẳng
  dọc theo hướng `c_k · σ`, posterior mean-field có thể trông "tự tin" giả tạo ở
  từng biên trong khi thực chất không định danh được). Tầng scenario chỉ giữ μ_k
  và ν_k.
- **(d) Hai giai đoạn Bayesian tách biệt**, không một ELBO chung: fit posterior
  MS-EGARCH độc lập trước (không cần scenario/return labels), tầng scenario sau
  điều kiện theo posterior draws của MS-EGARCH. Về bản chất đây là một dạng
  modular/"cut" Bayesian inference — cắt đường phản hồi ngược từ một mô-đun có thể
  bị misspecify (tầng scenario) sang mô-đun trước nó (MS-EGARCH), một kỹ thuật có
  cơ sở thống kê chứ không chỉ vì tiện triển khai. Khớp tự nhiên với việc tách
  sub-project 2 và sub-project 3 theo quản lý phạm vi đã thống nhất.

Interface hệ quả cho sub-project này: `MSEGARCHPosterior` phải expose được API lấy
**một joint draw duy nhất** (toàn bộ tham số MS-EGARCH + state path filtered cho
một horizon) để tầng scenario sau này điều kiện theo đúng cách "một joint draw cho
mỗi Monte Carlo path, giữ cố định suốt horizon" — dù việc dùng API này thuộc sub-
project 3, chữ ký API cần chốt đúng ngay từ bây giờ để không phải sửa lại.

## 11. Kế hoạch test bắt buộc

- `test_ingest.py`: parse đúng trên các dòng mẫu đã biết trước (từ phân tích thủ
  công), drop đúng 1 dòng trùng lặp (17/12/2024), drop đúng số dòng vi phạm OHLC
  kèm log lý do, không NaN, ngày tăng dần nghiêm ngặt sau khi drop trùng.
- `test_purged_split.py`: `target_end_date_h < boundary` cho cả 3 horizon, không
  leakage qua fold.
- `test_torch_backend.py`: gradient chảy qua reparameterization, early-stopping
  kích hoạt đúng, fallback/retry được ghi log (không im lặng), multi-seed pooling
  gộp đều (không thiên vị seed ELBO tốt nhất), assertion chặn `float16`.
- `test_ms_egarch.py`:
  - Trường hợp suy biến K=1 (ép toàn bộ khối lượng xác suất về một trạng thái):
    đệ quy Gray's collapsing phải khớp chính xác với đệ quy EGARCH đơn-regime
    thông thường (implement một hàm EGARCH đơn-regime tham chiếu tối giản trong
    test để so sánh).
  - Forward filter tại thời điểm `t` chỉ phụ thuộc dữ liệu đến `t` — test kiểu
    `test_posterior_uses_only_train_rows`: fit trên `[0, T]` và trên `[0, T+k]`
    rồi cắt, log-filtered-prob tại các thời điểm `<= T` phải giống hệt nhau.
  - Simplex constraint: mọi hàng ma trận chuyển và mọi `log_filt[:, t]` sau
    `exp()` phải sum về 1 (trong sai số số học).
  - Finite-value check: không NaN/Inf ở bất kỳ draw posterior nào trên toàn bộ
    chuỗi log_var.
  - API lấy một joint draw duy nhất trả về đúng shape, giữ cố định giá trị tham
    số suốt horizon khi dùng lại nhiều lần (không resample ngầm).

Toàn bộ test suite phải pass qua `python -m pytest -q` và
`python -m ruff check src tests` trước khi coi sub-project này hoàn thành.

## 12. Rủi ro và giới hạn đã biết

- 1.4% dữ liệu bị loại do lỗi định dạng nguồn — đã ghi log, không ảnh hưởng lớn
  đến ~6.306 phiên nhưng cần nêu rõ trong mọi báo cáo sau này.
- ADVI backprop qua đệ quy tuần tự trên toàn bộ ~6.306 phiên có thể chậm trên CPU
  — `cpu_smoke.yaml` dùng cửa sổ dữ liệu ngắn hơn và ít bước ADVI để lặp nhanh khi
  debug; chưa đo thời gian huấn luyện thật cho tới khi có bản chạy đầu tiên.
  Không bịa số liệu thời gian chạy trước khi đo thật.
- Quyết định (a) và (b) chưa có tài liệu học thuật chuẩn mực xác nhận — nêu rõ
  trong `ms_egarch_design_decisions.md` là lựa chọn có lý giải riêng, không phải
  kết luận đã được kiểm chứng bên ngoài.
