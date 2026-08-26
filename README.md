# RAEMF-VB-MC

Mô hình dự báo xác suất chế độ thị trường (Bull/Sideway/Bear/Stress) và phân phối
lợi suất VN-Index, dùng MS-EGARCH (Markov-Switching EGARCH 4 trạng thái) ước lượng
toàn phần bằng Variational Bayes (ADVI), lớp phân loại EBM, tầng scenario-return +
Monte Carlo, và risk metrics (VaR/CVaR/drawdown).

Xem `docs/superpowers/specs/` cho thiết kế chi tiết từng sub-project và
`docs/ms_egarch_design_decisions.md` cho các quyết định thiết kế còn mở.

## Cài đặt

Máy phát triển hiện tại: Windows 11, Python 3.13, **NVIDIA RTX 4060 8 GB
(sm_89, driver CUDA 13.2)**. Dùng venv riêng thay vì Python hệ thống, và
**bắt buộc cài bản torch CUDA** — bản mặc định trên PyPI là CPU-only:

```
python -m venv .venv
.venv/Scripts/python -m pip install --index-url https://download.pytorch.org/whl/cu130 torch==2.13.0+cu130
.venv/Scripts/python -m pip install numpy pandas scipy pyyaml scikit-learn statsmodels pyarrow pytest ruff interpret
.venv/Scripts/python -m pip install -e . --no-deps
```

`--no-deps` ở bước cuối là cố ý: `pip install -e ".[dev]"` sẽ kéo `torch`
từ PyPI và ghi đè bản CUDA vừa cài bằng bản CPU.

**Lưu ý về worktree**: một venv dùng chung cho nhiều worktree thì editable
install chỉ trỏ tới MỘT worktree tại một thời điểm. Chuyển worktree phải
chạy lại `pip install -e . --no-deps` từ worktree đó, nếu không sẽ chạy
nhầm code của nhánh khác mà không có dấu hiệu gì.

## Chạy test

```
python -m pytest -q
python -m ruff check src tests scripts
```

Lưu ý: full suite bao gồm nhiều integration test chạy ADVI/Monte Carlo THẬT trên
dữ liệu VN-Index thật, không phải mock — mỗi test có thể mất 15-90 phút. Xem
mục "Trạng thái hiện tại" dưới đây để biết chính xác test nào thuộc branch nào.

---

## Kiến trúc (4 sub-project)

| # | Tên | Mục đích | Trạng thái |
|---|---|---|---|
| 1 | Nền tảng + Lõi MS-EGARCH | Ingest dữ liệu, hạ tầng ADVI dùng chung, mô hình MS-EGARCH 4 trạng thái | **Xong, đã merge `master`, đã push GitHub** |
| 2 | EBM classifier + calibration | Phân loại regime từ đặc trưng OHLCV nhân quả + posterior sigma_t của MS-EGARCH, temperature calibration | **Xong, đã merge `master`, đã push GitHub** |
| 3 | Scenario-return + Monte Carlo + VaR/CVaR | Fit mu_k theo regime, mô phỏng Monte Carlo tiến, tính VaR/CVaR/drawdown | **Xong, đã merge `main`** (chưa push — xem mục dưới) |
| 4 | Đánh giá OOS + Model card | Hoàn thiện đánh giá EBM (macro F1, recall Bear/Stress, NLL), walk-forward OOS (CRPS/WIS/Kupiec), model card | **Xong** — xem `docs/model_card.md` |

---

## Kết quả (2026-08-24)

Cả 4 sub-project đã xong và merge vào `main`. **Đọc `docs/model_card.md`
trước khi diễn giải bất kỳ con số nào dưới đây** — đặc biệt mục 4, nơi liệt
kê những gì mô hình này CHƯA chứng minh được.

### Fit MS-EGARCH quy mô nghiên cứu

`configs/gpu_research.yaml`: 6 263 phiên (train 4 384), 8 seed song song,
1 500 bước ADVI, `n_mc_samples=16`, CPU. Thời gian thật **9,5 giờ**
(71 phút/seed).

| | |
|---|---|
| `regime_health_report` | **healthy: True** |
| Occupancy 4 chế độ | 0,3868 · 0,2078 · 0,0997 · 0,3056 |
| ELBO cuối, 8 seed | 13 175,85 … 13 215,78 (chênh 0,3%) |
| Seed rơi vào fallback | 0/8 |
| `nu` theo chế độ | 3,246 · 3,061 · 2,909 · 3,122 |

Trước đó, cấu hình smoke cũ cho occupancy `[0,827 · 0,074 · 0,072 · 0,028]`
và nhãn suy biến hoàn toàn về một lớp (1 049/1 049 phiên).

### Hiệu năng

Một bước ADVI, T=1 500, `n_mc=16`, trên RTX 4060 (WDDM) + CPU 16 nhân:

| Đường | s/bước |
|---|---|
| CUDA eager | 9,618 |
| CPU | 2,172 |
| **CUDA graph** | **0,204** |

Toàn bộ khoảng cách GPU/CPU của dự án này là overhead phát kernel (~42 µs
eager so với ~1,5 µs graph), không phải tính toán. Chi tiết:
`docs/perf_batching_notes.md`.

### Ba điều phải biết trước khi dùng số liệu

1. **Walk-forward OOS chưa có sức mạnh thống kê.** 8 mốc, p=0,05 → số vi
   phạm kỳ vọng 0,4. Kiểm định Kupiec ở cỡ mẫu này xác nhận công thức chạy
   đúng, không xác nhận mô hình hiệu chỉnh đúng.
2. **Nhãn chế độ đã đổi nghĩa** — "chế độ nào hoạt động bất thường so với
   mức thường của nó", không phải "chế độ nào khả năng cao nhất".
3. **VaR/CVaR chưa đáng tin.** `nu ≈ 3` nên kurtosis vẫn vô hạn (cần
   `nu > 4`).

### Việc còn lại, không nằm trong 4 sub-project

Lưu/tải mô hình, CLI orchestration, đường nạp dữ liệu sống. Ba mảnh này bắt
buộc để vận hành thật và chưa mảnh nào tồn tại — xem `docs/model_card.md`
mục 4.8.

---

## Rủi ro / giới hạn đã biết (tổng hợp xuyên suốt dự án)

- **~~CPU-only, chưa kiểm chứng GPU~~ — ĐÃ GIẢI QUYẾT (2026-08-20)**: đường
  GPU đã được chạy thật trên RTX 4060 và đã có test (`tests/runtime/test_cuda_device_path.py`).
  Một bug thật đã được tìm ra và sửa (`torch.randint` thiếu `device=`). Bài
  học: thread `device` qua từng lời gọi cấp phát là cần nhưng **chưa đủ** —
  các hàm nhận `generator` mà tự cấp phát tensor bên trong là một lớp lỗi
  riêng mà review tĩnh trên máy CPU-only không thể bắt. Xem "Chuyện GPU" ở
  trên về việc `auto` là lựa chọn SAI cho tầng ADVI.
- **Không walk-forward re-fit thật** ở sub-project 1/2/3 — MS-EGARCH được fit
  MỘT LẦN trên toàn bộ cửa sổ, không refit theo từng fold như một walk-forward
  sản xuất thật sự cần. Sub-project 4 có xây walk-forward NHƯNG chỉ ở quy mô
  smoke (5-10 mốc, ADVI rất nhẹ) — không phải benchmark thống kê đáng tin.
- **Nhãn regime suy biến — đã khoanh vùng và xử lý một phần (2026-08-20)**:
  triệu chứng nặng hơn ghi chép cũ (1049/1049 phiên cùng một nhãn, không phải
  >95%). Bốn giả thuyết đã bị **thí nghiệm bác bỏ**: (a) không phải do vector
  hoá — nhánh gốc cũng hỏng cùng bệnh; (b) không sửa được bằng cách chạy ADVI
  lâu hơn — `n_steps`=1200 suy biến ngang 300; (c) không phải `beta` không
  dừng; (d) `nu` sát sàn là nguyên nhân THẬT nhưng THỨ YẾU — siết `nu` về ~6
  xoá được ca đơn-lớp nhưng nhãn vẫn bị một lớp chiếm 95%+. Thủ phạm chính là
  `argmax`, đã xử lý (xem "Nhãn regime đã đổi nghĩa" ở trên).
  **Còn tồn tại**: `nu` vẫn rơi xuống sàn một cách hệ thống (2,15–2,21 trên cả
  ba seed) — chưa siết prior vì đó là quyết định mô hình cần đi qua spec. Và
  lớp `Bull` chỉ có 2 mẫu train, 0 mẫu val/test: **taxonomy 4 chế độ vẫn chưa
  được dữ liệu chống đỡ**, thực chất đang là bài toán ba lớp.
- **Một phần posterior MS-EGARCH nằm ngoài vùng ổn định (`|beta|<1`)**: prior
  chỉ là ràng buộc mềm (Normal(0.9, 0.15)), không phải constraint cứng — đã
  xác nhận thật qua debug sub-project 3 (37% draw/state trong một fit thử
  nghiệm không stationary). Tầng scenario-return đã xử lý an toàn (không "nổ"
  số), nhưng đây vẫn là dấu hiệu fit chưa hội tụ tốt, không phải đã giải quyết
  tận gốc (giải pháp dùng constraint cứng như `beta=tanh(beta_raw)` thuộc
  phạm vi sub-project 1, chưa làm).
  **Đính chính (2026-08-20)**: ràng buộc cứng này KHÔNG phải chìa khoá cho
  VaR/CVaR phi lý. Một fit đo trực tiếp trên cửa sổ `ebm_smoke` có
  `beta` = 0,717 / 0,703 / 0,745 / 0,744 — **toàn bộ nằm trong vùng dừng** —
  và `clamp_saturation_fraction` = 0. Thủ phạm là `nu` = 2,206 sát sàn 2,05
  (Student-t kurtosis vô hạn khi nu < 4). Hai vấn đề khác nhau, không cùng gốc.
- **Chưa có lưu/tải model (persistence)**: mỗi lần chạy đều fit lại từ đầu,
  không serialize posterior ra đĩa. **Đây là thứ chặn việc vận hành thật**:
  với mô hình đã fit sẵn, một lần cập nhật hằng ngày chỉ tốn ~4 giây (bộ lọc
  0,52 s + đặc trưng posterior ~0,6 s + Monte Carlo 500 path 2,49 s), nhưng
  vì không tách được "fit" khỏi "predict" nên thực tế mỗi lần chạy tối thiểu
  ~9 phút. Cùng với CLI và đường nạp dữ liệu sống, đây là ba mảnh hạ tầng
  bắt buộc để vận hành mà **không mảnh nào nằm trong 4 sub-project hiện tại**.
- **Chưa có CLI/pipeline orchestration**: chỉ chạy được qua test script, chưa
  có lệnh đơn lẻ kiểu `raemf-mc predict --date today`.
- **Chỉ tần suất ngày (daily)**: không có đặc trưng/mô hình intraday, dù dữ
  liệu 5 phút có sẵn cho VNINDEX — xem ghi chú trong lịch sử brainstorm
  (`docs/superpowers/specs/`) về lý do (regime kinh tế không vận hành ở tần
  số 5 phút, và cần xử lý seasonality trong ngày riêng nếu mở rộng sau này).
- **Không so sánh baseline**: quyết định chủ đích từ sub-project 1 — MS-EGARCH
  là hệ thống mới hoàn toàn, benchmark chỉ đánh giá nội tại (CRPS/WIS/coverage),
  không so sánh "tốt hơn mô hình cũ" vì mô hình cũ chưa tồn tại.

## Cấu trúc thư mục chính

```
src/raemf_mc/
  data/            # ingest + làm sạch VNINDEX_Daily.csv (sub-project 1)
  features/        # log returns, đặc trưng nhân quả OHLCV (sub-project 1, 2)
  bayesian/        # hạ tầng ADVI dùng chung + prior (sub-project 1)
  regime/          # MS-EGARCH, state alignment, posterior features (sub-project 1, 2)
  models/          # EBM classifier + calibration (sub-project 2)
  scenario/        # fit mu_k + mô phỏng Monte Carlo (sub-project 3, trên branch scenario-mc-var)
  risk/            # VaR/CVaR/drawdown (sub-project 3, trên branch scenario-mc-var)
  evaluation/      # metric phân loại, CRPS/WIS, Kupiec, walk-forward OOS (sub-project 4)
  validation/      # split, leakage checks (dùng chung)
configs/           # các profile ADVI (cpu_smoke, ebm_smoke, mc_smoke, gpu_research-stub)
docs/
  superpowers/specs/    # thiết kế chi tiết từng sub-project
  superpowers/plans/    # implementation plan từng sub-project
  perf_batching_notes.md # số đo hiệu năng CPU/GPU thật + điểm giao + việc chưa làm
  sdd-records/          # ledger + báo cáo debug của sub-project 3
  model_card.md         # ĐỌC TRƯỚC khi diễn giải bất kỳ số liệu nào
scripts/
  run_research_fit.py   # fit quy mô nghiên cứu + báo cáo sức khoẻ, --parallel
results/                # báo cáo JSON của các lần chạy nghiên cứu
```
