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
| 4 | Đánh giá OOS + Model card | Hoàn thiện đánh giá EBM (macro F1, recall Bear/Stress, NLL), walk-forward OOS (CRPS/WIS/Kupiec), model card | **Mới bắt đầu, chưa có code — xem mục dưới** |

---

## Trạng thái hiện tại (2026-08-20) — đang trên máy GPU

Máy GPU đã sẵn sàng và **sub-project 3 đã được merge vào `main`**. Full suite:
**130 passed, 0 failed** (49 phút, gồm cả ba integration test chạy thật trên
dữ liệu VN-Index). **Chưa push lên GitHub** — `main` cục bộ đang đi trước
`origin/main`.

### Việc đã làm trên máy GPU (2026-08-19 → 20)

**1. Vector hoá theo chiều batch.** Đo lần đầu trên GPU thật cho kết quả
ngược với kỳ vọng: CUDA **chậm hơn** CPU 5,6× ở recursion và 4,9× ở ADVI, vì
gần như toàn bộ chi phí là 1500 vòng lặp Python trên tensor 4 phần tử chứ
không phải phép tính. Đã bỏ chiều batch dẫn đầu vào recursion / ADVI /
Monte Carlo / `mu_fit`. Kết quả đo được:

| | trước | sau |
|---|---|---|
| ADVI 20 bước × 4 MC sample (T=1500, CPU) | 126,5 s | **34,9 s** (3,6×) |
| `simulate_mc_paths` 500 path (CPU) | ~275 s | **2,5 s** (~110×) |
| integration test sub-project 3 | ~90 phút | **11 phút 36** |

Và `n_mc_samples` từ 4 lên 16 chỉ tốn thêm **3%** — chất lượng gradient ELBO
giờ gần như miễn phí. Chi tiết đầy đủ: `docs/perf_batching_notes.md`.

**2. Sửa một bug đường CUDA thật.** `sample_joint_draw` gọi `torch.randint`
thiếu `device=`, nên với generator CUDA nó raise `Expected a 'cpu' device
type for generator but found 'cuda'` — chắn ngang toàn bộ tầng Monte Carlo
trên GPU. Repo trước đó **không có một test nào chạm GPU**; giờ có 4
(`tests/runtime/test_cuda_device_path.py`, tự skip trên máy không CUDA).

**3. Chẩn đoán fit regime suy biến** (`regime_health_report`). Đo được: mọi
chẩn đoán cũ đều báo sạch (`fallback_used=False`, `clamp_saturation=0`, ELBO
bình thường) trong khi nhãn regime là 1049/1049 phiên cùng một lớp và
`nu`=2,15–2,21 sát sàn 2,05. Đó đúng nghĩa là silent fallback mà `AGENTS.md`
cấm.

**4. Đổi scheme gán nhãn regime** từ `argmax` thô sang chuẩn hoá theo tần
suất nền. Xem mục "Nhãn regime đã đổi nghĩa" dưới đây — đây là thay đổi
quan trọng nhất cần biết khi đọc bất kỳ kết quả nào.

### Nhãn regime đã ĐỔI NGHĨA

Nhãn không còn trả lời *"hôm nay chế độ nào khả năng cao nhất"* mà trả lời
*"hôm nay chế độ nào đang hoạt động BẤT THƯỜNG so với mức thường của nó"*.
Mọi metric tính trên nhãn này — macro F1, recall Bear/Stress, NLL trước/sau
calibration — phải được đọc theo nghĩa thứ hai. **Phải ghi vào model card.**

Lý do: với một fit có occupancy `[0,724, 0,202, 0,052, 0,022]`, chế độ thứ
hai giữ 20,2% khối lượng trung bình nhưng chỉ được gán nhãn ở **47/1049**
phiên — nó hầu như không bao giờ *vượt* chế độ dẫn đầu ở bất kỳ ngày cụ thể
nào, nên `argmax` thô vứt bỏ toàn bộ biến thiên theo thời gian mà bộ lọc
thật sự có. Ablation 4 seed × 2 cấu hình (chạy cả hai scheme trên cùng
`log_filtered_prob` đã fit): `argmax` cho mục tiêu **đơn lớp ở 5/8** trường
hợp, `base_rate` **0/8**. Chi tiết và cả tiêu chí mà `base_rate` THUA nằm
trong docstring của `label_indices_from_filtered`.

Trên dữ liệu thật, `test_ebm_integration_smoke` đi từ `{Sideway: 1049}` sang
`{Sideway: 508, Stress: 279, Bear: 218, Bull: 2}`.

### Sub-project 4 (branch `evaluation-model-card`)

Vẫn chỉ có spec (`docs/superpowers/specs/2026-08-18-evaluation-model-card-design.md`)
và plan 8-task (`docs/superpowers/plans/2026-08-18-evaluation-model-card.md`)
— **chưa có dòng code nào**. Phụ thuộc chéo branch đã được gỡ: `fit_regime_mu`,
`simulate_mc_paths`, `compute_var` giờ đã có trên `main`, nên chỉ cần merge
`main` vào worktree đó rồi làm tuần tự từ Task 1.

**Ba điều chỉnh thiết kế nên đưa vào trước khi bắt đầu**, rút ra từ số đo
của phiên này:

1. **Task 4 (walk-forward) nên chạy song song theo tiến trình, không phải
   batch.** Các mốc có độ dài cửa sổ khác nhau nên không gộp được vào chiều
   batch, nhưng chúng độc lập hoàn toàn và máy này có 16 nhân. Đã kiểm chứng
   thật: 9 fit song song mất 24 phút thay vì ~107 phút tuần tự (~4,4×).
2. **Tách `device` theo tầng.** Xem "Chuyện GPU" dưới đây.
3. **Nâng `n_mc_samples`** trong mọi config (4 → 16+), vì nó gần như miễn phí.

### Việc còn phải làm, theo thứ tự

1. **Push lên GitHub**: `git push origin main:main` (chưa làm).
2. **Chốt chính sách `ruff`**: ruff 0.16.3 báo **40 lỗi có sẵn** trên code
   chưa đụng vào, vì `pyproject.toml` chỉ đặt `line-length` mà không khai
   báo `[tool.ruff.lint] select` — phiên bản mới bật thêm nhiều rule mặc
   định. Cần chọn: khai báo `select` để cố định chính sách, hoặc ghim phiên
   bản ruff. Không nên im lặng format lại 40 chỗ.
3. **`canonical_theta` khi nhiều seed — CHẶN `gpu_research.yaml`.** Đo được
   label switching giữa các seed: chỉ số state thô đứng đầu đổi theo seed
   (3, 2, 2, 3 trên 4 seed thử), và cấu trúc occupancy khác hẳn nhau. Nhưng
   `canonical_theta` làm `mus.mean(dim=0)` — trung bình cộng tham số qua các
   seed. Trung bình `omega[3]` của seed 0 với `omega[3]` của seed 1 khi hai
   chỉ số đó chỉ hai chế độ khác nhau là phép toán vô nghĩa. Chưa cắn ai vì
   mọi config smoke dùng `seeds: [0]`, nhưng `gpu_research.yaml` dùng 8 seed.
   Phải căn chỉnh trạng thái trước khi trung bình, hoặc bỏ hẳn cách trung
   bình tham số. (`sample_joint_draw` an toàn — nó chọn một seed rồi mới rút.)
4. **Batch theo seed trong ADVI** — nguồn tăng tốc lớn nhất còn lại cho
   `gpu_research.yaml` (8 seed → 1 lần chạy). Chưa làm vì logic
   retry/fallback là theo từng seed, gộp lại phải viết lại bằng mặt nạ theo
   hàng, chạm đúng quy tắc "không silent fallback".
5. **Đo thật ở T đầy đủ (~6 264 phiên)** rồi mới chốt `gpu_research.yaml`.
   Mọi số đo hiện có đều ở T=1500; T đầy đủ dự kiến đắt hơn ~4,2× nhưng đó
   là **ngoại suy chưa kiểm chứng**. Không nên khởi động một lần chạy nhiều
   giờ dựa trên giả định.
6. **Sub-project 4**, 8 task theo plan.
7. Sau cùng: chạy nghiên cứu quy mô thật → số liệu cho model card và phần
   "Kết quả" của README này (ghi rõ smoke-scale hay không).

### Chuyện GPU — kết luận ngược với dự đoán ban đầu

`device_preference: auto` **không phải** lựa chọn tốt cho mọi tầng. Trên máy
này `auto` chọn CUDA, và điều đó làm **ADVI chậm đi ~4,3 lần**.

| Tầng | Nên chạy ở đâu | Vì sao |
|---|---|---|
| ADVI (fit MS-EGARCH, mu_k) | **CPU** | batch thực tế chỉ 16×4 = 64 phần tử, quá nhỏ để bù overhead phát kernel |
| Monte Carlo | **GPU khi n_paths ≳ 2 000–3 000** | dưới ngưỡng CPU thắng; trên ngưỡng thời gian CUDA gần như phẳng |

Số đo Monte Carlo (T=1500, horizon=20): 500 path — CPU 2,49 s / CUDA 6,03 s;
5 000 path — CPU 13,79 s / CUDA 5,64 s; 20 000 path — CPU 19,85 s / **CUDA
5,68 s**.

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
  evaluation/      # (sub-project 4, đang làm, trên branch evaluation-model-card)
  validation/      # split, leakage checks (dùng chung)
configs/           # các profile ADVI (cpu_smoke, ebm_smoke, mc_smoke, gpu_research-stub)
docs/
  superpowers/specs/    # thiết kế chi tiết từng sub-project
  superpowers/plans/    # implementation plan từng sub-project
  perf_batching_notes.md # số đo hiệu năng CPU/GPU thật + điểm giao + việc chưa làm
  sdd-records/          # ledger + báo cáo debug của sub-project 3
  model_card.md         # (sẽ có sau sub-project 4)
```
