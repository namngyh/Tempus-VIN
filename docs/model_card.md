# Model card — RAEMF-VB-MC

Ngày: 2026-08-24. Trạng thái: **nghiên cứu, chưa dùng được để ra quyết định.**

Tài liệu này mô tả mô hình dự báo xác suất chế độ thị trường và phân phối lợi
suất VN-Index. Nó được viết theo nguyên tắc của `AGENTS.md`: **không bịa số
liệu; thành phần không ước lượng được phải ghi rõ lý do.** Mọi con số dưới đây
là số đo từ một lần chạy cụ thể, có ghi cấu hình kèm theo. Tài liệu này không
đưa ra lời khuyên đầu tư.

---

## 1. Mục đích và phạm vi sử dụng

**Mục đích**: ước lượng, cho mỗi phiên giao dịch, (a) phân phối xác suất trên
bốn chế độ thị trường và (b) phân phối lợi suất VN-Index cho chân trời tới,
kèm các chỉ số rủi ro (VaR, CVaR, drawdown).

**Phạm vi đã kiểm chứng**: VN-Index, tần suất **ngày**, dữ liệu 2000-07-28 →
2026-07-13 (6 264 phiên).

**Ngoài phạm vi, chưa kiểm chứng**:
- Bất kỳ mã nào khác ngoài VN-Index.
- Bất kỳ tần suất nào khác ngày. Dữ liệu 5 phút có sẵn nhưng thiết kế cố ý
  loại trừ (chế độ kinh tế không vận hành ở tần số đó; mở rộng cần xử lý
  riêng tính mùa vụ trong ngày).
- Ra quyết định giao dịch dưới bất kỳ hình thức nào. Xem mục 5.

---

## 2. Kiến trúc

| Tầng | Nội dung |
|---|---|
| MS-EGARCH | 4 chế độ, chuyển Markov, EGARCH log-phương sai, đổi mới Student-t. Ước lượng toàn phần bằng Variational Bayes (ADVI mean-field). |
| Nhãn chế độ | argmax của xác suất lọc **đã chuẩn hoá theo tần suất nền** — xem mục 4.2, nhãn đã ĐỔI NGHĨA. |
| EBM classifier | Explainable Boosting Machine trên 9 đặc trưng OHLCV nhân quả + posterior mean/sd của sigma_t. Hiệu chỉnh bằng temperature scaling. |
| Scenario-return | mu_k theo chế độ, fit bằng ADVI riêng, có điều kiện trên draw MS-EGARCH. |
| Monte Carlo | Mô phỏng tiến, mỗi path rút theta và mu riêng, lấy mẫu chế độ thực hiện theo ngày. |
| Rủi ro | VaR, CVaR, max drawdown từ phân phối path. |

`n_nu_params = n_states`: **một bậc tự do Student-t cho mỗi chế độ**, không
dùng chung. Xem mục 4.3 về lý do và về kết quả thực tế của thay đổi đó.

---

## 3. Kết quả đo được

### 3.1 Fit MS-EGARCH quy mô nghiên cứu

Cấu hình `configs/gpu_research.yaml`; 6 263 phiên (train 4 384); 8 seed chạy
song song; 1 500 bước ADVI; `n_mc_samples=16`; CPU. Thời gian thật: **9,5
giờ** (71 phút/seed).

| | |
|---|---|
| Chẩn đoán `regime_health_report` | **healthy: True** |
| Occupancy 4 chế độ | 0,3868 · 0,2078 · 0,0997 · 0,3056 |
| ELBO cuối, 8 seed | 13 175,85 … 13 215,78 (chênh 0,3%) |
| Số seed rơi vào fallback | 0/8 |
| `clamp_saturation_fraction` | 0,0 |

Tham số theo chế độ (đã căn chỉnh theo `omega/(1-beta)` tăng dần):

| k | omega | alpha | beta | gamma | nu | tự chuyển |
|---|---|---|---|---|---|---|
| 0 | −1,2463 | 0,6585 | 0,8864 | +0,0356 | 3,246 | 0,9754 |
| 1 | −1,1667 | 0,5826 | 0,8665 | −0,0488 | 3,061 | 0,9505 |
| 2 | −0,9892 | 0,7179 | 0,8693 | −0,0440 | 2,909 | 0,7609 |
| 3 | −1,0422 | 0,6496 | 0,8323 | −0,0788 | 3,122 | 0,9538 |

Tất cả `beta` nằm trong vùng dừng. Tự chuyển 0,76–0,98 (chế độ dai dẳng).

**So sánh với trước**: cấu hình smoke cũ (300 bước, `n_mc=4`, cửa sổ 1 500,
`nu` dùng chung) cho occupancy `[0,827 · 0,074 · 0,072 · 0,028]` — một chế độ
chiếm 83%, và nhãn suy biến hoàn toàn về một lớp (1 049/1 049 phiên).

### 3.2 Walk-forward OOS

`configs/eval_smoke.yaml`: cửa sổ 2 500 phiên, 8 mốc, chân trời 20 phiên,
500 path Monte Carlo, ADVI 200 bước. Thời gian thật **1 giờ 34 phút**.

| cutoff | realized_return | CRPS | WIS | var_95 | var_99 |
|---|---|---|---|---|---|
| 1249 | +0,0652 | 0,1468 | 0,1176 | 1,2646 | 2,2870 |
| 1424 | +0,0471 | 0,0949 | 0,0715 | 0,6210 | 1,0453 |
| 1600 | −0,0523 | 0,1460 | 0,1110 | 1,1438 | 1,8382 |
| 1776 | +0,0414 | 0,1197 | 0,1065 | 1,0131 | 1,9828 |
| 1951 | +0,0069 | 0,1334 | 0,1007 | 0,9921 | 1,9473 |
| 2127 | −0,0097 | 0,2296 | 0,2008 | 1,9543 | 2,7257 |
| 2303 | +0,0134 | 0,1042 | 0,0806 | 0,8210 | 1,8276 |
| 2479 | +0,0007 | 0,1360 | 0,1061 | 1,1890 | 1,9785 |

Vi phạm: **0/8** ở cả hai mức. Kupiec LR: 0,821 (95%) và 0,161 (99%) — cả hai
dưới ngưỡng bác bỏ 3,841.

**Đọc kỹ chỗ này: "không bác bỏ" ở đây KHÔNG phải tin tốt.** VaR_95 dao động
0,62–1,95, tức mô hình dự báo mức lỗ 62%–195% trong 20 phiên. Lợi suất thực tế
20 phiên trong cùng giai đoạn nằm trong khoảng ±0,065. **VaR rộng gấp 20–30
lần biên độ thật, nên không đời nào bị thủng.** Kupiec không bác bỏ vì VaR vô
dụng chứ không phải vì nó được hiệu chỉnh đúng — đây là minh hoạ trực tiếp cho
hai giới hạn ở mục 4.1 và 4.3, và là lý do không có khẳng định nào về hiệu
chỉnh trong tài liệu này.

CRPS 0,095–0,230 và WIS 0,072–0,201 phản ánh cùng vấn đề: một phân phối dự báo
quá rộng bị phạt bởi thành phần sắc nét (sharpness) của cả hai thước đo.

### 3.3 Hiệu năng

Đo trên RTX 4060 (WDDM) + CPU 16 nhân, một bước ADVI, T=1 500, `n_mc=16`:

| Đường | s/bước |
|---|---|
| CUDA eager | 9,618 |
| CPU | 2,172 |
| CUDA graph | **0,204** |

Overhead phát kernel: ~42 µs (eager) so với ~1,5 µs (graph). CUDA graph nhanh
nhất rất xa nhưng **không dùng được ở cửa sổ lớn** — xem mục 4.5.

---

## 4. Giới hạn đã biết

Đây là mục quan trọng nhất của tài liệu này.

### 4.1 Walk-forward OOS không có sức mạnh thống kê

`configs/eval_smoke.yaml` chạy 8 mốc. Với p = 0,05, số vi phạm VaR_95 kỳ
vọng là **0,4**. Quan sát 0, 1 hay 2 lần vi phạm đều nằm trong vùng không
phân biệt được một VaR đúng với một VaR sai hệ thống. Kiểm định Kupiec ở cỡ
mẫu này xác nhận **công thức chạy đúng**, không xác nhận **mô hình hiệu chỉnh
đúng**. Muốn kết luận thật cần hàng chục mốc trở lên, và mỗi mốc là một lần
fit lại đầy đủ (71 phút ở quy mô nghiên cứu) — tức một lần chạy tính bằng
ngày, chưa thực hiện.

CRPS và WIS ở quy mô này phản ánh trực tiếp chất lượng của một fit ADVI 200
bước (gần như chắc chắn chưa hội tụ: đo được ELBO chỉ phẳng từ bước ~700),
không phải tính chất của bản thân metric.

### 4.2 Nhãn chế độ đã ĐỔI NGHĨA

Nhãn **không** trả lời *"hôm nay chế độ nào khả năng cao nhất"* mà trả lời
*"hôm nay chế độ nào đang hoạt động BẤT THƯỜNG so với mức thường của nó"*.

Lý do: với một fit có occupancy `[0,724 · 0,202 · 0,052 · 0,022]`, chế độ thứ
hai giữ 20,2% khối lượng trung bình nhưng chỉ được gán nhãn ở **47/1 049**
phiên — nó hầu như không bao giờ *vượt* chế độ dẫn đầu ở bất kỳ ngày cụ thể
nào, nên argmax thô vứt bỏ toàn bộ biến thiên theo thời gian mà bộ lọc có
(max_prob dao động 0,33–0,92). Ablation 4 seed × 2 cấu hình trên cùng
`log_filtered_prob` đã fit: argmax cho mục tiêu **đơn lớp ở 5/8** trường hợp,
base_rate **0/8**.

**Mọi metric tính trên nhãn này — macro F1, recall Bear/Stress, NLL trước/sau
hiệu chỉnh — phải được đọc theo nghĩa thứ hai.**

### 4.3 `nu` vẫn thấp, và thay đổi cho `nu` riêng chế độ không đạt mục đích đề ra

`nu` bốn chế độ ra 3,246 / 3,061 / 2,909 / 3,122 — **gần như bằng nhau**. Lập
luận khi tách `nu` là "chế độ yên tĩnh và chế độ khủng hoảng không có lý do gì
cùng độ dày đuôi"; dữ liệu nói chúng *có*. Cho phép `nu` khác nhau không làm
chúng khác nhau.

`nu ≈ 3` nghĩa là **kurtosis vẫn vô hạn** (cần `nu > 4`). Ngưỡng chẩn đoán
`NU_FLOOR_MARGIN = 0,5` đặt ở mức "không hoàn toàn vô nghĩa", không phải "đã
tốt". Hệ quả trực tiếp: **các con số VaR/CVaR chưa đáng tin.** Ở quy mô smoke
đã đo `var_95 = 1,66` cho chân trời 5 ngày — mô hình nói "5% khả năng lỗ từ
166% trở lên", một con số vô nghĩa về kinh tế.

### 4.4 Không quy được nguyên nhân của cải thiện

Lần chạy nghiên cứu đổi **bốn thứ cùng lúc** so với lần đo trước: `nu` riêng
chế độ, `n_steps` 300 → 1 500, `n_mc_samples` 4 → 16, cửa sổ 1 500 → 6 263
phiên. Occupancy cải thiện rất mạnh, nhưng **chưa chạy thí nghiệm tách biến**,
nên không biết yếu tố nào gây ra. Bằng chứng ở mục 4.3 gợi ý `nu` **không**
phải nguyên nhân chính.

### 4.5 CUDA graph không tin cậy ở cửa sổ lớn

Đo được: T = 1 500 / 2 500 / 3 500 / 4 500 / 5 500 ghi graph được; T = 5 000 /
6 000 / 6 263 làm **cả tiến trình chết** với access violation `0xC0000005`.
Không đơn điệu theo T, và VRAM chỉ 0,07 GB nên không phải hết bộ nhớ. Vì vậy
`use_cuda_graph` mặc định **tắt** và cửa sổ đầy đủ chạy CPU.

### 4.6 Bốn chế độ vẫn chưa cho bốn lớp nhãn

Lần chạy nghiên cứu cho **2 lớp** (`Bull` 2 550, `Stress` 1 834) dù cả bốn chế
độ đều có khối lượng. `Sideway` và `Bear` không được gán phiên nào. Taxonomy 4
chế độ chưa được dữ liệu chống đỡ ở dạng nhãn cứng.

### 4.7 Label switching giữa các seed

Bốn trạng thái hoán đổi được, và các seed khác nhau đánh số khác nhau: đo được
4 seed cho ELBO gần như y hệt trong khi 3/4 có cùng cấu trúc occupancy chỉ
khác hoán vị. `canonical_theta` đã sắp lại từng seed theo `omega/(1-beta)`
trước khi trung bình. **Việc sắp xếp này dùng một tiêu chí thuần tham số**, và
chưa được kiểm chứng là ổn định trên mọi cấu hình.

### 4.8 Còn thiếu để vận hành

Không mảnh nào nằm trong 4 sub-project của kế hoạch:
- **Lưu/tải mô hình.** Mỗi lần chạy fit lại từ đầu. Một lần cập nhật hằng
  ngày về lý thuyết tốn ~4 giây (bộ lọc 0,52 s + đặc trưng posterior ~0,6 s +
  Monte Carlo 500 path 2,49 s) nhưng không tách được fit khỏi predict nên
  thực tế tối thiểu ~9 phút.
- **CLI/orchestration.** Chỉ có `scripts/run_research_fit.py` cho phần fit.
- **Đường nạp dữ liệu sống.** `load_vnindex_ohlcv` đọc CSV tĩnh.

### 4.9 Không so sánh với baseline

Quyết định có chủ đích từ sub-project 1: MS-EGARCH là hệ thống mới hoàn toàn,
benchmark chỉ đánh giá **nội tại**. Không có tuyên bố nào rằng mô hình này tốt
hơn bất kỳ phương pháp nào khác.

---

## 5. Không nên dùng để làm gì

Dựa trên mục 4, tại thời điểm này mô hình **không được dùng** để:

- Ra quyết định vào/ra vị thế, định cỡ vị thế, hay quản trị rủi ro thực tế.
  VaR/CVaR chưa đáng tin (mục 4.3).
- Tuyên bố rằng thị trường "đang ở chế độ X". Nhãn đã đổi nghĩa (mục 4.2) và
  chỉ có 2 trong 4 lớp xuất hiện (mục 4.6).
- Báo cáo bất kỳ số liệu OOS nào như bằng chứng về chất lượng dự báo. Walk-
  forward hiện tại không có sức mạnh thống kê (mục 4.1).

---

## 6. Tái lập

```
python -m venv .venv
.venv/Scripts/python -m pip install --index-url https://download.pytorch.org/whl/cu130 torch==2.13.0+cu130
.venv/Scripts/python -m pip install numpy pandas scipy pyyaml scikit-learn statsmodels pyarrow pytest ruff interpret
.venv/Scripts/python -m pip install -e . --no-deps

.venv/Scripts/python scripts/run_research_fit.py --config configs/gpu_research.yaml --parallel --out results/fit.json
.venv/Scripts/python -m pytest tests/test_evaluation_integration_smoke.py -q -s
```

Tính tái lập theo seed được giữ (cùng generator → cùng kết quả). Tính tái lập
**xuyên phiên bản** thì không: đợt vector hoá và đợt CUDA graph đều đổi thứ tự
tiêu thụ số ngẫu nhiên. Không có số liệu nào đã công bố dựa trên bản trước đó.

Môi trường của kết quả mục 3: Windows 11, Python 3.13.14, torch 2.13.0+cu130,
RTX 4060 8 GB (sm_89), CPU 16 nhân.
