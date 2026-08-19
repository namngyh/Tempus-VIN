# Ghi chú hiệu năng: vector hoá theo chiều batch và chuyện GPU

Ngày đo: 2026-08-19. Máy: RTX 4060 8 GB (sm_89, driver CUDA 13.2), CPU 16
nhân, Windows 11, Python 3.13.14, `torch 2.13.0+cu130`, float32.

Tài liệu này ghi lại **số đo thật**, không phải ước lượng. Mọi con số dưới
đây tái lập được bằng các script trong lịch sử phiên làm việc; phương pháp
đo được mô tả kèm từng bảng.

## 1. Vấn đề: vì sao GPU ban đầu CHẬM HƠN CPU

README trước đợt này ghi "đường GPU chưa từng được chạy thật — có thể còn
lỗ hổng chưa phát hiện". Khi chạy thật lần đầu, kết quả không phải là "GPU
nhanh hơn ít hơn kỳ vọng" mà là **GPU chậm hơn CPU nhiều lần**:

| Phép đo (T=1500, n_states=4) | CPU | CUDA | Tỷ lệ |
|---|---|---|---|
| `run_ms_egarch_recursion`, 1 lần | 522 ms | 2 921 ms | CUDA chậm **5,6×** |
| ADVI 20 bước × 4 MC sample | 126,5 s | 620,8 s | CUDA chậm **4,9×** |

Nguyên nhân không phải phần cứng mà là **hình dạng công việc**:

- `run_ms_egarch_recursion` là bộ lọc tuần tự theo thời gian — 1500 vòng lặp
  Python, mỗi vòng thao tác trên tensor **4 phần tử**. Chi phí thống trị là
  overhead phát kernel (~5–10 µs/kernel trên CUDA), không phải phép tính.
  Nhân 1500 bước × ~25 kernel = ~37 500 lần phát kernel cho một recursion.
- ADVI gọi recursion `n_steps × n_mc_samples` lần **tuần tự** → 300 × 4 =
  1 200 recursion cho mỗi seed.
- `simulate_mc_paths` chạy `n_paths` recursion riêng biệt rồi lặp
  `n_paths × horizon` bước vô hướng, mỗi bước gọi `.item()` — mỗi lời gọi
  ép CPU chờ GPU chạy xong (đồng bộ hoá), thứ giết throughput GPU triệt để.

Chiều thời gian **không thể** song song hoá (bộ lọc Hamilton bản chất là
tuần tự). Thứ song song hoá được là các chiều **độc lập**: MC sample của
ELBO, seed, path Monte Carlo, mốc walk-forward. Đó là nội dung đợt này.

## 2. Đã vector hoá những gì

| Chỗ | Trước | Sau |
|---|---|---|
| `run_ms_egarch_recursion` | 1 theta / lần gọi | `(B, ...)` theta / lần gọi |
| ADVI `_single_attempt` | `n_mc_samples` lần gọi log_joint tuần tự | 1 lần gọi, theta `(S, P)` |
| `simulate_mc_paths` | `n_paths` recursion + vòng lặp vô hướng có `.item()` | 1 recursion batch + `horizon` bước vector hoá, không `.item()` |
| `mu_fit._precompute_draws` | `n_draws` recursion tuần tự | 1 recursion batch |

Nguyên tắc thiết kế quan trọng: **đường không-batch là B=1 của chính đường
batch** (`unsqueeze` vào đầu, `squeeze` ra cuối), chứ không phải hai nhánh
code song song. Nhờ vậy hai dạng không thể trôi khỏi nhau về số học, và
toàn bộ test đã có của dạng không-batch đồng thời là test của kernel batch.
Bổ sung hai test trực tiếp cho hợp đồng này trong
`tests/regime/test_ms_egarch_recursion.py`: một test đối chiếu giá trị
batch với vòng lặp từng theta, một test chứng minh gradient **không rò rỉ**
giữa các hàng batch (điều kiện sống còn để gộp MC sample vào một bước ADVI —
giá trị forward vẫn có thể đúng khi chỉ đường gradient bị hỏng).

ADVI chỉ đi đường batch khi `log_joint` khai báo thuộc tính
`supports_batched_theta = True`. Các `log_joint` khác giữ nguyên đường tuần
tự cũ — không có thay đổi ngầm nào.

## 3. Kết quả đo sau vector hoá

### ADVI MS-EGARCH (T=1500, 20 bước, 1 seed)

| n_mc_samples | CPU | CUDA |
|---|---|---|
| 4 (trước vector hoá) | 126,5 s | 620,8 s |
| 4 | **34,9 s** | 159,9 s |
| 16 | **35,9 s** | 154,6 s |

Hai điều đáng chú ý:

1. CPU nhanh lên **3,6×**. Quy ra 300 bước: ~32 phút → **~8,7 phút**.
2. Tăng `n_mc_samples` từ 4 lên 16 chỉ tốn thêm **3%** thời gian. Đây là hệ
   quả trực tiếp và quan trọng nhất: chất lượng ước lượng gradient ELBO giờ
   gần như miễn phí. README liệt kê "nhãn regime có thể suy biến dưới config
   ADVI nhanh" và "một phần posterior nằm ngoài vùng ổn định" như rủi ro đã
   biết — cả hai đều là triệu chứng của ADVI chưa hội tụ, và `n_mc_samples`
   cao hơn là một trong những cách rẻ nhất để cải thiện.

### Monte Carlo (T=1500, horizon=20)

| n_paths | CPU | CUDA |
|---|---|---|
| 500 | **2,49 s** | 6,03 s |
| 5 000 | 13,79 s | **5,64 s** |
| 20 000 | 19,85 s | **5,68 s** |

Trước vector hoá, 500 path tốn ~275 s trên CPU (500 recursion tuần tự × 522
ms + vòng lặp vô hướng) → **~110×**.

## 4. Khi nào nên dùng GPU, khi nào không

Kết luận rút ra từ bảng trên, và nó **ngược với giả định ban đầu của
README**:

- **ADVI (fit MS-EGARCH và μ_k): dùng CPU.** Ngay cả sau vector hoá, kích
  thước batch thực tế là `n_mc_samples × n_states` = 16 × 4 = 64 phần tử —
  quá nhỏ để bù overhead phát kernel. CUDA vẫn chậm hơn CPU ~4,3×.
- **Monte Carlo: điểm giao nằm quanh ~2 000–3 000 path.** Dưới ngưỡng đó
  CPU thắng; trên ngưỡng đó GPU thắng và **thời gian CUDA gần như phẳng**
  (5,6 s cho cả 5 000 lẫn 20 000 path) vì vẫn còn giới hạn bởi số lần phát
  kernel chứ chưa chạm giới hạn tính toán. Ở quy mô nghiên cứu (nhiều chục
  nghìn path) GPU là lựa chọn đúng.

Vì vậy `device_preference: auto` trong config **không phải** lựa chọn tốt
cho mọi tầng: `auto` sẽ chọn CUDA trên máy này và làm ADVI chậm đi 4 lần.
Xem mục 6.

## 5. Bug đường CUDA đã tìm ra và sửa

`sample_joint_draw` gọi `torch.randint(n_seeds, (1,), generator=generator)`
**không có `device=`**. `torch.randint` mặc định cấp phát trên CPU, nên với
generator CUDA nó raise:

```
RuntimeError: Expected a 'cpu' device type for generator but found 'cuda'
```

Lỗi này chắn ngang đường đi của `simulate_mc_paths` với `device=cuda`, tức
**toàn bộ tầng Monte Carlo không thể chạy trên GPU** trước khi sửa. Đã tái
hiện được trên máy này trước khi sửa; `tests/runtime/test_cuda_device_path.py`
là guard hồi quy.

Đây đúng là loại lỗ hổng README dự đoán ("device đã được thread xuyên suốt
sau nhiều vòng review, nhưng đường GPU chưa từng được chạy thật"). Việc
thread `device` qua từng lời gọi cấp phát tensor là cần nhưng chưa đủ: các
hàm nhận `generator` mà tự cấp phát tensor bên trong là một lớp lỗi riêng
mà review tĩnh trên máy CPU-only không thể bắt.

## 6. Việc chưa làm (đề xuất, không phải đã xong)

- **Chưa batch theo seed.** `fit_multi_seed_advi` vẫn chạy tuần tự từng
  seed. Các seed hoàn toàn độc lập nên về nguyên tắc gộp được thành chiều
  batch (8 seed → 1 lần chạy). Chưa làm vì logic retry/fallback/phân kỳ là
  **theo từng seed**, và gộp lại đòi hỏi viết lại phần đó bằng mặt nạ theo
  hàng — chạm đúng vào quy tắc "không silent fallback" của dự án, nên không
  nên làm vội. Đây là nguồn tăng tốc lớn nhất còn lại cho
  `gpu_research.yaml` (8 seed).
- **Chưa đo ở T đầy đủ (~6 264 phiên).** Mọi số trên đo ở T=1500. Chi phí
  recursion tuyến tính theo T, nên T đầy đủ dự kiến đắt hơn ~4,2×, nhưng
  đây là **ngoại suy chưa kiểm chứng** — cần đo trước khi lên lịch chạy dài.
- **`configs/gpu_research.yaml` chưa được kiểm chứng về tính khả thi thời
  gian.** Với `n_steps: 20000`, `seeds: [0..7]`, `window_sessions: null`,
  ngoại suy từ số đo hiện tại cho ra thời gian chạy rất lớn. Phải đo thật ở
  T đầy đủ rồi mới chốt cấu hình, thay vì khởi động một lần chạy nhiều ngày
  dựa trên giả định.
- **`device_preference` nên tách theo tầng** (ADVI → cpu, Monte Carlo →
  cuda khi n_paths lớn) thay vì một khoá `auto` dùng chung. Config hiện tại
  chưa hỗ trợ, và việc này nên đi kèm sub-project 4 khi walk-forward cần
  chạy nhiều mốc.

## 7. Về tính tái lập

Vector hoá **đổi thứ tự tiêu thụ số ngẫu nhiên** (một tensor `(S, P)` thay
cho `S` tensor `(P,)`; một tensor chỉ số `(B,)` thay cho `B` lời gọi
`randint` riêng lẻ). Hệ quả:

- Tính tái lập theo seed **giữ nguyên**: cùng generator, cùng cấu hình →
  cùng kết quả, chạy lại bao nhiêu lần cũng vậy. Test tái lập hiện có vẫn
  pass không sửa đổi.
- So sánh **xuyên phiên bản** thì không: kết quả không trùng bit-đối-bit với
  bản trước vector hoá. Không có số liệu nào đã công bố dựa trên bản cũ, nên
  điều này không làm mất hiệu lực kết quả nào — nhưng phải ghi lại để không
  ai đi truy một sai khác không tồn tại.
- Lô hoá path (`path_chunk_size`) cũng đổi chuỗi ngẫu nhiên: chạy 70 path
  một lô khác chạy 70 path chia lô 32. Mặc định là `None` (một lô) nên hành
  vi mặc định không phụ thuộc tham số này.
