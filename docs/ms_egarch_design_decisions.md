# Quyết định thiết kế MS-EGARCH (a-d)

Ngày: 2026-08-17
Tham chiếu: docs/superpowers/specs/2026-08-17-ms-egarch-foundation-design.md

Tài liệu này ghi lại 4 quyết định thiết kế còn mở, kèm lý do và đánh đổi đã
cân nhắc. Đây là lựa chọn có lập luận riêng, không phải kết luận đã được
kiểm chứng bởi tài liệu học thuật chuẩn mực cho trường hợp EGARCH cụ thể -
sẽ cần bằng chứng OOS trước khi khẳng định mô hình tốt hơn.

## (a) Gộp ở level-space, không phải log-space

**Lựa chọn:** gộp tại `torch.logsumexp(log_filtered_prob + log_var)`, tương
đương với gộp trên level-space (trung bình có trọng số của sigma^2 theo
từng trạng thái) rồi lấy log, không phải trung bình trực tiếp trên
log-variance.

**Lý do:** gộp level-space bảo toàn đúng một moment thật của phân phối trộn
- `E[sigma_t^2 | F_{t-1}] = sum_k P_k * sigma_k,t^2` theo định lý phương
sai toàn phần. Gộp trực tiếp trên log-variance không phải một moment thật,
và do bất đẳng thức Jensen (log là hàm lõm) sẽ luôn thiên lệch xuống so
với giá trị đúng - sai lệch càng lớn khi xác suất filtered càng phân tán
giữa các trạng thái có biến động chênh lệch nhau nhiều, đúng lúc cần phân
biệt regime rõ nhất. Về tính toán, `torch.logsumexp` cho ra chính xác kết
quả này mà không cần exponentiate tường minh, ổn định số học hơn.

**Đánh đổi:** không có tài liệu học thuật chuẩn mực xác nhận lựa chọn này
cho trường hợp EGARCH cụ thể (Gray's paper gốc là cho GARCH thường, trên
level-space thuần, không có đệ quy log). Đã viết test suy biến K=1
(`test_degenerate_single_state_matches_reference_egarch`) để xác nhận
không có sai khác khi không còn bất định giữa các trạng thái.

## (b) Chuẩn hoá z[t-1] bằng sigma_bar[t-1] đã gộp

**Lựa chọn:** `z[t-1] = eps[t-1] / sigma_bar[t-1]`, với sigma_bar[t-1] lấy
từ chính giá trị gộp ở quyết định (a).

**Lý do:** `eps[t-1]` là một số quan sát được duy nhất (đã thực sự xảy ra),
cần một sigma tham chiếu duy nhất để chuẩn hoá. sigma_bar gộp là lựa chọn
tự nhiên nhất vì đây chính là giá trị được dùng làm đầu vào cho đệ quy
log-variance của bước kế tiếp, giữ toàn bộ hệ nhất quán nội bộ.

**Đánh đổi:** Gray's paper gốc không có số hạng bất đối xứng nên không có
tiền lệ trực tiếp cho lựa chọn này trong bối cảnh multi-regime.

## (c) Bỏ c_k ở tầng scenario-return

**Lựa chọn:** không giữ hệ số nhân `c_k` riêng theo regime ở tầng
scenario-return (sẽ triển khai ở sub-project 3) - chỉ giữ mu_k và nu_k,
vì MS-EGARCH đã cung cấp sigma_k,t riêng theo từng regime.

**Lý do:** giữ thêm c_k làm hệ số nhân tự do khác lên trên có nguy cơ
non-identifiability - hai tham số cùng giải thích một scale, ELBO có sống
núi phẳng dọc theo hướng c_k * sigma, posterior mean-field có thể trông
"tự tin" giả tạo ở từng biến trong khi thực chất không định danh được.

**Đánh đổi:** nếu sau này phát hiện phân tán trong-regime không được giải
thích hết bởi đệ quy EGARCH (ví dụ qua phân tích residual thực tế), có thể
cần thêm lại một tham số scale bổ sung - nhưng chưa có bằng chứng thực
nghiệm cho điều này tại thời điểm viết tài liệu, nên chưa thêm.

## (d) Hai giai đoạn Bayesian tách biệt, không một ELBO chung

**Lựa chọn:** fit posterior MS-EGARCH độc lập trước (không cần scenario/
return labels); tầng scenario-return (sub-project 3) sẽ điều kiện theo
posterior draws của MS-EGARCH thông qua `sample_ms_egarch_draw`.

**Lý do:** đây là một dạng modular/"cut" Bayesian inference - cắt đường
phản hồi ngược từ một mô-đun có thể bị misspecify (tầng scenario) sang
mô-đun trước nó (MS-EGARCH), một kỹ thuật có cơ sở thống kê chứ không chỉ
vì tiện triển khai. Cũng khớp tự nhiên với việc tách sub-project 2 và
sub-project 3 theo quản lý phạm vi đã thống nhất, giúp mỗi mô-đun test độc
lập được.

**Đánh đổi:** không lan truyền bất định của mu_k/nu_k ngược lại vào fit
MS-EGARCH (điều kiện một chiều) - chấp nhận được vì đây là xấp xỉ phổ biến
trong modular Bayes, và là đổi lấy được sự đơn giản/dễ debug rõ ràng.

## Hệ quả interface

`sample_ms_egarch_draw(posterior, layout, generator)` trả về đúng một
`MSEGARCHParams` cho mỗi lần gọi - nội dung mà sub-project 3 sẽ dùng để lấy
"một joint draw duy nhất cho mỗi Monte Carlo path, giữ cố định suốt
horizon" (không resample ngầm trong vòng lặp simulate).
