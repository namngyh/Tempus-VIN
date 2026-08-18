# RAEMF-VB-MC

Mô hình dự báo xác suất chế độ thị trường (Bull/Sideway/Bear/Stress) và phân phối
lợi suất VN-Index, dùng MS-EGARCH (Markov-Switching EGARCH 4 trạng thái) ước lượng
toàn phần bằng Variational Bayes (ADVI), lớp phân loại EBM, tầng scenario-return +
Monte Carlo, và risk metrics (VaR/CVaR/drawdown).

Xem `docs/superpowers/specs/` cho thiết kế chi tiết từng sub-project và
`docs/ms_egarch_design_decisions.md` cho các quyết định thiết kế còn mở.

## Cài đặt

Repo dùng trực tiếp Python 3.11 hệ thống hiện có (torch, numpy, pandas, scipy,
pytest, scikit-learn, statsmodels, pyarrow, interpret đã có sẵn). Cài package ở
chế độ editable:

```
pip install -e ".[dev]"
```

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
| 3 | Scenario-return + Monte Carlo + VaR/CVaR | Fit mu_k theo regime, mô phỏng Monte Carlo tiến, tính VaR/CVaR/drawdown | **Code xong + fix wave xong, CHƯA merge — xem mục dưới** |
| 4 | Đánh giá OOS + Model card | Hoàn thiện đánh giá EBM (macro F1, recall Bear/Stress, NLL), walk-forward OOS (CRPS/WIS/Kupiec), model card | **Mới bắt đầu, chưa có code — xem mục dưới** |

---

## Trạng thái hiện tại (2026-08-18) — DỪNG Ở ĐÂY, TIẾP TỤC TRÊN MÁY GPU

### Sub-project 3 (branch `scenario-mc-var`, worktree `.worktrees/scenario-mc-var`)

Cả 7 task đã xong và review sạch (spec + code quality từng task). Whole-branch
review (Opus) tìm ra 1 lỗi Critical + 3 Important + vài Minor — **đã fix hết**,
commit cuối `b68b492` trên branch `scenario-mc-var`. Chi tiết đầy đủ trong
`.worktrees/scenario-mc-var/.superpowers/sdd/2026-08-18-scenario-mc-var/progress.md`
(ledger — nếu worktree còn tồn tại; nếu đã bị xóa thì git log của branch là bản
ghi chính xác).

**Việc còn thiếu trước khi merge** (dừng lại đúng lúc này):

1. **Chạy lại test tích hợp thật** để xác nhận fix cuối cùng (`b68b492`) không
   làm hỏng gì:
   ```
   cd .worktrees/scenario-mc-var
   python -m pytest tests/test_scenario_mc_integration_smoke.py -v -s
   ```
   (mất ~90 phút trên CPU — trên GPU sẽ nhanh hơn nhiều nếu code được chạy qua
   `device=cuda`, nhưng lưu ý: hiện tại `simulate_mc_paths`/`fit_regime_mu` đã
   hỗ trợ `device` param nhưng CHƯA TỪNG được chạy/kiểm chứng trên GPU thật —
   xem "Rủi ro GPU" ở dưới).
2. Nếu pass: chạy full suite (`python -m pytest -q` trong worktree đó) +
   `python -m ruff check src tests scripts`.
3. Merge vào `master`: dùng skill `finishing-a-development-branch`, hoặc thủ
   công:
   ```
   cd "đường dẫn repo gốc"
   git checkout master && git pull
   git merge scenario-mc-var
   pip install -e . --no-deps   # re-link editable install
   python -m pytest -q          # xác nhận lại trên master sau merge
   ```
4. Push lên GitHub: `git push origin master:main`.
5. Xóa worktree/branch sau khi merge xong (không bắt buộc, có thể giữ để tham
   khảo ledger).

### Sub-project 4 (branch `evaluation-model-card`, worktree `.worktrees/evaluation-model-card`)

Mới chỉ có: spec (`docs/superpowers/specs/2026-08-18-evaluation-model-card-design.md`),
plan 8-task đầy đủ (`docs/superpowers/plans/2026-08-18-evaluation-model-card.md`),
và worktree/branch đã tạo sẵn — **CHƯA có code nào được commit**. Task 1 (metric
phân loại) vừa được dispatch cho subagent thì bị dừng giữa chừng — không có gì
bị mất (implementer chưa kịp ghi file nào).

**Lưu ý quan trọng về phụ thuộc chéo branch**: Task 4 và Task 6 của plan này
import trực tiếp `fit_regime_mu`, `simulate_mc_paths` (từ `raemf_mc.scenario`)
và `compute_var` (từ `raemf_mc.risk.metrics`) — những hàm này **chỉ tồn tại
trên branch `scenario-mc-var`**, chưa có trên `master`. Vì vậy:

**Thứ tự bắt buộc khi tiếp tục:**
1. Merge `scenario-mc-var` vào `master` trước (xem mục trên).
2. Rebase/merge `master` mới vào `evaluation-model-card`:
   ```
   cd .worktrees/evaluation-model-card
   git fetch origin
   git merge master   # hoặc git rebase master
   pip install -e . --no-deps
   ```
3. Tiếp tục plan từ Task 1 (dùng skill `subagent-driven-development`, đọc lại
   `docs/superpowers/plans/2026-08-18-evaluation-model-card.md` và ledger tại
   `.worktrees/evaluation-model-card/.superpowers/sdd/2026-08-18-evaluation-model-card/progress.md`).
   Task 1-3, 5, 7 (metric phân loại, CRPS/WIS, Kupiec, config file, model card)
   không phụ thuộc sub-project 3 nên có thể làm trước bước merge ở trên nếu
   muốn — chỉ Task 4/6 (walk-forward harness + integration test) cần code của
   sub-project 3 đã có trên nhánh đang làm việc.

### Sau khi cả hai sub-project trên xong và merge

- Viết phần "Kết quả" vào README này với số liệu thật (từ integration test
  thật), ghi rõ **smoke-scale, không phải kết quả nghiên cứu cuối cùng**.
- Xem xét chạy lại toàn bộ pipeline với cấu hình ADVI đầy đủ hơn
  (`configs/gpu_research.yaml` — hiện là stub, chưa từng chạy thật) trên máy
  GPU: nhiều ADVI step hơn, nhiều seed hơn, walk-forward Sub-project 4 với
  nhiều mốc lịch sử hơn (hiện smoke-scale chỉ dùng 5-10 mốc, không đủ để
  Kupiec test có ý nghĩa thống kê thật).

---

## Rủi ro / giới hạn đã biết (tổng hợp xuyên suốt dự án)

- **CPU-only, chưa kiểm chứng GPU**: máy hiện tại chỉ có Intel Iris Xe tích
  hợp, không có CUDA. `device`/`dtype` đã được thread xuyên suốt code (sau
  nhiều vòng fix tìm thấy bằng review), nhưng đường GPU (`device=torch.device("cuda")`)
  **chưa từng được chạy thật** — có thể còn lỗ hổng chưa phát hiện được vì
  không thể test trực tiếp ở đây.
- **Không walk-forward re-fit thật** ở sub-project 1/2/3 — MS-EGARCH được fit
  MỘT LẦN trên toàn bộ cửa sổ, không refit theo từng fold như một walk-forward
  sản xuất thật sự cần. Sub-project 4 có xây walk-forward NHƯNG chỉ ở quy mô
  smoke (5-10 mốc, ADVI rất nhẹ) — không phải benchmark thống kê đáng tin.
- **Nhãn regime có thể suy biến dưới config ADVI nhanh**: đã quan sát ở
  sub-project 2 — một số cửa sổ dữ liệu thật cho ra >95% phiên cùng một nhãn
  regime khi ADVI chỉ chạy 300 bước. Cần chạy ADVI hội tụ tốt hơn (nhiều bước/
  seed hơn) trước khi tin số liệu phân loại/rủi ro là đại diện thật.
- **Một phần posterior MS-EGARCH nằm ngoài vùng ổn định (`|beta|<1`)**: prior
  chỉ là ràng buộc mềm (Normal(0.9, 0.15)), không phải constraint cứng — đã
  xác nhận thật qua debug sub-project 3 (37% draw/state trong một fit thử
  nghiệm không stationary). Tầng scenario-return đã xử lý an toàn (không "nổ"
  số), nhưng đây vẫn là dấu hiệu fit chưa hội tụ tốt, không phải đã giải quyết
  tận gốc (giải pháp dùng constraint cứng như `beta=tanh(beta_raw)` thuộc
  phạm vi sub-project 1, chưa làm).
- **Chưa có lưu/tải model (persistence)**: mỗi lần chạy đều fit lại từ đầu,
  không serialize posterior ra đĩa.
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
configs/           # các profile ADVI (cpu_smoke, ebm_smoke, mc_smoke, eval_smoke, gpu_research-stub)
docs/
  superpowers/specs/    # thiết kế chi tiết từng sub-project
  superpowers/plans/    # implementation plan từng sub-project
  model_card.md         # (sẽ có sau sub-project 4)
```
