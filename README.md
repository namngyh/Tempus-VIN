# RAEMF-VB-MC

Mo hinh du bao xac suat che do thi truong (Bull/Sideway/Bear/Stress) va phan phoi
loi suat VN-Index, dung MS-EGARCH (Markov-Switching EGARCH 4 trang thai) uoc luong
toan phan bang Variational Bayes (ADVI), lop phan loai EBM, tang scenario-return +
Monte Carlo, va risk metrics (VaR/CVaR/drawdown).

Xem `docs/superpowers/specs/` cho thiet ke chi tiet tung sub-project va
`docs/ms_egarch_design_decisions.md` cho cac quyet dinh thiet ke con mo.

## Cai dat

Repo dung truc tiep Python 3.11 he thong hien co (torch, numpy, pandas, scipy,
pytest, scikit-learn, statsmodels, pyarrow, interpret da co san). Cai package o
che do editable:

```
pip install -e ".[dev]"
```

## Chay test

```
python -m pytest -q
python -m ruff check src tests scripts
```

Luu y: full suite bao gom nhieu integration test chay ADVI/Monte Carlo THAT tren
du lieu VN-Index that, khong phai mock — moi test co the mat 15-90 phut. Xem
muc "Trang thai hien tai" duoi day de biet chinh xac test nao thuoc branch nao.

---

## Kien truc (4 sub-project)

| # | Ten | Muc dich | Trang thai |
|---|---|---|---|
| 1 | Nen tang + Loi MS-EGARCH | Ingest du lieu, ha tang ADVI dung chung, mo hinh MS-EGARCH 4 trang thai | **Xong, da merge `master`, da push GitHub** |
| 2 | EBM classifier + calibration | Phan loai regime tu dac trung OHLCV nhan qua + posterior sigma_t cua MS-EGARCH, temperature calibration | **Xong, da merge `master`, da push GitHub** |
| 3 | Scenario-return + Monte Carlo + VaR/CVaR | Fit mu_k theo regime, mo phong Monte Carlo tien, tinh VaR/CVaR/drawdown | **Code xong + fix wave xong, CHUA merge — xem muc duoi** |
| 4 | Danh gia OOS + Model card | Hoan thien danh gia EBM (macro F1, recall Bear/Stress, NLL), walk-forward OOS (CRPS/WIS/Kupiec), model card | **Moi bat dau, Task 1/8 dang lam do — xem muc duoi** |

---

## Trang thai hien tai (2026-08-18) — DUNG O DAY, TIEP TUC TREN MAY GPU

### Sub-project 3 (branch `scenario-mc-var`, worktree `.worktrees/scenario-mc-var`)

Ca 7 task da xong va review sach (spec + code quality tung task). Whole-branch
review (Opus) tim ra 1 loi Critical + 3 Important + vai Minor — **da fix het**,
commit cuoi `b68b492` tren branch `scenario-mc-var`. Chi tiet day du trong
`.worktrees/scenario-mc-var/.superpowers/sdd/2026-08-18-scenario-mc-var/progress.md`
(ledger — neu worktree con ton tai; neu da bi xoa thi git log cua branch la ban
ghi chinh xac).

**Viec con thieu truoc khi merge** (dung lai dung luc nay):

1. **Chay lai test tich hop that** de xac nhan fix cuoi cung (`b68b492`) khong
   lam hong gi:
   ```
   cd .worktrees/scenario-mc-var
   python -m pytest tests/test_scenario_mc_integration_smoke.py -v -s
   ```
   (mat ~90 phut tren CPU — tren GPU se nhanh hon nhieu neu code duoc chay qua
   `device=cuda`, nhung luu y: hien tai `simulate_mc_paths`/`fit_regime_mu` da
   ho tro `device` param nhung CHUA TUNG duoc chay/kiem chung tren GPU that —
   xem "Rui ro GPU" o duoi).
2. Neu pass: chay full suite (`python -m pytest -q` trong worktree do) +
   `python -m ruff check src tests scripts`.
3. Merge vao `master`: dung skill `finishing-a-development-branch`, hoac thu
   cong:
   ```
   cd "duong dan repo goc"
   git checkout master && git pull
   git merge scenario-mc-var
   pip install -e . --no-deps   # re-link editable install
   python -m pytest -q          # xac nhan lai tren master sau merge
   ```
4. Push len GitHub: `git push origin master:main`.
5. Xoa worktree/branch sau khi merge xong (khong bat buoc, co the giu de tham
   khao ledger).

### Sub-project 4 (branch `evaluation-model-card`, worktree `.worktrees/evaluation-model-card`)

Moi chi co: spec (`docs/superpowers/specs/2026-08-18-evaluation-model-card-design.md`),
plan 8-task day du (`docs/superpowers/plans/2026-08-18-evaluation-model-card.md`),
va worktree/branch da tao san — **CHUA co code nao duoc commit**. Task 1 (metric
phan loai) vua duoc dispatch cho subagent thi bi dung giua chung — khong co gi
bi mat (implementer chua kip ghi file nao).

**Luu y quan trong ve phu thuoc chua branch**: Task 4 va Task 6 cua plan nay
import truc tiep `fit_regime_mu`, `simulate_mc_paths` (tu `raemf_mc.scenario`)
va `compute_var` (tu `raemf_mc.risk.metrics`) — nhung ham nay **chi ton tai
tren branch `scenario-mc-var`**, chua co tren `master`. Vi vay:

**Thu tu bat buoc khi tiep tuc:**
1. Merge `scenario-mc-var` vao `master` truoc (xem muc tren).
2. Rebase/merge `master` moi vao `evaluation-model-card`:
   ```
   cd .worktrees/evaluation-model-card
   git fetch origin
   git merge master   # hoac git rebase master
   pip install -e . --no-deps
   ```
3. Tiep tuc plan tu Task 1 (dung skill `subagent-driven-development`, doc lai
   `docs/superpowers/plans/2026-08-18-evaluation-model-card.md` va ledger tai
   `.worktrees/evaluation-model-card/.superpowers/sdd/2026-08-18-evaluation-model-card/progress.md`).
   Task 1-3, 5, 7 (metric phan loai, CRPS/WIS, Kupiec, config file, model card)
   khong phu thuoc sub-project 3 nen co the lam truoc buoc merge o tren neu
   muon — chi Task 4/6 (walk-forward harness + integration test) can code cua
   sub-project 3 da co tren nhanh dang lam viec.

### Sau khi ca hai sub-project tren xong va merge

- Viet phan "Ket qua" vao README nay voi so lieu that (tu integration test
  that), ghi ro **smoke-scale, khong phai ket qua nghien cuu cuoi cung**.
- Xem xet chay lai toan bo pipeline voi cau hinh ADVI day du hon
  (`configs/gpu_research.yaml` — hien la stub, chua tung chay that) tren may
  GPU: nhieu ADVI step hon, nhieu seed hon, walk-forward Sub-project 4 voi
  nhieu moc lich su hon (hien smoke-scale chi dung 5-10 moc, khong du de
  Kupiec test co y nghia thong ke that).

---

## Rui ro / gioi han da biet (tong hop xuyen suot du an)

- **CPU-only, chua kiem chung GPU**: may hien tai chi co Intel Iris Xe tich
  hop, khong co CUDA. `device`/`dtype` da duoc thread xuyen suot code (sau
  nhieu vong fix tim thay bang review), nhung duong GPU (`device=torch.device("cuda")`)
  **chua tung duoc chay that** — co the con lo hong chua phat hien duoc vi
  khong the test truc tiep o day.
- **Khong walk-forward re-fit that** o sub-project 1/2/3 — MS-EGARCH duoc fit
  MOT LAN tren toan bo cua so, khong refit theo tung fold nhu mot walk-forward
  san xuat that su can. Sub-project 4 co xay walk-forward NHUNG chi o quy mo
  smoke (5-10 moc, ADVI rat nhe) — khong phai benchmark thong ke dang tin.
- **Nhan regime co the suy bien duoi config ADVI nhanh**: da quan sat o
  sub-project 2 — mot so cua so du lieu that cho ra >95% phien cung mot nhan
  regime khi ADVI chi chay 300 buoc. Can chay ADVI hoi tu tot hon (nhieu buoc/
  seed hon) truoc khi tin so lieu phan loai/rui ro la dai dien that.
- **Mot phan posterior MS-EGARCH nam ngoai vung dung (`|beta|<1`)**: prior chi
  la rang buoc mem (Normal(0.9, 0.15)), khong phai constraint cung — da xac
  nhan that qua debug sub-project 3 (37% draw/state trong mot fit thu nghiem
  khong stationary). Tang scenario-return da xu ly an toan (khong "no" so),
  nhung day van la dau hieu fit chua hoi tu tot, khong phai da giai quyet tan
  goc (giai phap dung constraint cung nhu `beta=tanh(beta_raw)` thuoc pham vi
  sub-project 1, chua lam).
- **Chua co luu/tai model (persistence)**: moi lan chay deu fit lai tu dau,
  khong serialize posterior ra dia.
- **Chua co CLI/pipeline orchestration**: chi chay duoc qua test script, chua
  co lenh don le kieu `raemf-mc predict --date today`.
- **Chi tan suat ngay (daily)**: khong co dac trung/mo hinh intraday, du du
  lieu 5 phut co san cho VNINDEX — xem ghi chu trong lich su brainstorm
  (`docs/superpowers/specs/`) ve ly do (regime kinh te khong van hanh o tan
  so 5 phut, va can xu ly seasonality trong ngay rieng neu mo rong sau nay).
- **Khong so sanh baseline**: quyet dinh chu dich tu sub-project 1 — MS-EGARCH
  la he thong moi hoan toan, benchmark chi danh gia noi tai (CRPS/WIS/coverage),
  khong so sanh "tot hon mo hinh cu" vi mo hinh cu chua ton tai.

## Cau truc thu muc chinh

```
src/raemf_mc/
  data/            # ingest + lam sach VNINDEX_Daily.csv (sub-project 1)
  features/        # log returns, dac trung nhan qua OHLCV (sub-project 1, 2)
  bayesian/        # ha tang ADVI dung chung + prior (sub-project 1)
  regime/          # MS-EGARCH, state alignment, posterior features (sub-project 1, 2)
  models/          # EBM classifier + calibration (sub-project 2)
  scenario/        # fit mu_k + mo phong Monte Carlo (sub-project 3, tren branch scenario-mc-var)
  risk/            # VaR/CVaR/drawdown (sub-project 3, tren branch scenario-mc-var)
  evaluation/      # (sub-project 4, dang lam, tren branch evaluation-model-card)
  validation/      # split, leakage checks (dung chung)
configs/           # cac profile ADVI (cpu_smoke, ebm_smoke, mc_smoke, eval_smoke, gpu_research-stub)
docs/
  superpowers/specs/    # thiet ke chi tiet tung sub-project
  superpowers/plans/    # implementation plan tung sub-project
  model_card.md         # (se co sau sub-project 4)
```
