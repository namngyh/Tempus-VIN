# RAEMF-VB-MC

Mo hinh du bao xac suat che do thi truong (Bull/Sideway/Bear/Stress) va phan phoi
loi suat VN-Index, dung MS-EGARCH (Markov-Switching EGARCH 4 trang thai) uoc luong
toan phan bang Variational Bayes (ADVI) lam loi tang regime + risk.

Xem `docs/superpowers/specs/` cho thiet ke chi tiet va `docs/ms_egarch_design_decisions.md`
cho cac quyet dinh thiet ke con mo.

## Cai dat

Repo dung truc tiep Python 3.11 he thong hien co (torch, numpy, pandas, scipy,
pytest, scikit-learn, statsmodels, pyarrow da co san). Cai package o che do
editable:

```
pip install -e ".[dev]"
```

## Chay test

```
python -m pytest -q
python -m ruff check src tests
```
