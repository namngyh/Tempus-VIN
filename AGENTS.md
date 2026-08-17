# Ky luat nghien cuu

- Khong bia so lieu — thanh phan khong uoc luong duoc phai ghi ro ly do.
- Khong dung test set de tuning, chon feature, chon calibration hay chon nguong.
- Voi moi horizon, train/validation phai duoc purge bang `target_end_date_h < boundary`.
- Khong silent fallback: moi lan ADVI fail/retry/rot xuong mean-field phai duoc ghi log.
- Khong float16 o bat ky dau trong ELBO, log-likelihood, log-sum-exp, quantile duoi.
- Tai lieu thiet ke va bao cao bang tieng Viet, trung lap, khong dua loi khuyen dau tu.
