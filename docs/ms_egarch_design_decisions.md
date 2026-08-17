# Quyet dinh thiet ke MS-EGARCH (a-d)

Ngay: 2026-08-17
Tham chieu: docs/superpowers/specs/2026-08-17-ms-egarch-foundation-design.md

Tai lieu nay ghi lai 4 quyet dinh thiet ke con mo, kem ly do va danh doi da
can nhac. Day la lua chon co lap luan rieng, khong phai ket luan da duoc
kiem chung boi tai lieu hoc thuat chuan muc cho truong hop EGARCH cu the -
se can bang chung OOS truoc khi khang dinh mo hinh tot hon.

## (a) Gop o level-space, khong phai log-space

**Lua chon:** gop tai `torch.logsumexp(log_filtered_prob + log_var)`, tuong
duong voi gop tren level-space (trung binh co trong so cua sigma^2 theo
tung trang thai) roi lay log, khong phai trung binh truc tiep tren
log-variance.

**Ly do:** gop level-space bao toan dung mot moment that cua phan phoi tron
- `E[sigma_t^2 | F_{t-1}] = sum_k P_k * sigma_k,t^2` theo dinh ly phuong
sai toan phan. Gop truc tiep tren log-variance khong phai mot moment that,
va do bat dang thuc Jensen (log la ham lom) se luon thien lech xuong so
voi gia tri dung - sai lech cang lon khi xac suat filtered cang phan tan
giua cac trang thai co bien dong chenh lech nhau nhieu, dung luc can phan
biet regime ro nhat. Ve tinh toan, `torch.logsumexp` cho ra chinh xac ket
qua nay ma khong can exponentiate tuong minh, on dinh so hoc hon.

**Danh doi:** khong co tai lieu hoc thuat chuan muc xac nhan lua chon nay
cho truong hop EGARCH cu the (Gray's paper goc la cho GARCH thuong, tren
level-space thuan, khong co de quy log). Da viet test suy bien K=1
(`test_degenerate_single_state_matches_reference_egarch`) de xac nhan
khong co sai khac khi khong con bat dinh giua cac trang thai.

## (b) Chuan hoa z[t-1] bang sigma_bar[t-1] da gop

**Lua chon:** `z[t-1] = eps[t-1] / sigma_bar[t-1]`, voi sigma_bar[t-1] lay
tu chinh gia tri gop o quyet dinh (a).

**Ly do:** `eps[t-1]` la mot so quan sat duoc duy nhat (da thuc su xay ra),
can mot sigma tham chieu duy nhat de chuan hoa. sigma_bar gop la lua chon
tu nhien nhat vi day chinh la gia tri duoc dung lam dau vao cho de quy
log-variance cua buoc ke tiep, giu toan bo he nhat quan noi bo.

**Danh doi:** Gray's paper goc khong co so hang bat doi xung nen khong co
tien le truc tiep cho lua chon nay trong boi canh multi-regime.

## (c) Bo c_k o tang scenario-return

**Lua chon:** khong giu he so nhan `c_k` rieng theo regime o tang
scenario-return (se trien khai o sub-project 3) - chi giu mu_k va nu_k,
vi MS-EGARCH da cung cap sigma_k,t rieng theo tung regime.

**Ly do:** giu them c_k lam he so nhan tu do khac len tren co nguy co
non-identifiability - hai tham so cung giai thich mot scale, ELBO co song
nui phang doc theo huong c_k * sigma, posterior mean-field co the trong
"tu tin" gia tao o tung bien trong khi thuc chat khong dinh danh duoc.

**Danh doi:** neu sau nay phat hien phan tan trong-regime khong duoc giai
thich het boi de quy EGARCH (vi du qua phan tich residual thuc te), co the
can them lai mot tham so scale bo sung - nhung chua co bang chung thuc
nghiem cho dieu nay tai thoi diem viet tai lieu, nen chua them.

## (d) Hai giai doan Bayesian tach biet, khong mot ELBO chung

**Lua chon:** fit posterior MS-EGARCH doc lap truoc (khong can scenario/
return labels); tang scenario-return (sub-project 3) se dieu kien theo
posterior draws cua MS-EGARCH thong qua `sample_ms_egarch_draw`.

**Ly do:** day la mot dang modular/"cut" Bayesian inference - cat duong
phan hoi nguoc tu mot mo-dun co the bi misspecify (tang scenario) sang
mo-dun truoc no (MS-EGARCH), mot ky thuat co co so thong ke chu khong chi
vi tien trien khai. Cung khop tu nhien voi viec tach sub-project 2 va
sub-project 3 theo quan ly pham vi da thong nhat, giup moi mo-dun test doc
lap duoc.

**Danh doi:** khong lan truyen bat dinh cua mu_k/nu_k nguoc lai vao fit
MS-EGARCH (dieu kien mot chieu) - chap nhan duoc vi day la xap xi pho bien
trong modular Bayes, va la doi lay duoc su don gian/de debug ro rang.

## He qua interface

`sample_ms_egarch_draw(posterior, layout, generator)` tra ve dung mot
`MSEGARCHParams` cho moi lan goi - noi dung ma sub-project 3 se dung de lay
"mot joint draw duy nhat cho moi Monte Carlo path, giu co dinh suot
horizon" (khong resample ngam trong vong lap simulate).
