# Task 6 Debug Report: the real-scale integration failure

**Status:** (see final section)

**Skill used:** `superpowers:systematic-debugging` (invoked before any
investigation; Phase 1 evidence gathering completed before any fix was
proposed).

## 1. The failure being debugged

`tests/test_scenario_mc_integration_smoke.py`, run against the real,
unmodified `configs/mc_smoke.yaml` (1500-session VN-Index window, MS-EGARCH
ADVI 300 steps, mu_k ADVI 300 steps over 50 real MS-EGARCH draws, 500 MC
paths x 20-day horizon), ran 68 minutes and failed:

```
horizon   VaR_95    CVaR_95  ...  median_max_drawdown  p95_max_drawdown
1       0.146039   0.208108  ...                  0.0               0.0
20      2.114666  20.453289  ...                  NaN               NaN

>  assert 0.0 <= row["mean_max_drawdown"] <= 1.0
E  assert 0.0 <= np.float64(nan)

RuntimeWarning: overflow encountered in exp     (risk/metrics.py:25)
RuntimeWarning: invalid value encountered in subtract  (risk/metrics.py:27)
```

Reading the warnings carefully pins the immediate mechanism precisely:
`compute_max_drawdown` does `price = np.exp(cumsum(daily_returns))`. The
overflow is on the **positive** side — float32 `exp` overflows above ~88.7 —
so at least one simulated path had a cumulative 20-day log return above +88.7
(a price multiple of ~1e38). `price` became `+inf`, `running_max` became
`+inf`, and `(inf - inf)/inf` produced the `NaN` the assertion caught. The
metrics module is therefore a faithful reporter of a pathological input, not
the defect — as the task brief already stated.

## 2. Phase 1 — root-cause investigation

### 2.1 Reproduction

The 68-minute real run is not a usable debugging loop, so I built a cheaper
one, in three independent strands so no single strand had to carry the
conclusion:

1. **A deliberately under-converged but REAL fit** — real VN-Index data,
   `window_sessions=400`, ADVI `n_steps=80`, then the real
   `mc_n_paths=500 x mc_horizon=20`. Total ~9 minutes. This reproduced the
   failure in an exaggerated form on the first attempt.
2. **An instrumented copy of the `simulate_mc_paths` loop** (throwaway,
   uncommitted; the committed file was not touched during Phase 1) recording,
   per path: the sampled `omega/alpha/beta/gamma`, `nu`, `E|z|`, the terminal
   `log_var_bar` inherited from the filter, and then for every simulated day
   the realized state, the raw and clamped `log_var`, the Student-t shock and
   the resulting return.
3. **A parameter study of the forward recursion in isolation** — no fit, no
   posterior, 20,000 paths per parameter set — to establish which parameter
   regions explode and which do not.

### 2.2 What strand 1+2 showed (500 paths, under-converged real fit)

```
cum log return: min=-18025150   median=-0.3710   max=16025677
beta draws: min=0.020 max=2.026 mean=0.909
paths with beta>=1.0 in ANY state: 418/500
max simulated log_var per path: median=-3.63  p95=14.77  max=30.00
paths where the +/-30 clamp bound: 159/500
|eps| draws: median=0.306  p99=3.071  max=19.271
```

The five worst paths all had `init_log_var_bar` of **28.4 - 29.9** — i.e. the
log-variance was already sitting on ms_egarch's +30 clamp at the *end of the
filtering recursion*, before forward simulation took its first step. So a
first mechanism was confirmed: a sampled draw with `|beta_k| >= 1` compounds
over the 1500-session filtering window until it saturates the fitting-time
clamp, and `simulate_mc_paths` seeds the forward loop with
`history["log_var_bar"][-1]` — i.e. `sigma_0 = exp(15) = 3.3 million` — taken
at face value.

Confirmed directly (120 fresh draws, terminal filter state by stationarity):

```
STATIONARY (|beta|<1 all states, 17/120):
    terminal log_var_bar in [-8.60, -3.11]   clamp saturation 0.0000
NON-STATIONARY (103/120):
    terminal log_var_bar in [-11.69, 29.86]  clamp saturation up to 0.5865
```

### 2.3 The hypothesis in the brief was tested — and REFUTED as the root cause

The brief's hypothesis was that `beta > 1` sampled from the variational
posterior drives the explosion. `beta > 1` is real and common (37% of all
(path, state) beta cells in the cheap fit), and it does cause 2.2's inherited
saturated start. But it is **not** the root cause, and the fix cannot be
built on it. Experiment: reject every draw with `max_k |beta_k| >= 1` and
re-run 200 paths using only stationary draws.

```
per-draw acceptance rate ~ 0.144
paths with beta>=1.0 in ANY state: 0/200
cum log return: min=-76284.53   max=341.76
--- risk table with stationary-only draws ---
horizon   VaR_95     CVaR_95  ...  mean_max_drawdown
1       0.205966    0.299862  ...
20      3.922534 7636.690430  ...            NaN
```

**Still broken.** The worst path had `beta = [0.825, 0.766, 0.367, 0.654]` —
comfortably stationary — and `init_log_var_bar = -4.43`, a perfectly ordinary
starting variance. It exploded *during* the 20 simulated days:

```
day 12: z = -6.309
day 13: raw_log_var = 21.92   ->  sigma = exp(10.96) = 57,600  ->  r = -72,260
```

Attribution over the 155 pathological paths: `beta>=1` in 0 of them; median
`max|z|` 1.66 but max 13.3; median `max alpha` 1.62.

Also note, in the ORIGINAL failing run at real scale, `CVaR_95` at the 1-day
horizon was only 0.208. A path that inherited `sigma_0 = 3.3e6` would have
produced a day-1 return in the millions and blown that number up. It did not.
So at real scale essentially no path started saturated — the explosion is
built *inside* the 20 forward days. The inherited-saturated-start mechanism
is real but secondary; the brief's hypothesis is a contributing factor, not
the root cause.

### 2.4 Strand 3 — the actual root cause, isolated

Forward EGARCH recursion alone, no fit, no Markov switching, no posterior;
`beta < 1` in every case; 20,000 paths x 20 days each; `x0` = the real
window's log-variance:

```
beta  alpha  gamma  nu    min cum20    max log_var  frac(cum<-88)
0.90  0.10   -0.05  8.0     -0.2812        -6.82      0.00000
0.95  0.15   -0.05  5.0     -0.3822        -6.16      0.00000
0.98  0.20   -0.10  4.0     -0.8249        -0.89      0.00000
0.99  0.30   -0.10  3.0     -117.4          8.96      0.00005
0.99  0.30   -0.10  2.1     -1.148e+06     30.00      0.00005
0.95  0.50   -0.10  2.1     -1.010e+06     30.00      0.00020
0.99  1.00   -0.10  2.3     -8.133e+06     30.00      0.00140
```

The failure is entirely governed by `alpha` and `nu`, with `beta` strictly
below 1 throughout. This is a known, exact result, not an empirical
coincidence:

> The forward EGARCH recursion contains `alpha * |z_h|`, and the volatility
> it implies is `exp(log_var/2)`. `E[exp(alpha*|z|)]` **does not exist** when
> `z` is Student-t: the t has power-law tails, so its moment generating
> function is infinite everywhere. The simulated conditional variance process
> therefore has **no finite moment of any order.**

This is exactly the moment condition Nelson (1991) imposes on EGARCH
innovations — he requires tails no heavier than exponential (GED with shape
>= 1) precisely so that `E[sigma_t^2]` exists. A Student-t innovation
violates it. The project's Global Constraints fix Student-t innovations
(shared `nu`, reused from the MS-EGARCH draw), so the *forward simulator* is
the layer that has to cope with it.

### 2.5 Why fitting is fine and only simulation breaks

The same formula is numerically safe inside `run_ms_egarch_recursion` because
the filter is anchored by data at every step: it divides the **observed**
return back out, `z_prev = returns[t] / sigma_bar_t`. Real VN-Index daily log
returns are bounded (max |r| = 0.0822 over the full 6263-session history,
consistent with HOSE's +/-7% daily price band on constituents), so `|z_t|`
stays in a normal range and `alpha*|z_t|` never detonates. Any theta that
does push the variance somewhere silly is punished immediately by the
likelihood.

Forward simulation has neither anchor. It **draws** `z` from the model's own
Student-t and feeds it straight back in. One tail draw — near-certain
somewhere in `n_paths * horizon = 10,000` draws once the fitted `nu` sits
near its 2.05 floor — multiplies the next day's variance by
`exp(alpha*|z|)`, and the `beta ~ 0.9` persistence then *holds* that level for
the rest of the path.

### 2.6 The third defect: the borrowed clamp cannot catch any of it

`simulate.py` reused `_LOG_VAR_CLAMP = 30.0` verbatim from `ms_egarch.py`.
That constant's own comment states its purpose: keep `exp(log_var)` away from
float32 underflow/overflow so ADVI never sees a NaN gradient. It is a
**gradient-safety** device. Reusing it as the only guard in forward simulation
is not conservative, it is vacuous: `log_var = 30` is a daily volatility of
`exp(15) = 3.3 million`, i.e. a several-million-percent daily move. The bound
can never catch anything economically wrong; it only prevents a single-step
`inf` — which is precisely why the pathology reached `risk/metrics.py` intact.

### 2.7 Root cause, stated

> `simulate_mc_paths` runs the EGARCH news-impact term
> `alpha*(|z| - E|z|) + gamma*z` on an **unbounded Student-t draw** and then
> exponentiates it, so the simulated conditional-variance process has no
> finite moments (Nelson 1991); and its only guard against the resulting
> divergence is a numerical clamp borrowed from the fitting recursion that
> permits a daily volatility of 3.3 million, so nothing stops a path's
> cumulative log return from reaching the hundreds and overflowing float32
> `exp()` downstream. Secondarily, because the ADVI mean-field Gaussian is
> unconstrained it leaks mass outside the model's own `|beta| < 1`
> stationarity region, and such a draw saturates the *filter* over a
> 1500-session window so forward simulation can also start from an
> already-absurd variance.

## 3. Phase 4 — the fix

All changes are in `src/raemf_mc/scenario/simulate.py` (Task 3's file, which
the brief designates as this bug's territory). `regime/ms_egarch.py`,
`bayesian/priors.py` and `risk/metrics.py` were **not** touched, and no test
assertion was weakened.

The borrowed `_LOG_VAR_CLAMP = 30.0` is removed and replaced by two bounds
that are about the simulated economics rather than about float32.

### Fix 1 — the news-impact term is not extrapolated past its identified range

`alpha*(|z| - E|z|) + gamma*z` is a **linear** function of the standardized
innovation, and it was identified on exactly one sample of standardized
innovations: the `z_t = returns[t] / sigma_bar_t` this draw's own filter
produced on the estimation window. Evaluating that linear fit at a `|z|`
several times beyond anything in that sample is unsupported extrapolation,
and per 2.4 it is the specific term that destroys the variance process's
moments. So `z` is winsorized, per draw, to
`max_t |returns[t] / sigma_bar_t|` before it enters the variance recursion.

Crucially this is **not** applied to `eps_h` where it generates the return
itself. The fat tail of the return distribution is the thing a VaR/CVaR model
exists to measure and must never be censored — only the variance **feedback
loop** is bounded. Bounding the news-impact input is what restores
`E[exp(alpha*|z|)] < inf`, i.e. it repairs exactly the moment condition that
2.4 identified as violated.

*Known trade-off, stated explicitly:* winsorizing the news-impact input makes
volatility respond slightly less than the unbounded model would to a shock
larger than any in the estimation window. Over a 20-day horizon a draw beyond
the max of a T=1500 sample occurs on roughly 1.3% of paths, so the effect is
small, and it is conservative in the correct direction only in the sense that
it removes an unbounded response — it is a mild *understatement* of that
particular feedback. There is no unbiased alternative: leaving the term
unbounded gives a variance process with no finite moments at all (2.4), so
"unbiased" is not on the menu.

### Fix 2 — an economic band on simulated volatility, replacing the numerical clamp

`sigma_h` is confined to `[vol_window / M, vol_window * M]`, where
`vol_window` is the realized volatility of the estimation window the draw was
fit on (so the bound is data-derived and rescales automatically with any
window or market — no absolute magic constant). Enforced in log-variance, the
half-width is `2*log(M)`.

`M = 10` is read off the VN-Index record, not picked:

```
full history (6263 sessions):  daily std 0.014446
  most volatile 20-session stretch: 0.058859/day  =  4.07x full-sample std
  calmest      20-session stretch: 0.001481/day  =  0.10x full-sample std
```

`M = 10` therefore leaves ~2.5x headroom beyond the most violent volatility
regime in the index's recorded history, and still admits the calmest one, so
it cannot censor any economically plausible scenario. A simulated day outside
that band is not a tail scenario — it is a numerical artifact.

The band is applied to the state **inherited from the filter** as well as to
every simulated step, which is what closes the secondary mechanism of 2.2:
a non-stationary draw that saturated ms_egarch's +30 clamp over the filtering
window can no longer hand forward simulation a `sigma_0` of 3.3 million.

### Fix 3 — the posterior's health is reported, not silently absorbed

Following the brief's suggestion and mirroring `ms_egarch.py`'s existing
`CLAMP_SATURATION_REPORT_FRACTION` idiom, `simulate_mc_paths` now counts
(a) sampled draws with `|beta_k| >= 1` (outside the model's stationarity
region) and (b) simulated steps where the volatility band bound, and appends
a diagnostic record to `fallback_log_path` when either exceeds 5%.

Non-stationary draws are deliberately **counted but not rejected**. Rejection
sampling onto the stationary region is defensible in principle, but 2.3 proves
it does not fix this bug, and silently retruncating the posterior is a
modelling decision that belongs to the MS-EGARCH sub-project's own review,
not to a simulator. Reporting keeps the information visible without making
that decision here.

### Signature / call-site changes

`simulate_mc_paths` gains two keyword arguments with defaults:
`vol_band_multiple: float = SIM_VOL_BAND_MULTIPLE` and
`fallback_log_path: str | Path = "fallbacks.json"` (the same default the rest
of the codebase uses). To keep the test suite hermetic, the four existing
`tests/scenario/test_simulate.py` tests and the integration test now pass
`fallback_log_path=tmp_path / ...` — purely additive, identical to what the
integration test already does for its two ADVI fits. **No assertion was
changed, removed or weakened anywhere.**

## 4. Verification

### 4.1 The four committed Task 3 tests still pass

```
$ python -m pytest tests/scenario/test_simulate.py -q
....                                                          [100%]
4 passed in 5.96s
$ python -m ruff check src tests
All checks passed!
```

### 4.2 The reproduction is fixed — on the WORST posterior available

Re-ran the exact reproduction from 2.2 (the deliberately under-converged real
fit: `alpha ~ 1.1`, `nu ~ 2.35`, 84% of draws non-stationary) through the
**real, fixed** `simulate_mc_paths` at the real Monte Carlo scale
(500 paths x 20 days), with numpy floating-point errors escalated to
exceptions (`np.errstate(over="raise", invalid="raise")`):

```
band = [-13.234, -4.023] -> sigma in [0.001338, 0.133758]  (window std 0.013376)
daily returns: all finite=True  min=-1.06594 max=0.79721
cum 20d log return: min=-4.3659  median=-0.3426  max=3.3478

horizon   VaR_95   CVaR_95   VaR_99   CVaR_99  mean_mdd  median_mdd  p95_mdd
1       0.210572  0.293161  0.300839 0.454340  0.000000    0.00000  0.000000
20      2.713428  3.162596  3.458925 3.780581  0.477511    0.45016  0.926574

h=1 : VaR_99>=VaR_95 CVaR_95>=VaR_95 CVaR_99>=VaR_99 0<=mdd<=1 mdd finite  -> all True
h=20: VaR_99>=VaR_95 CVaR_95>=VaR_95 CVaR_99>=VaR_99 0<=mdd<=1 mdd finite  -> all True
ALL INTEGRATION-TEST INVARIANTS HOLD: True
```

No overflow was raised, the cumulative log return went from `min=-1.8e7 /
max=+1.6e7` to `min=-4.37 / max=+3.35`, and every drawdown is finite and in
`[0, 1]`. The diagnostic record was written, honestly reporting what this
posterior is:

```json
{"reason": "mc_simulation_bounds_engaged",
 "nonstationary_draw_fraction": 0.836,
 "vol_band_saturation_fraction": 0.3666, ...}
```

`VaR_95(20d) = 2.71` is still economically large — but that is a truthful
readout of an 80-step ADVI fit with `alpha ~ 1.1` and `nu ~ 2.35`, and the
diagnostic says so. The simulator's job is to stop producing *impossible*
numbers, which it now does; it is not to disguise a bad fit.

### 4.3 The fix is bit-for-bit a no-op on a healthy posterior

The obvious risk with any bound is that it quietly censors real risk. Tested
directly: an economically healthy MS-EGARCH parameter set (`beta = 0.92`,
`alpha = 0.12`, `gamma = -0.06`, `nu = 8.05`, `omega` set so the log-variance
stationary mean equals the real window's), on the real 1500-session VN-Index
window, 500 paths x 20 days, run through both the **fixed**
`simulate_mc_paths` and a **verbatim copy of the pre-fix loop** driven off an
identically-seeded generator:

```
band=[-13.327,-4.117] -> sigma in [0.00128, 0.12764]   (window sigma=0.01276)

max |new-old| over all 10000 simulated returns: 0.000e+00
identical: True
new daily std=0.012828   old daily std=0.012828   (window daily std=0.012764)
band diagnostic written? False

horizon   VaR_95   CVaR_95   VaR_99   CVaR_99  mean_mdd  median_mdd  p95_mdd
1       0.020775  0.024997 0.027238  0.032029  0.000000    0.000000 0.000000
20      0.101174  0.134408 0.138058  0.188907  0.054876    0.049164 0.111861
```

**Bit-for-bit identical.** Neither the news-impact winsorization nor the
volatility band engaged once in 10,000 simulated days, and the diagnostic was
not triggered. The resulting risk numbers are economically sensible for
VN-Index (1-day VaR_95 = 2.1%, 20-day VaR_95 = 10.1%, mean 20-day max
drawdown 5.5%). The fix therefore engages *only* where the simulated process
is genuinely divergent, and changes nothing where it is not.

### 4.4 Wider unit-test surface

```
$ python -m pytest tests/scenario tests/risk -q
..........                                                    [100%]
10 passed in 10.41s
```

## 5. Real full-scale verification status

`python -m pytest tests/test_scenario_mc_integration_smoke.py -v -s` against
the real, unmodified `configs/mc_smoke.yaml` was launched as a background
process (log:
`...\scratchpad\task6_realscale.log`).
The previous run of this test took 68 minutes; this platform hard-backgrounds
any single Bash call at ~600s, so per the task's own instruction I did not
loop or vaguely wait on it.

**Status of that run: see section 6.**

Everything the real run exercises has been verified at reduced scale on real
VN-Index data, using the real `simulate_mc_paths`, at the real Monte Carlo
scale (500 paths x 20 days), against both the worst posterior available
(4.2) and a healthy one (4.3). What remains unverified is only whether the
300-step ADVI fit on the full 1500-session window happens to land somewhere
the reduced-scale fits did not.

## 6. Outcome

**Status:** (filled in at the end of the session)

**Commit:** `1051d81` — *fix(scenario): bound the forward-simulated EGARCH
variance process*, on branch `scenario-mc-var`, on top of `e33f4c7`.

**Files changed**

| File | Change |
| --- | --- |
| `src/raemf_mc/scenario/simulate.py` | the fix (both bounds + diagnostics) |
| `tests/scenario/test_simulate.py` | `fallback_log_path=tmp_path` on 4 call sites; **no assertion touched** |
| `tests/test_scenario_mc_integration_smoke.py` | `fallback_log_path=tmp_path` on 1 call site; **no assertion touched** |

**Not touched:** `src/raemf_mc/regime/ms_egarch.py`,
`src/raemf_mc/bayesian/priors.py`, `src/raemf_mc/risk/metrics.py`,
`configs/mc_smoke.yaml` — all out of scope per the brief, and none of them
needed to change.

**Follow-up worth raising with the controller (not fixed here, out of
scope):** the diagnostics this fix adds exist because the underlying
MS-EGARCH variational posterior leaks a great deal of mass outside its own
model's `|beta| < 1` stationarity region — 37% of `(draw, state)` beta cells
in the reduced-scale fit. That is a property of an unconstrained mean-field
Gaussian over an unconstrained `beta`, and addressing it (a constrained
parametrization such as `beta = tanh(beta_raw)`, or a tighter prior) belongs
to sub-project 1's own review, not here. The simulator now reports it instead
of being destroyed by it.

## Fix round 1 (test coverage)

**Reviewer finding addressed (Important):** no committed test exercised
either of the fix's two bounds, and the four pre-existing tests in
`tests/scenario/test_simulate.py` were shown to be provably blind to them
(they assert only shape, finiteness, generator reproducibility, and device
equality — identical results whether the band half-width is `2*log(M)`,
`log(M)`, or absent entirely). The reviewer specified exactly two tests that
would close the gap; both were added to `tests/scenario/test_simulate.py`,
no other file touched.

### Test 1 — `test_simulation_log_var_band_arithmetic`

Calls the public, pure `simulation_log_var_band` (`simulate.py:126`)
directly, with no simulation involved. Uses `centered_returns =
torch.tensor([-1.0, 1.0])`, whose unbiased (ddof=1) variance is computable
by hand — `((-1)^2 + 1^2) / (2-1) = 2`, so `std = sqrt(2)` — deliberately
kept independent of torch's own `.var()` call so the expected values are not
circularly re-derived from the function under test. Asserts
`exp(0.5*log_var_lo) == pytest.approx(std / SIM_VOL_BAND_MULTIPLE)` and
`exp(0.5*log_var_hi) == pytest.approx(std * SIM_VOL_BAND_MULTIPLE)`. Passes
against the real `SIM_VOL_BAND_MULTIPLE = 10.0` with both bounds matching to
within float32 precision (`sigma_lo` 0.14142135 vs expected 0.14142135,
`sigma_hi` 14.142136 vs expected 14.142135).

### Test 2 — `test_simulate_mc_paths_does_not_censor_the_tail`

Verifies the specific invariant the review asked for: that `z_impact`
(winsorized to `z_impact_max`, used only inside the variance recursion's
news-impact term) and `eps_h` (drawn fresh from the Student-t each step,
never winsorized, the only term that feeds the *returned* value) are
genuinely different — i.e. that the returned daily log returns are not
capped at the volatility band's upper bound.

Fixture: `beta_raw = -0.7` in every state (comfortably stationary, so this
exercises genuine fat-tailed sampling rather than the non-stationary/band-
saturation path the debug report already covers) and `nu_raw = -4.0`, which
via `nu_from_raw`'s `min_nu + softplus(nu_raw)` (`min_nu=2.05`) gives `nu ~=
2.068` — close to the model's floor, where the Student-t tail is heaviest.
`omega = alpha = gamma = 0` (only `beta` and `nu_raw` set), matching the
existing fixtures' style.

Parameter/seed selection was swept empirically before committing to a fixed
choice, per the same rigor as Task 2's shrinkage A/B test: ran the fixture
(`n_paths=200, horizon=20`) through `simulate_mc_paths` for generator seeds
1 through 29 and recorded `max|r_h| / sigma_hi` for each. Every single seed
in that range produced a ratio between 5.97 and 68.4 — i.e. the effect is
not a lucky draw, it is the typical outcome at this `nu`. Seed `1` (ratio
~8.2) was fixed into the committed test, comfortably above 1 without being
the single most extreme seed found, and the test asserts `ratio > 5.0` (a
threshold every swept seed clears with margin to spare, so the test is not
flaky).

Raw sweep output (seed: ratio), for the record:

```
1: 8.23   2: 8.02   3: 5.97   4: 10.45   5: 15.27   6: 19.97   7: 8.24
8: 7.21   9: 68.44  10: 9.52  11: 18.11  12: 26.41  13: 6.62  14: 7.71
15: 7.76  16: 9.92  17: 12.35 18: 7.74  19: 7.90  20: 15.22  21: 7.51
22: 10.37 23: 8.01  24: 13.20 25: 6.72  26: 14.20  27: 7.62  28: 23.19
29: 8.82
```

### Verification

```
$ python -m pytest tests/scenario/test_simulate.py -v
....../6 passed in 8.81s (4 pre-existing + 2 new, all pass)
$ python -m ruff check src/raemf_mc/scenario/ tests/scenario/
All checks passed!
```

The 90-minute real-data integration test was **not** re-run for this fix
round, per instruction — it was already independently verified passing, and
this round only adds test coverage in `tests/scenario/test_simulate.py`; no
production code in `src/raemf_mc/scenario/simulate.py` changed.

**Out of scope, correctly not touched in this round** (reviewer's Minor
findings, belong to the whole-branch review): the non-stationary-detection
logic/comment, `z_prev` initialization, the module-level comment blocks'
organization, `src/raemf_mc/risk/metrics.py`, and `z_impact_max`'s
computation. None of these were changed.

**Commit:** `test: cover the volatility band arithmetic and tail-preservation
invariant` — touches only `tests/scenario/test_simulate.py`.

**Status: DONE.**
