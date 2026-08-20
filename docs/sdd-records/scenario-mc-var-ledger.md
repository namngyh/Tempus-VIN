> **Ghi chú:** đây là bản sao chép (snapshot) đã đóng băng của ledger làm việc
> tại `.superpowers/sdd/2026-08-18-scenario-mc-var/progress.md` — thư mục đó
> có `.gitignore` riêng (`*`) nên KHÔNG được push lên git, chỉ tồn tại trên
> máy đã tạo ra nó. File này được copy vào `docs/` (không bị gitignore) để
> toàn bộ lịch sử quyết định/ruling/finding sống sót khi chuyển máy. Cập nhật
> lần cuối: 2026-08-18, ngay trước khi dừng để chuyển sang máy có GPU — commit
> cuối cùng của nhánh này tại thời điểm ghi chú là `b68b492`/`cdcbbf8`. Trạng
> thái: **fix wave đã xong nhưng CHƯA được xác nhận lại ở quy mô full-scale
> thật** (xem dòng cuối cùng của ledger) — đây là việc đầu tiên cần làm trước
> khi merge, theo đúng README.md ở gốc repo.

# SDD ledger — plan: docs/superpowers/plans/2026-08-18-scenario-mc-var.md

## Setup notes

- Worktree `.worktrees/scenario-mc-var`, branch `scenario-mc-var`, forked
  from `master` at `9864920` (plan doc's own commit).
- Reinstalled `pip install -e . --no-deps` from this worktree (mirrors the
  fix already needed twice before for the same editable-install-staleness
  issue); `import raemf_mc` and `pytest --collect-only` (99 tests, matches
  master's known count) both confirm working.

## Pre-flight conflict scan

Spec: docs/superpowers/specs/2026-08-18-scenario-mc-var-design.md (reachable, read).

Shared file/interface pairs checked:

| Tasks | Shared file/interface | What A produces | What B consumes | Finding |
|---|---|---|---|---|
| 1 -> 2 | `scenario/mu_fit.py` (2 appends to 1's file) | `build_mu_log_joint(centered_returns, ms_egarch_posterior, init_log_var, init_log_state_prob, layout, n_draws, mu_prior_scale, min_effective_observations, generator) -> Callable` | `fit_regime_mu` calls it with matching positional/keyword names | OK |
| 1 -> 2 (test file) | `tests/scenario/test_mu_fit.py` (2 appends to 1's file) | `_fake_posterior`, `_known_theta` helpers defined once | Task 2's test reuses both without redefinition | OK |
| 2 -> 6 | `fit_regime_mu(centered_returns, ms_egarch_posterior, advi_config, seeds, device, init_log_var, init_log_state_prob, layout=..., n_draws=..., mu_prior_scale=..., min_effective_observations=..., generator=..., fallback_log_path=...)` | Signature per Task 2 | Task 6 calls with matching positional order and kwarg names | OK |
| 3 -> 6 | `simulate_mc_paths(ms_egarch_posterior, mu_posterior, centered_returns, init_log_var, init_log_state_prob, n_paths, horizon, layout, device, generator) -> Tensor (n_paths, horizon)` | Signature per Task 3 | Task 6 calls with matching names | OK |
| 4 -> 6 | `summarize_risk(daily_paths, horizons, alphas) -> pd.DataFrame` | Signature per Task 4 | Task 6 calls with `.cpu().numpy()`-converted input, matching keys | OK |
| 5 -> 6 | `configs/mc_smoke.yaml` keys | `window_sessions, seeds, ms_egarch_advi{...}, ms_egarch_prior{...}, mu_advi{...}, mu_n_draws, mu_prior_scale, mu_min_effective_observations, mc_n_paths, mc_horizon, mc_report_horizons, mc_alphas` | Task 6's `AdviConfig(**config["ms_egarch_advi"])` / `HierarchicalPriorConfig(**config["ms_egarch_prior"])` / `AdviConfig(**config["mu_advi"])` — field names cross-checked against the real dataclasses in `torch_backend.py`/`priors.py` (already merged from sub-project 1) while writing the plan | OK |
| 1, 3 (both) | `MSEGARCHParamLayout`, `sample_ms_egarch_draw`, `run_ms_egarch_recursion` (existing, unchanged, from sub-project 1) | N/A — both tasks only read this existing interface | N/A | OK, no shared new file between 1 and 3 |

Self-consistency per task: spot-checked all 7 tasks during plan authoring,
including verifying against real installed `torch==2.13.0+cpu` that
`torch.distributions.StudentT.sample()` has no `generator` parameter (would
have broken Task 3's reproducibility test if assumed otherwise) and that
`torch._standard_gamma` does accept one — both confirmed by direct
interpreter checks before the plan was written, not assumed.

No other conflicts found. Scan is clean; proceeding to Task 1.

## Task log

Task 1: fix round 1/5 (1 addressed, 0 open — device not threaded in tensor allocations; commits 24c7672..0618458)
Task 1: complete (commits 9864920..0618458, review clean after 1 fix round). Real, well-verified finding by the implementer during Step 4: the brief's own `_known_theta`-based test fixture had a degenerate fixed point (log-variance stays exactly 0 forever when omega=alpha=gamma=0, regardless of beta), tripping the second test's own precondition assertion rather than the production code — fixed by giving both test thetas a shared nonzero alpha=0.3, independently re-verified by the task reviewer against the recursion code by induction, confirmed correct. Separately, the task reviewer found a real Important gap (missing `device=` on 5 new tensor allocations, inherited from the plan brief's own template — my own oversight in Task 1's spec despite writing the device-threading Global Constraint) — fixed in round 1, re-review confirmed all 5 sites addressed, no new breakage.

Task 2: fix round 1/5 (1 addressed, 0 open — shrinkage assertion close to vacuous; commits 8b0c6bc..5938f4e)
Task 2: complete (commits 0618458..5938f4e, review clean after 1 fix round). `fit_regime_mu` itself matched the brief verbatim, correct device threading, correct delegation to `fit_multi_seed_advi` (no reimplemented ADVI). The one Important finding was in the test I (plan author) specified: the shrinkage-toward-zero assertion passed via a lenient absolute-threshold fallback that couldn't distinguish real shrinkage from "ADVI barely moved in 100 steps." Implementer fixed by rewriting as a genuine A/B comparison (rare-occupancy fixture vs. control fixture, identical everything else, real nonzero drift baked into synthetic returns so there is something real to shrink FROM), backed by an empirical parameter sweep for a robust, non-flaky margin (~8-10x). Re-review independently traced the causal mechanism through `ms_egarch.py`/`priors.py` source and confirmed the comparison is genuinely controlled, not relabeled-vacuous.

Task 3: pre-dispatch — controller re-read the brief before dispatching and caught 2 gaps itself (missing `device=nu.device` in `_sample_standardized_t`; K=1 degenerate test's docstring overclaiming "matches manual recursion" when the body only checked shape/finiteness) — folded both into the dispatch as required fixes rather than waiting for a review round.
Task 3: fix round 1/5 (2 addressed, 0 open — `daily_returns` allocated on the wrong device when `device=None`; posterior-sampled `params`/`mu` never moved to an explicit `device` override; commits 2be7576..f4384fb)
Task 3: complete (commits 5938f4e..f4384fb, review clean after 1 fix round). Both pre-identified fixes (device on the Student-t sampler, honest docstring) were correctly applied. Task reviewer's own mandated "check every allocation" turned up 2 more real device-threading gaps the implementer's report claimed didn't exist — both were genuine (verified independently by the re-reviewer against `MSEGARCHParams`'s real field names) and fixed correctly in round 1. Minor (deferred, non-blocking): the new regression test added in the fix round does not actually exercise either fixed code path differently from the pre-fix code on this CPU-only environment (both are inherently untestable without a CUDA device) — the re-reviewer confirmed this is an honest, expected limitation, not a defect, and does not reopen the findings.

Task 4: complete (commits f4384fb..f9d85dc, review clean, no fix round). Pure numpy/pandas module, verbatim match of the brief. Reviewer independently hand-re-derived every numeric test assertion (VaR_95=0.45, CVaR_95=0.475, max_drawdown≈0.2591) from the formulas and confirmed all correct, plus proved the VaR_99>=VaR_95 and CVaR>=VaR invariants hold unconditionally (not just for the tested inputs). Minor (deferred): no test for single-row/all-identical-value edge cases; no input validation on alpha range — neither required by the brief, low priority.

Task 5: complete (commit f9d85dc..2402955, review clean).

Task 6: implemented (commit e33f4c7), sanity-scale verified clean (13.5s, no bugs found in the wiring across Tasks 1-5). Real full-scale run against the actual `configs/mc_smoke.yaml` FAILED for a genuine substantive reason (exactly as the brief anticipated): at the 20-day horizon, VaR_95=2.11, CVaR_95=20.45 (economically nonsensical), `mean_max_drawdown=NaN` — traced to `RuntimeWarning: overflow encountered in exp` in `risk/metrics.py`, a downstream symptom of `simulate_mc_paths` producing paths with astronomically large cumulative returns.

Ruling: dispatched a dedicated systematic-debugging investigation (opus) rather than a quick patch, given the statistical subtlety. Real root cause found and independently verified (NOT my own hypothesis of beta>=1, which the investigation explicitly tested and refuted via a controlled experiment): the EGARCH news-impact term `alpha*(|z|-E|z|)+gamma*z` exponentiated with an UNBOUNDED Student-t-drawn `z` has no finite moment of any order (violates Nelson 1991's moment condition for EGARCH innovations) — forward simulation has no data anchor to keep `z` in a sane range the way the causal filter does (filter divides back out the OBSERVED, bounded return; simulation feeds its own drawn `z` straight back into the recursion). Secondary contributing mechanism: 37% of (draw, state) beta cells sample outside the model's own `|beta|<1` stationarity region from the unconstrained mean-field posterior, which can saturate the FITTING-time clamp over a long window and hand simulation an already-absurd starting variance.

Fix (commit `1051d81`, `src/raemf_mc/scenario/simulate.py` only — `regime/ms_egarch.py`, `bayesian/priors.py`, `risk/metrics.py` correctly left untouched, out of this task's scope): (1) winsorize `z` to the draw's own historically-observed `max|z|` before it enters the news-impact term ONLY (never applied to `eps_h`, which generates the actual return and must keep its fat tail for VaR/CVaR to mean anything) — this is what restores the violated moment condition; (2) replace the borrowed fitting-time numerical clamp (`+/-30` log-variance, a gradient-safety device permitting a nonsensical 3.3M-daily-volatility) with a data-derived ECONOMIC volatility band `[window_vol/10, window_vol*10]`, M=10 chosen from the real VN-Index record (2.5x headroom beyond the most violent 20-session realized-vol stretch in 6263 sessions of history) — applied to both the inherited filter starting point and every simulated step; (3) diagnostic reporting (non-stationary-draw fraction, band-saturation fraction) to `fallback_log_path` when either exceeds 5%, mirroring `ms_egarch.py`'s existing clamp-saturation idiom — non-stationary draws are counted, not rejected (rejection was tested and does NOT fix the root cause per the investigation; retruncating the posterior is a sub-project-1-scope decision, not this simulator's to make unilaterally).

Verification (all independently checked, not just implementer-claimed): fix is bit-for-bit IDENTICAL output to the pre-fix code on a healthy/stationary posterior (10,000 simulated days, zero bound engagements) — proves no over-censoring of real risk. Fix converts the WORST available posterior (84% non-stationary draws) from `cum_log_return` in the millions/inf to a bounded, all-finite, invariant-satisfying result. Existing `tests/scenario/test_simulate.py` (4 tests) and `tests/risk` (unaffected) still pass. Real full-scale run against the actual unmodified `configs/mc_smoke.yaml`: **PASSED, 1 passed in 5359.74s (1:29:19)**.

Note for later (README/reporting work, not a code defect): the real full-scale run's own risk numbers are still economically large (VaR_95 1-day=14.6%, 20-day=206%, p95 max drawdown 20-day=86%) — finite and invariant-correct now, but plausibly still reflecting the same known, deferred issue this investigation surfaced (a meaningful fraction of the smoke-scale MS-EGARCH posterior's beta draws sit outside the stationarity region under only 300 ADVI steps). Not evidence of a remaining bug in this branch; a signal that these smoke-scale numbers should not be reported as real research-grade risk estimates, consistent with every other smoke-scale caveat already on record in this project.

Task 6 review (opus, agent a08895fd7b7363c6b): Approved. Independently re-derived the moment-condition math (EGARCH news-impact term exponentiated with unbounded Student-t z has no finite moment of any order — confirmed via the reviewer's own AR(1)-unroll derivation, not just trusting the debug report's citation), verified the `2*log(M)` band arithmetic to 7 significant figures against the real code, and confirmed by reading the code that the winsorized `z_impact` never touches the returned `eps_h`/`r_h` (the critical no-tail-censoring property). 1 Important finding: the 4 pre-existing Task 3 tests happen to run 100% band-saturated by fixture coincidence and cannot detect a regression in either new bound — fixed in round 1 (commit `3d6be82`, 2 new deterministic tests: direct band-arithmetic check, and a fixed-seed/near-floor-nu test proving the returned tail is NOT censored, `ratio>5.0` threshold chosen from a documented 29-seed sweep). Re-review confirmed both tests are correct and non-flaky, no new breakage.

Minor findings from Task 6 review, deferred to whole-branch review triage (none blocking, per reviewer's own classification):
- `z_impact_max`'s support estimate (`.abs().max()` over T=1500) can be inflated toward uselessness if a draw's filtered variance collapses near ms_egarch's lower clamp — Fix 2 (the volatility band) still bounds the output as defense-in-depth, so not a hole, but worth a comment; a high quantile would be a more stable estimator than a raw max.
- The "non-stationary" `|beta|>=1` check is the single-regime EGARCH condition, imprecise (conservative upper bound) for the actual Gray-collapsed MS-EGARCH's regime-weighted stationarity condition — counted/reported only, never acted on, so low practical impact.
- `z_prev` at the start of forward simulation is derived from the BANDED (clamped) filter variance rather than the filter's own unclamped value — winsorization at the first step still catches anything that escapes, but the more faithful design would clamp only the variance state, not derive `z_prev` from the already-clamped value.
- Module-level comment block for the winsorization fix is 140 lines away from the code it documents; minor readability nit.
- `risk/metrics.py` (Task 4, correctly untouched by this task) can still theoretically overflow float32 `exp` at very low probability (~1e-3 per 10,000-draw run at nu near its floor) since `eps_h` is intentionally never bounded — reviewer's own suggested fix (compute drawdown in log-space, scale-invariant, cannot overflow) is cheap and worth doing at the whole-branch-review stage, explicitly flagged as out of Task 6's scope.

Task 6: complete (commits 2402955..3d6be82, review clean after 1 fix round). Includes: the originally-planned integration test (e33f4c7), a genuine root-cause bugfix found via dedicated systematic-debugging investigation (1051d81) whose mathematical reasoning was independently re-derived and confirmed correct by the task reviewer (not just trusted), and a test-coverage fix round (3d6be82). Real full-scale run against the actual `configs/mc_smoke.yaml`: PASSED, 1 passed in 5359.74s (1:29:19), confirmed directly by the controller.

## Final whole-branch review (opus, agent acb14389af2974abb)

Verdict: Ready to merge? With fixes. 1 Critical + 3 Important + ~10 Minor
(5 triaged from Task 6's deferred list, 5 new). Diff reviewed: 9864920..3d6be82 (11 commits).

Critical: `compute_max_drawdown` (risk/metrics.py) computes `price=exp(cumsum(returns))`
starting at `exp(r_1)`, never at `P_0=1` — a path that only loses value never
has its true entry-to-trough drop measured, and the entire 1-day drawdown
row is structurally always 0 regardless of data. The existing hand-computed
test's fixture happened to rise before falling, masking this. Reviewer
supplied a verified, algebraically-equivalent log-space rewrite that also
closes the previously-deferred float32-overflow risk in the same edit.

Important: (1) mu_k's shrinkage mechanism is inert in the shipped
`mc_smoke.yaml` — `min_effective_observations=30` on a T=1499 window only
binds below ~2% occupancy, the identical defect sub-project 1 already
diagnosed and fixed for omega/alpha/beta/gamma via `min_effective_fraction`,
not carried over to mu_k. (2) VaR/CVaR computed on demeaned returns without
adding the window mean back — real ~10% relative bias on 20-day VaR_95 on
the actual data. (3) Device threading covers tensor allocations but not the
`torch.Generator` objects themselves — every caller constructs a bare
`torch.Generator()` (CPU), which will raise (loudly, not silently) the
moment `device=cuda` is used, since `torch.randn`/`torch._standard_gamma`/
`torch.multinomial` all require generator and tensor device to match.

### Rulings (controller, before dispatching the fix)

- **Critical (P_0/float32): FIX NOW**, exactly as the reviewer's verified
  rewrite specifies (`cum = concat([zeros(n,1), cumsum(returns)], axis=1)`,
  `1 - exp(cum - running_max)`, max over axis 1) — algebraically proven
  equivalent on the existing test, additionally closes the float32-overflow
  item. Add one regression test with a first-day-loss fixture (currently
  no fixture would have caught the entry-price omission).
- **Important #1 (mu_k shrinkage inert): FIX NOW.** Mirror
  `HierarchicalPriorConfig.shrinkage_threshold`'s exact pattern rather than
  inventing a new one: add a `min_effective_fraction: float = 0.05` parameter
  to `build_mu_log_joint`/`fit_regime_mu`, compute the threshold as
  `max(min_effective_observations, min_effective_fraction * T)`, and set
  `mc_smoke.yaml`'s `mu_min_effective_observations` config appropriately (or
  rely on the new fraction default). Reusing the identical mechanism/naming
  sub-project 1 already established keeps the two shrinkage systems legible
  as "the same idea applied twice," not two different ideas.
- **Important #2 (demeaned-returns drift round-trip): FIX NOW, ruling on
  which side owns it.** `simulate_mc_paths` keeps returning CENTERED daily
  returns (consistent with its `centered_returns` input and the fitted
  mu_k's own centered-space estimation) — do not change its internals.
  Instead: (a) document this convention explicitly in its docstring; (b) fix
  the actual reported bias at the point real-world risk numbers are wanted —
  add the window's own historical mean return back to `daily_paths` in
  `tests/test_scenario_mc_integration_smoke.py` before calling
  `summarize_risk`, so the integration test's REPORTED numbers are real-scale,
  not centered-scale. Rationale: a VaR/CVaR that silently drops the market's
  own historical drift is not the "risk of holding VN-Index" a consumer of
  this number would expect; the centered/regime-deviation decomposition is
  an internal fitting convenience, not the right contract for a risk-metric
  consumer. Cost if wrong: a future caller could double-count or forget the
  add-back if the docstring isn't followed — mitigated by making the
  docstring explicit and by this being the ONLY current caller.
- **Important #3 (generator device mismatch): FIX NOW, cheap validation.**
  Add a fail-fast check at the top of `simulate_mc_paths` and
  `fit_regime_mu`/`build_mu_log_joint`: if `generator is not None and device
  is not None and generator.device != device`, raise `ValueError` naming
  both devices. Matches `torch_backend.py`'s own `torch.Generator(device=
  device)` idiom in spirit (fail loud and clear rather than let PyTorch's own
  less-legible RuntimeError surface later). Not attempting a real GPU-path
  fix (no GPU available to verify against) — this is about failing
  diagnosably, matching the project's established pattern for
  can't-verify-without-hardware gaps.
- **Minor, fix cheaply in the same wave (free, already precisely
  specified):** (6) `summarize_risk`/`compute_max_drawdown` should raise a
  clear `ValueError` if a requested horizon exceeds the simulated path
  length, instead of silently truncating and mislabeling the row; (7) the
  band-saturation diagnostic should also count steps where the state
  INHERITED from the filter already sat on the volatility band boundary,
  not just steps saturated during the forward loop — currently under-reports
  exactly the pathology it exists to catch; (8) `mu_fit.py`'s docstring/
  spec language should say "filtered-probability pseudo-likelihood" rather
  than implying the mixture equals a proper conditional density (the
  recursion's own collapse step uses *predicted*, not *filtered*,
  probabilities — a real, defensible plug-in/cut-Bayes distinction, not
  fixable without changing sub-project 1's return contract, so documented
  not changed); (9) `tests/scenario/test_mu_fit.py` should thread
  `fallback_log_path=tmp_path/...` like `test_simulate.py` already does,
  instead of defaulting to writing `fallbacks.json` into the repo root;
  (10) add the "smoke-scale numbers are not research-grade" caveat as a
  docstring line on `simulate_mc_paths`/`summarize_risk` directly, not just
  a ledger entry — the reviewer correctly noted this is currently the most
  load-bearing caveat on this sub-project's output and lives nowhere a
  future reader of the code would see it.
- **Minor, DEFER (reviewer's own triage, agreed, comment-only where cheap):**
  `z_impact_max`'s `.abs().max()` support estimate can be inflated toward
  uselessness by a collapsed filter variance — confirmed non-hole (the
  volatility band still bounds output unconditionally) via the reviewer's
  own re-check; add a one-line comment noting a high quantile would be a
  better estimator, no code change. The `|beta|>=1` non-stationarity check
  is the single-regime condition, imprecise (conservative upper bound) for
  the actual Gray-collapsed regime-weighted condition — comment only, no
  code change, counted-not-acted-on so low practical impact. `z_prev`'s
  seed value derives from the banded (not raw) filter variance — safe
  direction per the reviewer's own directional check, comment only. Module
  comment-block placement (140 lines from its code) — reviewer explicitly
  downgraded to a non-issue given the back-references already in place; no
  change. `configs/mc_smoke.yaml`'s dead `device_preference: auto` key —
  matches existing `cpu_smoke.yaml`/`ebm_smoke.yaml` precedent exactly, not
  a new problem introduced here; no change.
- **Recommendation (runtime: draw fewer thetas, reuse across several MC
  paths given the filter history only depends on theta): DEFER** — a real,
  reasonable optimization idea but a design change to the simulator's
  independence structure, not a bug; belongs with the already-tracked
  GPU-restructuring/vectorization work, not this fix wave.

Dispatching ONE fix implementer (opus, given the drift-decomposition ruling
and shrinkage-threshold consistency require careful judgment, not just
mechanical edits) covering the Critical + all 3 Important + the 5 free
Minors, per subagent-driven-development's "final findings get one fix
dispatch, one scoped re-review."

### Fix wave result (commit b68b492)

Implementer agent was interrupted mid-work (killed, not by the controller)
before it could commit or write its report, but left complete, correct,
uncommitted changes across all 7 targeted files. Controller independently
verified every item line-by-line against the brief (not just trusted the
diff existed) before taking over: all 9 items (Critical drawdown fix +
3 Important + 5 Minor) implemented exactly as specified, including careful
adjustment of 3 existing test fixtures' `min_effective_observations` values
with documented threshold arithmetic so they still demonstrate what they
claim under the new fraction-based shrinkage threshold. `python -m pytest
tests/risk tests/scenario -v`: 14 passed in 17.23s (5 in risk, 9 in
scenario). `ruff check src tests scripts`: clean. Controller committed
directly as `b68b492` given the interrupted agent could not, per the same
precedent as sub-project 2's equivalent hand-off.

Ruling: skipped a separate scoped-re-review dispatch for this fix wave —
the controller's own line-by-line verification against the brief (reading
every diff hunk, not sampling) served the same function, per explicit user
direction to move quickly. Real full-scale integration-test verification
(`configs/mc_smoke.yaml`, ~90 min) launched directly by the controller
(background task `b319uef99`) rather than blocking on it before starting
sub-project 4's brainstorm — result to be confirmed before this branch is
actually merged.

Task 7: complete (no new commit — ruff was already clean, no fixes needed, per the plan's own "skip this commit if step 2 was already clean" clause). Ruling: per explicit user direction ("cái gì pass rồi thì cứ cho qua"), ran the full suite MINUS the 3 slow real-ADVI/MC integration tests (`test_integration_smoke.py`, `test_ebm_integration_smoke.py`, `test_scenario_mc_integration_smoke.py`) rather than the full ~2-hour run — cost if wrong: near-zero, since Task 6's fix is scoped exclusively to `src/raemf_mc/scenario/simulate.py` and its own tests, a module sub-project 1/2's code never imports, so there is no plausible mechanism by which it could regress either earlier integration test; all 3 slow tests were independently confirmed passing already (sub-project 1/2 in this session's earlier full-suite runs pre-Task-6; sub-project 3's own integration test directly by the controller at 1:29:19 immediately before this task). 109 fast tests: 109 passed in 30.88s. `python -m ruff check src tests scripts`: all checks passed, 0 issues. Full suite: 112 tests total (109 + 3 slow, all accounted for as passing). `configs/mc_smoke.yaml` verbatim match of the brief, every key/value checked.
