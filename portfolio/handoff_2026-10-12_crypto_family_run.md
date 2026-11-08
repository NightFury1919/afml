# AFML Handoff — Crypto family test: freeze, Kraken confirmation, the run, outcome, exploratory audit
**Session date:** October 11-12, 2026 (the run and outcome were recorded on October 12)
**One-line status:** The pre-registered three-horizon crypto test ran once with no deviations. All three arms found a ranking signal (edge_found True); only the 21-day arm was tradable after costs; arms A and B were not Kraken-confirmed. A post-outcome audit (exploratory) shows the "edge" is mostly a low-volatility sort and that the 21-day tradable verdict does not survive switching from log to simple returns. Nothing here shows a strategy that makes money.

## Standing context (unchanged)
Boss's challenge: $1,000 start, $10,000 stretch, daily rebalancing, boss is the account holder. Env `mlfinlab` (Python 3.10.20, pandas 1.5.3, numpy 1.23.5, sklearn 1.2.2); never run the pipeline in `base`. Claude has no machine access to git; Ethan runs PowerShell. Commit by explicit path only. Never chain a pre-registered run after a commit in the same block. Claude can and should read and edit repo files through the device link instead of asking for pastes.

## What got done
1. **Freeze.** Pre-registration committed first (`a170d2e`, `b42b39a`), code after (`71057c0`). Costs confirmed by Ethan on 2026-10-11: Kraken Tier 1, 0.40% maker / 0.80% taker per side, plus placeholder half-spreads 0.40% / 0.15% / 0.05% (low/middle/top volume tier). Tier 1 kept for all arms (conservative for A and B, which would pay Tier 2/3 fees at volume). Kraken floor $100,000 trailing dollar volume (kept as is; $1M leaves 17 dates). Power numbers in section 9 were replaced with measured ones (PR 67 / ENB 92). Six data-file hashes verified before the run. `.gitignore` entries added at the repo root for the big CSVs.
2. **Kraken confirmation built before the freeze** (`kraken_confirmed`, rule 3, A and B only): frozen Binance-fitted model, no refit, $100,000 floor, 19 Kraken pairs excluded (flagged stablecoin/fiat plus Binance exclusion bases); passes if mean IC > 0 and mean net active return > 0 (point estimates). 41 tests (12 new Kraken tests); 41 passed in the sandbox and 41 in mlfinlab; 17 of 17 mutations caught.
3. **The run (once).** No crash, no change to anything frozen, Deviations log empty. Output embedded at the bottom of `crypto_family.py`; results in `crypto_family_results.csv`, `crypto_family_{A,B,C}_ic.csv`, `crypto_family_{A,B,C}_active.csv`. **Committed `13161fd`** (pre-registration with appended Outcome section, `crypto_family.py` with run output, seven CSVs).

   | | Arm A (1 day) | Arm B (5 days) | Arm C (21 days) |
   |---|---|---|---|
   | Scope, periods | held-out, 729 | held-out, 145 | full-history purged CV, 109 |
   | Mean IC (lower end) | 0.0810 (0.0694) | 0.0884 (0.0684) | 0.1346 (0.1049) |
   | Frozen volatility control mean IC | -0.0821 | -0.1140 | -0.1320 |
   | Net, maker schedule (lower end) | -0.33% (-0.39%) | -0.15% (-0.39%) | +1.66% (+0.77%) |
   | edge_found / tradable / kraken_confirmed | True / False / False | True / False / False | True / True / n.a. |

   Kraken check (A, B): mean IC positive (0.057, 0.066) but net active return negative (-0.42%, -0.52%), so rule 3 fails on its second condition.
4. **Exploratory audit** (labelled exploratory; `crypto_family_audit_AB.py`, `crypto_family_audit_C.py`; Ethan's rerun reproduced the numbers). **Committed `5510912`** with an "Exploratory (post-outcome)" section in the pre-registration:
   - **Sign-matched control:** model IC against a low-volatility sort: A 0.081 vs 0.082, B 0.088 vs 0.114 (sort significantly better, paired lower end -0.046), C 0.135 vs 0.132. Model scores are 0.71 / 0.77 rank-correlated with the low-vol rank (A / B). The model never beats the sort.
   - **Log vs simple returns:** gross active A 0.106% -> 0.046%, B 0.482% -> 0.248%, C 2.208% -> 1.319%. Arm C net falls from +1.66% to +0.77% and its lower end to about -0.30% (approximate, 2,000 resamples), so it would not pass "tradable".
   - **Absolute return:** long-only books in A and B lose money in absolute terms; C averages +4.1% per 21 days (median -1.0%) against +2.8% (median -2.2%) for the universe.
   - **Long vs short side:** the bottom-ranked fifth lags the universe by more than the top fifth leads it (A 0.093% vs 0.046%; B 0.455% vs 0.248%; C 2.0% vs 1.3%). A long-only book collects the smaller half.
   - **IC by year and tier:** positive in every year and tier in all arms. A and B strongest in the lowest-volume tier; C strongest in the highest.
5. **Google Doc "ETFs versus Crypto: Where to Look for an Edge"** updated: new "Crypto result (recorded October 12, 2026)" section before "Decisions needed"; Summary, comparison text and Next steps no longer say no crypto result exists; Next steps now recommend not adopting the model.
6. Side questions answered: other crypto brokers (Alpaca does crypto; fee tiers differ); Coinbase Advanced fee page was checked from a screenshot.

## Bugs found and process notes
- **Frozen control had the wrong sign for testing.** The control ranked high volatility first, so its IC was strongly negative and "model beats control" passed by roughly twice the effect size. The rule was met as written but did not test what it was meant to. Recorded in the Outcome section (observation 1) before the audit and confirmed by it.
- **Log-return active return flatters a low-volatility book** (variance drag). The frozen rule used log returns; the verdict stands as recorded, but arm C's "tradable" depends on that choice. Any future pre-registration should use simple returns for the active return and a sign-matched (or both-sign) control.
- **Pytest output piped into `Add-Content` on the files pytest was importing** caused a PermissionError and left a commented trace in both files; fixed by cutting the blocks and having pytest write to a temp file first. A `Set-Content` cut introduced a BOM and CRLF; rewritten as LF without BOM.
- **VS Code Overwrite mode (OVR)** ate text when hashes were pasted; restored from the draft.
- **Claimed the old Kraken-only draft did not exist** after listing only the top level; it is at `portfolio\plans\crypto_preregistration_DRAFT.md` (untracked, left as is).
- **`.gitignore` entries went into `portfolio/.gitignore`** (cwd was `portfolio`); moved to the repo root.
- **Device-bridge writes can be served stale when the same staged filename is reused** (happened again with the audit scripts: the device got the older sandbox-path version and Ethan's run failed with FileNotFoundError). Fix: write under a fresh staged filename, re-stage, verify by md5. Earlier clobbering of pre-registration edits was blamed on VS Code without proof; auto-save was already off.
- A test I wrote failed (exclusion test left 29 coins, under the 30 minimum); fixed with 37 coins. The empty-Kraken result now returns the same keys.
- PowerShell `*>` redirect writes UTF-16. `git status` output includes many unrelated `pipeline/` changes; explicit-path commits avoid them. LF-to-CRLF warnings on commit are harmless.
- Sandbox cannot reach exchange sites; data pulls run on Ethan's machine.

## Open decisions
- **Ethan / boss:** whether the exploratory findings (a stable low-volatility tilt; dependence on log vs simple returns) justify a new pre-registered test on data not yet used. Claude's recommendation (in the Google Doc): do not adopt the model; keep the risk-based portfolio unless a new pre-registered test passes.
- **Boss (unchanged since Oct 9):** acceptable drawdown vs 10x (decide first), benchmark, per-ETF cap, loss limits; class budget rule, exposure ramp, long-only vs long/short, rebalance frequency are Ethan's choices once the risk question is answered.
- Half-spreads are still placeholders (measured spreads would replace them in any new test).

## Outstanding / next steps
- [ ] Decide with the boss whether to design a new pre-registered crypto test (simple-return active return, sign-matched control, held-out data not already used); crypto_family.py refuses to run again because `crypto_family_results.csv` exists.
- [ ] Optional: pull Coinbase Advanced fee tier requirements and compare brokers if execution on Kraken is not realistic.
- [ ] Untracked and undecided: `portfolio/plans/crypto_preregistration_DRAFT.md`, `prices_daily.csv`, `prices_summary.csv`, `handoff_2026-10-04_etf_positive_control.md`.
- [ ] Carried over from Oct 9 (status not re-verified this session): commit `hrp_allocation.py`, `test_hrp_allocation.py`, `show_hrp.py`; verify `update_prices.py`, `paper_runner.py`, `real_history_check.py` are committed; regenerate `repo_structure.txt`.
- [ ] Wire the chosen ETF class-budget rule into `paper_runner.py` (new protocol version, dry run only) once the boss answers the risk question.

## Next session
Start by reading this file and `preregistration_crypto_family.md` (Outcome and Exploratory sections). Report results as "edge found by the frozen rules, mostly a low-volatility sort, not shown to be tradable"; never as "profitable".
