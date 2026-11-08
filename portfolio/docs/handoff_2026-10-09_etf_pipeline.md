# AFML Handoff — ETF portfolio challenge: real-history checks, class budgets, HRP
**Session date:** October 9, 2026
**One-line status:** Both real-history checks (pooled and within-class) found **no demonstrated predictive edge**; the fallback is a transparent, non-predictive, class-diversified portfolio, and the open question is now how much risk the boss wants to take for the 10x goal.

## Standing context (unchanged)
Boss's challenge: $1,000 start, $10,000 stretch, daily rebalancing, Biola as preferred account holder. Env `mlfinlab` (Python 3.10.20, pandas 1.5.3, numpy 1.23.5, sklearn 1.2.2); never run the pipeline in `base`. Alpaca paper keys live only in Windows user environment variables, never in files or chat; scripts use `paper=True`. Claude has no machine access; Ethan runs PowerShell. Commit by explicit path only.

## What got done
1. **Signed-flow predictability** finished: null result, outcome embedded, committed (c89eb2d; `signed_flow_run.txt` at the repo root).
2. **Positive-control harness, v1.2 within-class ranking** (26 tests). Frozen rule met: power 0.870 at nominal IC 0.03 (pooled 0.290), 1.000 at 0.05 (pooled 0.722); null false-positive rate 10% (pooled 26%).
3. **Portfolio construction** (`portfolio_construction.py`, 22 tests, acb7e99): select, weight, exposure, cap; supports class budgets.
4. **`update_prices.py`** (19 tests): restatement-safe price updates; real run created `prices_daily_asof_2026-10-08.csv` (8,481 rows, 49 tickers, old history unchanged).
5. **`paper_runner.py`** (17 tests) + `paper_protocol_v1.md`: dry run only, no trading imports; first plan `plans/plan_2026-10-08.json`.
6. **Pooled real-history check** (`real_history_check.py`, 14 tests): baseline indistinguishable from a volatility sort (tilt 0.885; current minus vol_rule IR [-0.09, +0.28]). Vol-scaled label over-corrected into a low-volatility tilt and earned nothing.
7. **Within-class real-history check** (`real_history_check_wc.py`, 17 tests, f1d06da / caf7d2b). Run once on the real snapshot; all six pre-registered rules False.

   | Arm | IR | 90% interval | vol tilt |
   |---|---|---|---|
   | wc_current | 0.18 | [-0.13, +0.53] | 0.57 |
   | wc_no_vol | -0.02 | [-0.32, +0.26] | -0.13 |
   | wc_vol_rule | 0.03 | [-0.27, +0.31] | 1.00 |

   Inside classes a pure volatility sort earns almost nothing (IR 0.03 vs 0.13 pooled), so the pooled "volatility signal" was largely a class effect. "Not shown", not "proved useless" (IR standard error about 0.2).
8. **Class budgets** (`class_budgets.py`, 13 tests, 26d2fd6): size, equal, inverse-vol budgets and a non-predictive portfolio (every ETF held, equal inside its class).
9. **HRP wrapper** (`hrp_allocation.py`, 10 tests; uncommitted): thin wrapper over the repo's existing `ch16/hrp/hrp.py`; `show_hrp.py` compares class shares.

## Real-data class shares (snapshot 2026-10-08, 48 ETFs, 252-day window)
| Class | size | equal | inverse-vol | HRP |
|---|---|---|---|---|
| US equity | 0.417 | 0.167 | 0.092 | 0.046 |
| International developed | 0.146 | 0.167 | 0.077 | 0.013 |
| Emerging markets | 0.125 | 0.167 | 0.046 | 0.006 |
| Bonds | 0.167 | 0.167 | 0.277 | 0.764 |
| Commodities | 0.083 | 0.167 | 0.047 | 0.018 |
| Currencies | 0.062 | 0.167 | 0.461 | 0.153 |

- **Inverse-vol puts 46% in currencies** (about 15% each in FXE, UUP, FXY).
- **HRP puts 76% in bonds, 48% in SHY alone.** Risk-parity methods favor the calmest assets. Both are capital-preservation portfolios and cannot plausibly reach 10x from $1,000.
- **Size budgets** (plain equal weight over all 48) are the neutral default: largest position 2.1%, 42% US equity.
- Conclusion: the stretch goal needs risk, which is a boss decision, not something a budget rule can settle.

## Process notes and bugs
- **Pre-registration freeze broken once** (within-class check): my command block cd'd into `portfolio` but used repo-root git paths, so the commit failed, and the run chained after it executed anyway. Script and pre-registration text were unchanged, but the first commit (f1d06da, 13:02:10) came after the run. Recorded in the outcome sections. **Rule going forward: never chain a pre-registered run after a commit in the same block; run only after the commit is confirmed.**
- **Duplicate work avoided late:** I asked for Chapter 16 snippets and wrote a second HRP before checking the repo; `ch16/hrp/hrp.py` already existed. Deleted my copy. **Check the repo and project docs (the book PDF is a project doc) before asking for snippets or writing chapter code.**
- Float-guard test used an exact case (0.2 x 15 = 3.0), so the mutation survived; fixed with 0.28 x 25. Several wc tests were strengthened after mutation checks (tilt within vs. across classes, ceil vs. floor, no-vol wiring, within-class IC).
- One mutation (class-neutral active return vs. universe mean) is equivalent, not a gap, when classes are weighted by size.
- Windows line-ending warnings (LF to CRLF) on commit are harmless.

## Open boss decisions
Account holder; benchmark; **acceptable drawdown versus the 10x goal (decide first)**; class budget rule (size / equal / inverse-vol / HRP); exposure ramp; per-ETF cap; long-only vs. long/short; daily vs. 21-day rebalance; loss limits.

## Outstanding / next steps
- [ ] Commit `hrp_allocation.py`, `test_hrp_allocation.py`, `show_hrp.py` by explicit path.
- [ ] Verify earlier work is committed: `update_prices.py`, `paper_runner.py`, `real_history_check.py` (and the plan JSON); `git log --oneline -- <file>`.
- [ ] Untracked, decide: `prices_daily.csv`, `prices_summary.csv` (likely gitignore), `handoff_2026-10-04_etf_positive_control.md`.
- [ ] Boss-facing one-page decision list (Google Doc), leading with the risk-vs-10x question and the four budget options.
- [ ] Read-only Alpaca holdings step; wire time decay (panel_time_decay) if a model returns; regenerate `repo_structure.txt`; strategy-doc open lines.
- [ ] No further model variants on this history (4 fitted variants already tried); any new model needs held-out data and a new registration.

## Next session
Start with the boss-decision page, then wire the chosen budget rule into `paper_runner.py` (new protocol version, dry run only).
