# AFML Handoff: ETF pipeline built, positive control v1 run, v1.1 ready

**Session date:** October 4, 2026 (continues the 10/2 handoff; work started Oct 3)
**One-line status:** The ETF pipeline pieces are built, tested and pushed. The BTC 20-window extension is null-consistent. The ETF positive control v1 did NOT meet its pre-registered bar, and the demeaned-world check (v1.1) did not produce a clean null either (false positives 26%, model null IC 0.028; the momentum oracle is clean at 0.002). Leftover IC is mostly vol_60. A label-level v1.2 design is proposed, not yet pre-registered.

---

## What got done

- **Account-holder email.** Finalized and sent to the boss; reply pending. Decision asked: BSMIF account vs a new Biola-held account.
- **Broker research (search excerpts, not full pages).** Alpaca lists a $30,000 minimum for business accounts. IBKR's docs conflict on fractional API orders (several pages say no; the Web API changelog mentions cashQty for stocks). Public.com ruled out for a university account (business accounts must be single-owner entities). Tradier unresolved.
- **QF-Portfolio Strategy 1.0.** First draft is a Google Doc (id 175l-vLXJM2qx4AZ2EhnWKV_pO0yjTvaQPFzcQICispw). Open lines remain: weights format, label definition (now decided, see below), numeric scale-up thresholds, ramp details.
- **K-1 check.** DBC, DBA and UUP all issue K-1s (Invesco notice, 2023). PDBC is a no-K-1 commodity alternative; none checked for DBA or UUP.
- **BTC window extension.** Windows 9 to 20 captured (about 20 months total with windows 1 to 8). Pooled 20-window result: mean per-window PBO 0.4956 vs null 0.4925, z = +0.07, p = 0.946. Null-consistent under the pre-registered rule.
- **Signed trade-flow test.** Built (13 tests, sandbox) and pre-registered. NOT yet run.
- **ETF pipeline, all tests-first and pushed unless noted:**
  - `pull_prices.py` + 49-ticker `universe.txt` (3 tests); dated price snapshot `prices_daily_asof_2026-10-02.csv` committed.
  - `panel_data.py`, `etf_features.py`, `etf_labels.py` (31 tests).
  - `panel_uniqueness.py` + `panel_time_decay.py` (22 tests).
  - `panel_purged_cv.py` (19 tests, mutation-checked), embargo 252 dates.
  - `effective_breadth.py` + `report_effective_breadth.py` (12 tests).
  - `positive_control_etf.py` (18 tests with the v1.1 `--demean` option; v1 had 16). Pytest for the 18 not yet run on Ethan's machine.
  - `etf_pipeline_design.md` records every decision.
- **Real-data check of features and labels** (structure only): 285,720 feature rows, 6,735 dates, first 1999-12-22; dataset 284,712 rows to 2026-09-02; labels balanced (0.4975); t1 exactly 21 rows ahead; full 48-ETF breadth from 2008-12-18. Three daily moves over 25% (GDX and EWZ March 2020, SLV 2026-01-30) look real.

## Decisions made this session

- Time decay (AFML 4.7) goes in the pipeline; expanding training window.
- Labels: 1 if the ETF's 21-day forward return beats the cross-sectional median, else 0. Five fixed features (mom_12_1, mom_3m, ret_5d, vol_60, trend_200), vol-adjusted and ranked each date. Minimum 10 ETFs per date.
- Embargo 252 dates. Baseline model: logistic regression on the five ranked features.
- Provisional confidence-ramp zero point 0.5363 (confidence = DSR x (1 - PBO)).
- Planted edge for the positive control: the mom_12_1 rank.

## Bugs found and corrected (mine)

- Confidence is DSR x (1 - PBO), not the DSR alone. T_effective = T_raw x tw_mean (effective independent labeled events), so my "years" conversion was wrong.
- I said per-ETF uniqueness "ignores overlap between ETFs"; it is book-faithful if labels use each ETF's own returns, but relative labels need checking against the shuffle null.
- I overstated what one semester of forward trading shows; history validation is where evidence comes from.
- A noise test failed by chance (a 4% false-positive rule); replaced by a simulation of the false-positive rate.
- The first calibration test of planted vs realized IC was too noisy; replaced with an all-dates estimator.
- The positive-control "no-edge" world was not no-edge (see findings).

## Key findings

- **Effective breadth:** 16.5 independent bets against the median (9.0 on raw returns). At a 21-day horizon that is about 198 bets per year. Implied IC needed for 50% power: 0.049 (full capture), 0.098 (if long-only captures half; TC = 0.5 is an illustration).
- **Positive control v1 (500 worlds x 6 IC levels):** power 21% / 40% / 81% / 99.6% / 100% at nominal IC 0.02 / 0.03 / 0.05 / 0.075 / 0.10, against a threshold of t = 4.3. False-positive rate of t >= 1.96 at no edge = 69.4% (needed 2% to 8%). **Primary bar NOT met.**
- **Cause (partly verified):** the bootstrap keeps each ETF's own average return (annualized -2.2% to +15.8%). In 40 fresh worlds the mom_12_1 rank had mean IC +0.020 with the averages kept and -0.003 with them removed. Not yet explained: why the model's null IC (0.057) exceeds the oracle's (0.024), and why realized planted IC runs about 1.4x nominal.
- **Capture ratio above 1 is not skill:** it includes the model exploiting those static differences.

## Next steps, in priority order

1. **v1.1 done (null not clean).** Still to do: paste `pytest test_positive_control_etf.py -v` (expect 18 passed) for embedding; commit `positive_control_etf_demeaned_results.csv` and the analysis text by explicit path. Then decide v1.2 (label-level null and plant), and the walk-forward CV variant.
2. **Paper orders: DONE.** `check_orders.py` showed SPY, QQQ and GLD all FILLED with fractional quantities (about $4.99 of each $5.00 order; avg prices 769.73, 749.49 and 380.72) and three open positions worth about $15.04. The three CANCELED rows are the first batch, cancelled by the early cleanup on 10/2, with no fills, so there is no duplicate batch. Not yet validated: whole-share mode, selling or rebalancing, shorting. Gap closed afterwards: `rebalance.py` (targets vs current holdings, sells first) was added with 19 tests, and a whole-share rounding bug in `size_orders` was fixed (0.29 x $100 at $1.00 bought 28 shares, not 29). Both need `pytest test_sizing.py test_rebalance.py -v` on Ethan's machine, then embed and commit by explicit path. Closing the test positions is optional.
3. **Boss reply on the account holder; account-opening doc due Tuesday Oct 6**, labeled by holder and broker. Paste the links in the work doc and tag the boss with an @-mention in a comment.
4. **Signed trade-flow test:** run `pytest test_signed_flow_predictability.py -v` in `pipeline\diagnostics`, commit the pre-registration first, then run once.
5. **If v1.1 is not clean:** walk-forward (past-only) CV variant as a new pre-registration. **If clean:** lever what-ifs (long/short vs long-only, horizon, fewer features), then a plan for the real-data test.
6. **Lower priority:** strategy-doc open lines, scale-up thresholds, `repo_structure.txt` regeneration (stale, missing `ch16/`), Area51/N16 pipeline (still unidentified), TAO thread.

## Repo state

- `origin/main` last verified by me at `474bbac` (price snapshot). Later pushes (effective breadth, positive-control harness and results) were not re-verified by me.
- Not yet committed or not confirmed: v1.1 files above; `positive_control_etf_results.csv` and its analysis text; the signed-flow files and pre-registration; windows 9 to 20 calibration CSVs (`kraken_target_bars_calibration_window*.csv`) and the 20-window pooled printout.
- The 12 new `raw_trades.parquet` files (about 350 MB) are intentionally not committed.
- API keys exist only as environment variables. Add files by explicit path, never `git add .`.
