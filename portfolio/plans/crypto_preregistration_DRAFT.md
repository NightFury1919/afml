# Crypto cross-sectional test: pre-registration (DRAFT)

Status: DRAFT, written 2026-10-10. Becomes binding only when committed.
Nothing in this document may be changed after the first held-out return is
computed. Any change after that point is logged as a deviation and the
result is reported as exploratory.

Items marked **[ETHAN]** are decisions Ethan makes before committing.
Items marked **[PROPOSED]** are Claude's suggested defaults.

## 1. Question

Does the AFML pipeline's signal rank Kraken USD coins by next-day return
better than chance, within the cross-section, after the market factor is
removed, and does that survive realistic trading costs?

Why this test: both ETF checks (pooled and within-class) found no
demonstrated edge. Crypto has far more independent bets (about 55 effective
for the 146-coin universe) so a null here is informative. Power study:
SE of mean rank-IC 0.0053 on the held-out period, 80%-power detectable
IC 0.013, power 1.00 at IC 0.03.

## 2. Data (frozen)

- Source: Kraken `/OHLC`, interval 1440, via `portfolio/pull_crypto_daily.py`.
- Files: `crypto_{prices,ohlcv}_daily_asof_2026-10-10.csv` and
  `crypto_pairs_manifest_asof_2026-10-10.csv`. These exact files are used.
  No re-pull. Record their SHA-256 hashes in this document when committing.
- Window: 2024-10-21 to 2026-10-09 (719 daily returns).
- Returns: daily log return of the close. The in-progress candle is already
  dropped by the downloader.

## 3. Universe rule (frozen)

A pair is in the universe if ALL of:
1. `stable_or_fiat_base` is False in the manifest.
2. Daily return volatility over its history is at least 0.5% (removes 16
   unflagged stablecoin-like pairs).
3. `median_dollar_volume` is at least $50,000 **[PROPOSED]**.
4. `n_rows` is at least 365 **[PROPOSED]**.
5. Not in the exclusion list in section 3a.

Result on the frozen files: about 146 coins before section 3a exclusions.

The universe is set once, from the full-sample manifest. Because the
liquidity and history filters use data from the held-out period, this adds a
mild look-ahead. It is accepted and recorded here so it is not a surprise
later. A stricter variant (filters computed on the development period only)
is the robustness check in section 8.

### 3a. Exclusions **[ETHAN]**

Proposed default: exclude wrapped or staked copies of another coin in the
universe, because they are the same bet counted twice, and exclude tokenized
commodities, because they are not crypto assets.

- Wrapped/staked: WBTC, TBTC, MSOL, JITOSOL, LSETH, METH, CMETH, LSSOL
  (confirm each against the manifest before committing).
- Tokenized gold: PAXG, XAUT.
- Review the full manifest for tokenized stocks or other pegged assets and
  add them here before committing.

## 4. Split (frozen)

- Development: 2024-10-21 to 2025-12-25 (431 days). Any exploration,
  parameter choice, or model fitting happens here only.
- Held-out: 2025-12-26 to 2026-10-09 (288 days). Touched once, at the end,
  after every choice is frozen and committed.
- Embargo of 5 days between the two sets **[PROPOSED]** so labels near the
  boundary do not leak.

## 5. Signal and statistic (frozen)

- Signal: the pipeline's predicted score for each coin at close of day t,
  using only data up to day t. The exact model, features, and training
  procedure are fixed in a separate commit before the held-out run.
  **[ETHAN]** decide whether the signal is the full AFML pipeline (triple
  barrier, meta-labeling, bet sizing) or a simpler rank signal.
- Target: next-day log return, with the cross-sectional mean removed each day.
- Statistic: mean over days of the daily Spearman rank correlation (IC)
  between signal and next-day demeaned return.
- Inference: one-sided test that mean IC is greater than 0. Standard error
  from a block bootstrap over days (block length 5) **[PROPOSED]**, because
  a real signal is not independent from day to day.

## 6. Cost model (frozen) **[ETHAN]**

- Proposed: 0.40% taker fee per side plus half of a spread estimate of
  0.10% to 0.50% depending on the coin's liquidity tier. Final numbers to be
  set from Kraken's current fee schedule before committing.
- Portfolio for the net test: long the top quintile, short the bottom
  quintile, equal weight, rebalanced daily. If shorting is not allowed
  for the real account, report the long-only top-quintile minus universe
  return as well.
- Turnover is measured and reported next to the result.

## 7. Pass / fail rules (frozen)

The model returns to consideration only if ALL hold on the held-out set:
1. Mean IC > 0 at one-sided p < 0.05 (block bootstrap).
2. Mean IC at least 0.01 **[PROPOSED]**, the smallest edge worth trading
   after costs.
3. Net-of-cost long-short return is positive, with a Deflated Sharpe Ratio
   above 0.95 counting every variant tried in the development period.
4. The result holds in the liquid-only variant (median volume at least
   $250,000, 65 coins) with the same sign.

Fail on any one is a fail. A fail is reported as "no edge demonstrated at
this power", not as proof that none exists.

## 8. Robustness checks (reported either way, cannot rescue a fail)

- Universe filters computed on the development period only.
- Liquid-only universe (section 7 rule 4).
- Excluding the 20 coins with the largest single-day move.
- IC by quarter, to show whether one period carries the result.

## 9. Known weaknesses (accepted)

- Survivorship: only coins currently listed on Kraken. Delisted coins are
  absent, which flatters any strategy that buys losers' rebounds and punishes
  momentum shorts. Direction of bias is not known in advance.
- Two years of data, one market regime.
- 232 pairs have older history that the daily endpoint will not return.
- Kraken-only prices; thin coins may not trade at the close price.
- Spread and slippage are estimated, not measured.

## 10. Process rules

- Commit this file and the signal-code freeze as separate commits before the
  held-out run. Never chain the held-out run after a commit in the same
  command block.
- The held-out run happens once. If it crashes, fix the crash without
  looking at any result and rerun; log it.
- Record deviations in a section appended below, never by editing above.

## Deviations log

(empty)
