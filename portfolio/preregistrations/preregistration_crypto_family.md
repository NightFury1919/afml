# Pre-registration: crypto cross-sectional edge test, three horizons

Status: **written 2026-10-10, finalized 2026-10-11. Nothing below has been run on returns.** It becomes binding when committed.
After the first held-out return is computed, nothing above the Deviations log may change.

Defaults proposed by Claude were accepted by Ethan on 2026-10-11.

## 1. Question

Can a model rank coins by future return, inside liquidity tiers, better than chance, and does any edge survive
trading costs? Three arms test three forecast horizons. They are one family: all three are reported, and no
fourth is added under this registration.

| Arm | Horizon | Features | Why it is there |
|---|---|---|---|
| A | 1 day | short set (section 5) | matches the challenge's daily rebalancing; most power, highest turnover |
| B | 5 days | short set (section 5) | one fifth of A's turnover, so a small edge has a chance to survive costs |
| C | 21 days | the five ETF features, windows unchanged | the same test the ETFs got; needs the long Binance history |

Each arm also has a **volatility-rule control** at the same horizon: no model, score = the within-tier trailing
volatility rank. An arm only counts if it beats its own control (the ETF check found a pooled "volatility signal"
that was mostly a class effect).

## 2. Data (frozen)

- **Long history, primary:** Binance spot USDT daily candles from `pull_binance_daily.py`:
  `binance_{prices,ohlcv}_daily_asof_2026-10-10.csv` and the manifest. 708 symbols, 2017-08-17 to 2026-09-30
  (the archive stops at the last complete month).
- **Confirmation, venue check:** Kraken USD daily candles from `pull_crypto_daily.py`:
  `crypto_{prices,ohlcv}_daily_asof_2026-10-10.csv`, 2024-10-20 to 2026-10-09.
- **File hashes (SHA-256, computed 2026-10-11):**
  - binance_prices_daily_asof_2026-10-10.csv: 55AE9E4C16B069142FBEFD75FA195E0FFE3FA79E60649110B0EAC7CFFDDC8DF3
  - binance_ohlcv_daily_asof_2026-10-10.csv: 64FC91D01911F4D53C27C478BEE113003715C5828345735215796B9F4C8DF4C3
  - binance_pairs_manifest_asof_2026-10-10.csv: 1061E9A13374D1055BE970E876E90912390E2D3D4A4D100611E36831D6BDF3EC
  - crypto_prices_daily_asof_2026-10-10.csv: 389F68183D7956963DA4F9EE0F37E16D84BA39E9CB2B2C04616A584A365FACA0
  - crypto_ohlcv_daily_asof_2026-10-10.csv: F5DF90611707DCE88F63C7332B78D12E4E48BC1396D441E1A964BDD6FC363F61
  - crypto_pairs_manifest_asof_2026-10-10.csv: 98C47E4BC57FA6D0CAFDB77EB097F551C73475AF7B667945C70FF9F7525B7481
- These exact files are used. No re-pull.
- **Returns:** daily log return of the close. A return that spans a gap in the data (a missing day or more) is set to
  missing, not computed across the gap. This matters for halted coins (FTT has a 311-day gap, CVC 154).

## 3. Universe (frozen, point in time)

A coin is in the universe **on date t** if all of these hold using only data up to and including t:
1. It has at least 260 observed days of history (the longest window in arm C is 252 days).
2. Its trailing 60-day median quote volume is at least **$1,000,000** USDT.
3. It has at most 3 missing days in the trailing 60 days.
4. It is not in the exclusion list below.

A date needs at least 30 eligible coins; a tier needs at least 10. Because eligibility is recomputed each date, coins
that later fell apart or were delisted are included while they traded. This removes most survivorship bias in the
Binance data. It does not remove all of it: the archive lists pairs Binance chose to list.

### 3a. Exclusion list
From the data and manifest:
- Stablecoin and fiat bases flagged by the downloader (21): AEUR, AUD, BUSD, DAI, EUR, EURI, FDUSD, GBP, PAX, RLUSD,
  SUSD, TUSD, USD1, USDC, USDE, USDP, USDSB, USDS, USTC, UST, XUSD.
- Four unflagged pairs with daily volatility under 0.5%: BFUSD, BKRW, KGST, U.
- LUNAUSDT: the ticker covers two different assets (Terra before the May 2022 collapse, a new chain after), so its
  series is not one asset. Three daily moves above 150% confirm it.
- Wrapped, staked and pegged copies of another coin, and gold (approved defaults, each checked against the manifest):
  WBTC, WBETH, BETH, BNSOL, PAXG, XAUT.

### 3b. Tiers
On each date the eligible coins are split into three tiers by trailing 60-day median quote volume (terciles,
recomputed daily). Every feature, label and portfolio is built inside a tier, as in the within-class ETF test.

### 3c. Delistings
A label that runs past a coin's last price uses that last price, so a delisting counts as the loss it was. A coin that
is only missing the end date (a halt, or the end of the data) gets no label. The return across any missing day is missing.

## 4. Split and fitting (frozen)

- **Development:** start of data to 2024-09-30. **Held-out:** 2024-10-01 to 2026-09-30. Training rows must have labels that
  resolve by 2024-09-30 (a purge, so no training label reaches into the held-out window).
- Models are fitted **once** on the development period and then frozen. No hyperparameter is tuned: logistic
  regression, C = 1, as in the ETF check.
- **Arms A and B** are judged on the held-out period only.
- **Arm C** has only about 35 independent 21-day periods in the held-out window, too few to judge alone. Its primary
  test is out-of-fold scoring over the full history with `PanelPurgedKFold` (5 folds, embargo 21 dates), as the ETF
  check did (its embargo was 252 dates, the longest feature window; here the 252-day features need 260 days of
  history per coin already, and the shorter embargo is a stated simplification). The held-out result is reported next to it, not instead of it.
- The held-out window is touched once, for all three arms together, after every choice here is committed.

## 5. Features and labels

- **Arm C:** the five frozen ETF features and windows (`mom_12_1`, `mom_3m`, `ret_5d`, `vol_60`, `trend_200`), ranked
  inside the tier each date. Label: forward 21-day return above the tier median.
- **Arms A and B (new, frozen here):** `ret_1d`, `ret_5d`, `mom_20_5` (return from t-20 to t-5), `vol_20`,
  `trend_50` (price over its 50-day mean, minus 1). Return features are divided by the coin's trailing 20-day volatility,
  then ranked inside the tier. Label: forward 1-day (A) or 5-day (B) return above the tier median.
- These short windows are a new variant, chosen in advance from the ETF design (same five roles, shorter windows). They
  are not tuned on any return.

## 6. Statistics

- **mean IC:** mean over (date, tier) of the Spearman correlation between score and forward excess return.
- **Active return:** each period, hold the top fifth of each tier (tiers weighted by size) minus the mean return of all
  eligible coins, dates spaced one horizon apart. Gross, then net of costs (section 7).
- **Intervals:** moving-block bootstrap, block = 5 periods, 5,000 resamples, seed 20261010.

## 7. Costs (frozen)

The account trades on Kraken, so costs are Kraken's. At the lowest volume tier (under $2.5K of 30-day volume, where a
$1,000 account sits) the spot fee is **0.40% maker and 0.80% taker per side** (Kraken Pro spot fee table, Tier 1,
$0+ of 30-day volume; Tier 2 starts at $2.5K; read and confirmed by Ethan on 2026-10-11). Tier 1 is used for all three
arms. It is exact for arm C, which trades a few times a month. For arms A and B it is conservative: fee tiers count
30-day volume on both buys and sells, so an account rebalancing every day or every five days would likely reach a
cheaper tier within the month (for example Tier 4, 0.20% maker and 0.35% taker, at about $25K a month). Using Tier 1
makes "tradable" harder to pass, not easier. Turnover is reported with every result so the cheaper-tier case can be
worked out afterward.

- **Primary schedule (decides "tradable"):** limit orders, so the maker fee plus a half-spread by volume tier:
  0.40% + 0.40% = **0.80%** (bottom tier), 0.40% + 0.15% = **0.55%** (middle), 0.40% + 0.05% = **0.45%** (top), per side.
- **Sensitivity (reported, never decides):** every trade pays the taker fee, same half-spreads: 1.20%, 0.95%, 0.85%.
- The half-spreads (0.40%, 0.15%, 0.05%) are **placeholders** until measured. Whether the account can use limit orders
  is a boss/execution question, not a tested fact.
- Long-only top-fifth minus the universe mean is the primary portfolio, rebalanced to equal weights each period; weight
  drift between rebalances is ignored. A long-short version is reported only if the real account can short.
- Turnover is reported with every result.
- **Expected consequence, stated before any result:** at these costs arm A cannot be tradable unless its gross edge is
  several times the 0.17% a day that an IC of 0.03 implies; B probably is not; C has the best chance because it trades
  least. A result of "edge found, not tradable" would be an expected outcome, not a failure of the test.

## 8. Pass rules (frozen), one family

Significance level for each rule: **0.05 / 3 = 0.0167, one-sided** (Bonferroni across the three arms). Each lower end is
the 1.67th percentile of the block-bootstrap means, i.e. a 98.33% one-sided interval.

For each arm X in {A, B, C}:
1. **edge_found:** the lower end of the mean-IC interval is above 0 AND mean IC is at least 0.01 AND the
   arm beats its volatility-rule control (paired interval for the IC difference has a lower end above 0).
2. **tradable:** edge_found AND the net-of-cost active return, primary (maker) schedule, has a lower interval end above 0.
   The taker-fee result is reported next to it but does not decide.
3. **kraken_confirmed:** for arms A and B with edge_found only. The frozen Binance-fitted model is scored on the Kraken
   panel with no refit, using the same features, tiers, top-fifth portfolio and primary (maker, Tier 1) cost schedule.
   Kraken universe: the Section 3 rules, except the trailing 60-day median dollar volume floor is **$100,000** (the
   $1,000,000 floor leaves 30 or more coins on only 17 of 461 dates). Excluded: the Kraken manifest's stablecoin and
   fiat bases, plus the base assets of the Section 3a exclusion list. Every date with 260 days of history is used
   (about 461 dates, 2025-07-06 to 2026-10-09, less the horizon). **Pass:** mean IC above 0 AND mean net active return
   above 0. These are point estimates; intervals are reported but do not decide. The Kraken window lies inside the
   Binance held-out window, so this checks the venue and quote currency, not a new time period.

Outcomes: no arm has edge_found, so no edge shown. edge_found without tradable means the edge exists but costs consume
it. tradable without kraken_confirmed means a venue-specific result, not yet usable on the account.

## 9. Power (honest)

Measured 2026-10-11, before any held-out return was scored. Method: the standard error of the mean IC is
1/sqrt(effective bets x periods), and the smallest detectable IC is (z_alpha + z_power) x that standard error, at 80%
power. The Bonferroni level is 0.05 / 3 one-sided (z = 2.13); the uncorrected 0.05 level (z = 1.64) is shown for
comparison. Effective bets come from the correlations of daily Binance returns minus the same-day median of the eligible
coins, for coins eligible on at least 80% of days from 2022-01-01 to 2024-09-30. **Only development-window returns were
used; the held-out window was not touched.** Breadth stops growing as coins are added: about 67 by participation ratio
(the conservative measure) or about 92 by Meucci ENB, against a median of 222 eligible coins on held-out dates. Both
are shown.

| Arm | Periods | Standard error (PR 67 / ENB 92) | Detectable mean IC at the Bonferroni level | At uncorrected 0.05 |
|---|---|---|---|---|
| A, held-out | 730 | 0.0045 / 0.0039 | **0.013** / 0.0115 | 0.011 / 0.010 |
| B, held-out | 146 | 0.0101 / 0.0086 | **0.030** / 0.026 | 0.025 / 0.022 |
| C, full-history purged CV | 118 | 0.0112 / 0.0096 | **0.033** / 0.029 | 0.028 / 0.024 |
| C, held-out alone | 35 | 0.0207 / 0.0176 | 0.061 / 0.052 | 0.051 / 0.044 |

The bold figures use the conservative breadth. The earlier draft figures (A about 0.011, B about 0.02, C about 0.02 to
0.023 with about 140 effective bets, C held-out about 0.04) were more optimistic; the 140 could not be reproduced. The
MIN_IC floor of 0.01 sits below what any arm can detect, so the interval rule, not the floor, will decide edge_found.
These assume the signal is independent from period to period and that breadth after 2022 holds earlier. A slow or
clustered signal needs more data. Failing means "no edge shown at this power", not "none exists". Arms B and C can only
detect a mean IC of about 0.03 or more, so a smaller real edge would most likely be missed.

## 10. Known weaknesses (accepted)

- Survivorship remains: only pairs Binance listed. Coins listed after a run-up enter late.
- 2020 to 2021 is a single bull cycle in the development data; the held-out window is a different regime.
- Binance pairs are USDT, not USD; a stablecoin move enters every return. USDT-peg stress (for example May 2022) is not
  removed.
- Half-spreads and slippage are placeholders. Small coins may not fill at the close. A $1,000 account may also hit
  Kraken's minimum order sizes on small coins, which the test does not model.
- Long-history coins are exchange-selected; exchange delistings tend to follow bad performance, which a point-in-time
  universe handles only while the coin still traded.
- Three arms on one dataset is three tries. The Bonferroni level in section 8 accounts for it.

## 11. Process rules

- Commit this file, then the code freeze, as separate commits. **Never chain the held-out run after a commit in the same
  command block** (the Oct 9 ETF run did, and the freeze did not hold).
- The held-out run happens once. A crash is fixed without looking at any result and rerun, and logged.
- Deviations are appended below, never edited above.

## Deviations log
(empty)


## Outcome (recorded 2026-10-12; appended below the Deviations log, nothing above it changed)

**The run.** Once, on 2026-10-12, with the code and pre-registration as committed (pre-registration `a170d2e` and
`b42b39a`, code `71057c0`). Before it ran, the six data files matched the hashes in section 2, the three code and
pre-registration files had no uncommitted changes, and no results file existed. No crash and no change to anything
frozen, so the Deviations log stays empty. The full printed output is at the bottom of `crypto_family.py`; the numbers are in
`crypto_family_results.csv`, with per-period series in `crypto_family_{A,B,C}_ic.csv` and `crypto_family_{A,B,C}_active.csv`.

**Results as the pre-registered rules decide them (section 8).**

| | Arm A (1 day) | Arm B (5 days) | Arm C (21 days) |
|---|---|---|---|
| Scope, periods | held-out, 729 | held-out, 145 | full-history purged CV, 109 |
| Mean IC (lower end) | 0.0810 (0.0694) | 0.0884 (0.0684) | 0.1346 (0.1049) |
| Volatility control mean IC | -0.0821 | -0.1140 | -0.1320 |
| Model minus control, lower end | 0.1399 | 0.1612 | 0.2036 |
| Gross active return per period | 0.11% | 0.48% | 2.21% |
| Net, primary schedule (lower end) | -0.33% (-0.39%) | -0.15% (-0.39%) | +1.66% (+0.77%) |
| Net, taker sensitivity (lower end) | -0.62% (-0.68%) | -0.57% (-0.80%) | +1.31% (+0.43%) |
| Mean turnover per period | 0.727 | 1.049 | 0.874 |
| **edge_found** | **True** | **True** | **True** |
| **tradable** | **False** | **False** | **True** |
| **kraken_confirmed** | **False** | **False** | not applicable |

Kraken check, arms A and B (frozen Binance-fitted model, no refit, $100,000 floor, 2025-07-06 onward, median 84 coins a
date): A has 460 periods, mean IC 0.0574 (lower end 0.0394), mean net active return -0.42%. B has 92 periods, mean IC
0.0663 (lower end 0.0326), mean net active return -0.52%. In both the IC is positive and costs make the net return
negative, so the rule fails on its second condition. Arm C, descriptive only: the held-out window alone has 34 periods and
mean IC 0.1764.

**What these verdicts do not show (noted at recording time, before any further analysis; the verdicts above stand).**
These are observations about the design and the numbers. None has been tested.

1. The volatility control is signed as frozen (score = volatility rank, high volatility ranks first). Each arm's model IC
   is about the mirror image of its control IC (A 0.081 against -0.082, B 0.088 against -0.114, C 0.135 against -0.132),
   so the "beats its control" test is passed by roughly twice the size of the effect. In size, the model is no better
   than a low-volatility sort. The rule was met as written; it did not test what section 1 says it was for.
2. Gross return is small for the IC. Section 7 says an IC of 0.03 implies about 0.17% a day; an IC of 0.081 would imply
   about 0.46% a day, and arm A's gross is 0.11%. Most of the rank information is not reaching the long-only top fifth.
3. Active return is computed from log returns and measured against the equal-weight mean of all eligible coins, so it is
   a relative return, not an absolute one. Averaging log returns penalises high-volatility coins by about half their
   variance, which could favour a low-volatility top fifth. The size of that effect is not measured.
4. The ICs are several times larger than the smallest detectable values in section 9 (about 0.013 for arm A at the
   corrected level) and larger than the 0.03 used as an example of a good model in section 7.

Any later analysis of these results is exploratory and will be recorded below, labelled as such, and does not change the
verdicts above. A pass here is not a claim that a profitable strategy exists: arm C's "tradable" is a net return relative to
the universe mean on Kraken's lowest fee tier, over 109 periods, with no test of absolute return.


## Exploratory (post-outcome; recorded 2026-10-12; not pre-registered, does not change any verdict above)

Scripts: `crypto_family_audit_AB.py` and `crypto_family_audit_C.py`. They use the same data, the frozen held-out model
(arms A and B) and the same out-of-fold scores (arm C), changing nothing frozen. Bootstrap lower ends below use 2,000
resamples (the frozen run used 5,000), so they are approximate. Nothing here was chosen to rescue or sink a verdict.

**1. Sign-matched control (score = low-volatility rank, so the control's sign is the one that predicts).**

| | Arm A | Arm B | Arm C |
|---|---|---|---|
| Model mean IC | 0.0810 | 0.0884 | 0.1346 |
| Low-volatility sort mean IC | 0.0821 | 0.1140 | 0.1320 |
| Model minus low-vol sort (paired lower end) | -0.0011 (-0.0120) | -0.0256 (-0.0461) | +0.0026 (-0.0057) |
| Net per period: model / low-vol sort | -0.33% / +0.02% | -0.15% / +0.41% | +1.66% / +1.76% |
| Rank correlation, model score vs low-vol rank | 0.71 | 0.77 | not computed |

In no arm does the model beat a plain low-volatility sort. In arm B the low-volatility sort is significantly better.
The frozen control's sign (high volatility first) made "beats its control" a test the model passed by the size of the
control's own negative IC (observation 1 above). Measured against the sign that predicts, the model adds nothing.

**2. Log against simple returns.** Re-measuring the same books with simple (not log) forward returns:

| | Arm A | Arm B | Arm C |
|---|---|---|---|
| Gross active return, log -> simple | 0.106% -> 0.046% | 0.482% -> 0.248% | 2.208% -> 1.319% |
| Net active return, simple | -0.39% | -0.38% | +0.77% (lower end -0.30%) |

Arm C's net return stays positive but its lower end no longer clears zero, so under simple returns arm C would not meet
the "tradable" rule. The frozen rule used log returns and its verdict stands as recorded; this shows the verdict depends
on that measurement choice. Observation 3 above was right in direction: the log measure flatters a low-volatility book.

**3. Absolute return of the long-only book (simple returns).** Arms A and B lose money in absolute terms: the book
averages -0.03% per day (A) and -0.22% per 5 days (B), against -0.08% and -0.47% for the equal-weight universe. It beats
the universe by losing less. Arm C averages +4.1% per 21 days (median -1.0%) against +2.8% (median -2.2%) for the
universe, so its average is carried by a minority of strongly positive periods.

**4. Long and short side.** Holding the bottom-ranked fifth instead, the universe beats it by more than the top fifth
beats the universe: A 0.093% against 0.046%, B 0.455% against 0.248%, C 2.0% against 1.3%. A long-only book can only
collect the smaller half, which may account for part of the gap in observation 2 above (ICs large, returns small).

**5. IC by year and tier.** Mean daily IC is positive in every calendar year and every tier in all three arms. Arm A by
year: 0.096 (2024), 0.087 (2025), 0.068 (2026); arm B: 0.061, 0.103, 0.084; arm C by year (non-overlapping periods): 0.095
to 0.192 over 2020-2026. By tier: arms A and B are strongest in the lowest-volume tier (A 0.099, 0.085, 0.059 for tiers
0, 1, 2; B 0.103, 0.106, 0.063) and arm C strongest in the highest (0.110, 0.134, 0.165 for tiers 0, 1, 2). The low-volatility
sort's IC is similarly positive in every year, so stability does not separate the model from the sort.

**What this does and does not say.** The cross-sectional ordering of coins by volatility (low first) has a stable,
positive rank correlation with next-period outperformance in this sample. The model's score is largely that ordering. This
does not show a tradable strategy: arms A and B lose after costs, and arm C's tradable verdict rests on log returns, a
relative return measure and a minority of large periods. No new hypothesis from this section has been tested on data not
already used here.
