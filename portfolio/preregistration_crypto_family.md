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
