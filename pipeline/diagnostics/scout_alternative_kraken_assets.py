"""
pipeline/diagnostics/scout_alternative_kraken_assets.py

Cheap, fast scouting check across a handful of candidate Kraken pairs,
BEFORE committing to building out a full pipeline for any of them --
same discipline as tonight's BTCUSD-vs-BTCUSDT and Coinbase-vs-Kraken
scouts (compare_btcusd_vs_btcusdt_density.py, compare_exchange_density.py).

Motivation: Ethan's question (2026-08-25 session, continued) -- rather
than keep testing BTC (five-plus independent methods have already
converged on "no detectable edge" there), is there a different asset
worth trying, ideally one more likely to show SOMETHING detectable?
Market inefficiency tends to track inversely with liquidity/institu-
tional attention -- BTC is the most heavily arbitraged crypto asset by
a wide margin, plausibly part of why this project keeps finding
nothing there. A lower-cap, still-liquid altcoin is a REASONABLE
HYPOTHESIS for "more likely to show something," not a proven fact --
this script tests that hypothesis empirically rather than guessing a
ticker.

Reports, for each candidate pair, over a SHORT window (default 6h, kept
short deliberately so this scout itself stays fast):
  1. Trade density (trades/hour) -- must stay reasonably liquid, or we
     reintroduce the exact density problem tonight's whole Kraken
     evaluation was about solving.
  2. Raw lag-1 autocorrelation on SIMPLE TIME BARS (5-minute resample --
     NOT dollar bars, NOT CUSUM-filtered, NOT triple-barrier-labeled --
     this is a quick, rough diagnostic signal, not the real book-
     faithful pipeline). A materially higher |autocorr| than BTC's own
     established near-zero/random-walk character (Ch13's phi_hat~1.03
     finding) would be a real, if crude, first signal worth following
     up on with the real pipeline -- not proof of anything by itself.

Reuses ingestion_kraken.pull_recent_trades_kraken() directly -- no new
ingestion code, same schema, same pagination/disambiguation fixes
already tested tonight.

Usage
-----
    conda activate mlfinlab
    cd C:\\ws\\AFML
    python pipeline\\diagnostics\\scout_alternative_kraken_assets.py
    python pipeline\\diagnostics\\scout_alternative_kraken_assets.py --hours 12
"""
import argparse
import os
import sys

import numpy as np
import pandas as pd

HERE = os.path.dirname(os.path.abspath(__file__))
ORCH = os.path.join(HERE, '..', 'orchestration')
sys.path.insert(0, ORCH)

from ingestion_kraken import pull_recent_trades_kraken   # noqa: E402

# Candidate pairs: a spread from "probably still quite efficient" (ETH,
# the next-most-institutionally-followed asset after BTC) down to
# "smaller, less arbitraged" -- deliberately NOT assuming any one of
# these is the right answer; that's what this scout is for. Kraken's
# pair-alias format matches BTC's own (asset + USD, e.g. 'ETHUSD') --
# if any of these aliases are stale/delisted by the time this runs, the
# per-pair error handling below reports that clearly rather than
# crashing the whole scout.
CANDIDATE_PAIRS = ['ETHUSD', 'SOLUSD', 'XRPUSD', 'LTCUSD', 'ADAUSD', 'DOGEUSD']

BTC_REFERENCE_AUTOCORR_NOTE = (
    "BTC's own established character: Ch13's phi_hat~1.03, consistent "
    "with a random walk (near-zero meaningful autocorrelation) -- "
    "compare candidates against that baseline, not zero in the abstract."
)


def scout_one_pair(pair, hours):
    raw_trades = pull_recent_trades_kraken(pair, hours)
    n_trades = len(raw_trades)
    span_hours = (
        raw_trades['Timestamp'].max() - raw_trades['Timestamp'].min()
    ) / 1_000_000 / 3600.0
    rate = n_trades / span_hours if span_hours > 0 else float('nan')

    # Quick time-bar resample (NOT dollar bars) purely for this scout's
    # own rough autocorrelation check -- see module docstring.
    ts = pd.to_datetime(raw_trades['Timestamp'], unit='us')
    price_series = pd.Series(raw_trades['Price'].values, index=ts)
    bars_5min = price_series.resample('5min').last().dropna()
    returns = bars_5min.pct_change().dropna()

    if len(returns) < 10:
        autocorr = float('nan')
    else:
        autocorr = returns.autocorr(lag=1)

    return {
        'pair': pair, 'n_trades': n_trades, 'trades_per_hour': rate,
        'n_5min_bars': len(bars_5min), 'lag1_autocorr': autocorr,
    }


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--hours', type=float, default=6.0,
                         help='Short window, kept small so this scout '
                              'itself stays fast.')
    parser.add_argument('--pairs', type=str, default=None,
                         help='Comma-separated subset of CANDIDATE_PAIRS '
                              'to pull (e.g. "ETHUSD,SOLUSD,XRPUSD") -- '
                              'for a longer confirmatory pull on just the '
                              'top candidates from a first, shorter scout, '
                              'without re-pulling all six. Defaults to '
                              'all of CANDIDATE_PAIRS if not given.')
    args = parser.parse_args()
    pairs_to_scout = (
        [p.strip() for p in args.pairs.split(',')] if args.pairs
        else CANDIDATE_PAIRS
    )

    print('=' * 70)
    print(f'SCOUTING alternative Kraken assets: last {args.hours}h '
          f'({", ".join(pairs_to_scout)})')
    print('=' * 70)
    print(f'\n{BTC_REFERENCE_AUTOCORR_NOTE}\n')

    results = []
    for pair in pairs_to_scout:
        print(f'Pulling {pair}...')
        try:
            r = scout_one_pair(pair, args.hours)
            results.append(r)
            print(f"  {r['n_trades']} trades ({r['trades_per_hour']:.1f}/hour), "
                  f"{r['n_5min_bars']} five-min bars, "
                  f"lag-1 autocorr={r['lag1_autocorr']:+.4f}")
        except Exception as e:
            print(f'  FAILED: {type(e).__name__}: {e}')
            results.append({'pair': pair, 'n_trades': None,
                             'trades_per_hour': None, 'n_5min_bars': None,
                             'lag1_autocorr': None})

    df = pd.DataFrame(results)
    suffix = '_'.join(pairs_to_scout).lower() if args.pairs else 'all'
    out_path = os.path.join(HERE, f'kraken_asset_scout_{int(args.hours)}h_{suffix}.csv')
    df.to_csv(out_path, index=False)

    print('\n' + '=' * 70)
    print('SUMMARY (sorted by |lag-1 autocorr|, most interesting first)')
    print('=' * 70)
    valid = df.dropna(subset=['lag1_autocorr']).copy()
    if len(valid) > 0:
        valid['abs_autocorr'] = valid['lag1_autocorr'].abs()
        valid = valid.sort_values('abs_autocorr', ascending=False)
        print(valid[['pair', 'trades_per_hour', 'lag1_autocorr']].to_string(index=False))
    else:
        print('No valid results -- check pair names / API errors above.')

    print(f'\nFull results written to {out_path}')
    print("""
INTERPRETATION:
  - This is a SINGLE short pull per pair, simple time bars, not the real
    dollar-bar/CUSUM/triple-barrier pipeline -- a rough first filter,
    not a finding. Treat a candidate as "worth a real pipeline build"
    only if BOTH density is reasonable (compare against tonight's real
    BTC numbers: Binance.US ~512-2325/hour, Kraken BTC ~1900-4100/hour)
    AND autocorrelation is meaningfully above BTC's own near-zero
    character -- not on autocorrelation alone, which is noisy at this
    short a window and could easily be a one-off fluctuation.
  - A second, longer pull (matching this scout's --hours to something
    larger) before fully trusting any single candidate's autocorrelation
    reading would be a reasonable next check, same caution as every
    other single-pull measurement in this project's diagnostics.
""")


if __name__ == '__main__':
    main()
