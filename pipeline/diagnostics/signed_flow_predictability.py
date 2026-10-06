"""
pipeline/diagnostics/signed_flow_predictability.py

Pre-registered test of ONE hypothesis (see preregistration_signed_flow.md,
commit it BEFORE running this script):

    Signed trade-flow imbalance in one 4-hour block carries information about
    the log return of the NEXT 4-hour block.

What it computes, per 4-hour block (UTC clock boundaries):
    imbalance = (buy-initiated volume - sell-initiated volume) / total volume
    close     = price of the last trade in the block
    fwd_ret   = log(close of next block / close of this block)

Buy-initiated means IsBuyerMaker == False (the taker bought). That mapping is an
assumption documented in ingestion_kraken.py. If it is backwards, only the SIGN
of the correlation flips; the test is two-sided.

Statistic: pooled Spearman correlation between imbalance and fwd_ret over all
window-internal block pairs of the 7 Kraken windows, with t = rho*sqrt((n-2)/(1-rho^2)).
Pass rule (fixed in advance): |t| >= 1.96 AND the same sign in at least 5 of 7 windows.

Run once, from the repo root, in the mlfinlab env:
    python pipeline\\diagnostics\\signed_flow_predictability.py
It refuses to overwrite an existing results file (run-once rule).
"""
import os
import sys

import numpy as np
import pandas as pd

HERE = os.path.dirname(os.path.abspath(__file__))

BLOCK_HOURS = 4
MIN_TRADES_PER_BLOCK = 50
T_THRESHOLD = 1.96
MIN_SAME_SIGN_WINDOWS = 5
RESULTS_FILE = os.path.join(HERE, 'signed_flow_predictability_results.csv')

# The 7 disjoint 30-day Kraken windows 2 to 8 (amendment 1 to the pre-registration).
# Window 1 (kraken_snapshot_720h_2026-08-25, Jul 26 to Aug 25) is excluded: it overlaps window 2
# (Jul 11 to Aug 10) by about 15 days. Windows 9 to 20 are reserved for replication.
WINDOW_DIRS = [
    'kraken_snapshot_720h_window2_2026-09-08',     # window 2
    'kraken_snapshot_720h_window3_2026-09-10',
    'kraken_snapshot_720h_window4_2026-09-11',
    'kraken_snapshot_720h_window5_2026-09-10',
    'kraken_snapshot_720h_window6_2026-09-10',
    'kraken_snapshot_720h_window7_2026-09-11',
    'kraken_snapshot_720h_window8_2026-09-11',
]


def block_table(raw_trades, block_hours=BLOCK_HOURS, min_trades=MIN_TRADES_PER_BLOCK):
    """One row per complete block: buy_vol, sell_vol, close, n_trades, imbalance, fwd_ret.

    The first and last blocks are dropped because the window starts and ends
    mid-block. Blocks with fewer than min_trades trades are dropped. fwd_ret is
    NaN unless the next block follows immediately (no gap).
    """
    df = raw_trades.sort_values(['Timestamp', 'TradeID'])
    ts = pd.to_datetime(df['Timestamp'], unit='us')
    buy = ~df['IsBuyerMaker'].astype(bool)
    vol = df['Volume'].astype(float)
    work = pd.DataFrame({
        'block': ts.dt.floor(f'{block_hours}h'),
        'buy_vol': vol.where(buy, 0.0),
        'sell_vol': vol.where(~buy, 0.0),
        'price': df['Price'].astype(float),
        'n': 1,
    })
    agg = work.groupby('block').agg(
        buy_vol=('buy_vol', 'sum'), sell_vol=('sell_vol', 'sum'),
        close=('price', 'last'), n_trades=('n', 'sum'),
    )
    agg = agg.iloc[1:-1]                       # partial edge blocks
    agg = agg[agg['n_trades'] >= min_trades]   # thin blocks
    agg['imbalance'] = (agg['buy_vol'] - agg['sell_vol']) / (agg['buy_vol'] + agg['sell_vol'])

    fwd = np.log(agg['close'].shift(-1) / agg['close'])
    gap = agg.index.to_series().shift(-1) - agg.index.to_series()
    fwd[gap != pd.Timedelta(hours=block_hours)] = np.nan
    agg['fwd_ret'] = fwd
    return agg


def window_pairs(table):
    """(imbalance, fwd_ret) pairs from one window, rows without a forward return dropped."""
    return table[['imbalance', 'fwd_ret']].dropna().reset_index(drop=True)


def _spearman(x, y):
    return float(x.rank().corr(y.rank()))


def pooled_test(pairs_list):
    """Pooled Spearman test over per-window pair tables. Pairs never cross windows."""
    allp = pd.concat(pairs_list, ignore_index=True)
    n = len(allp)
    rho = _spearman(allp['imbalance'], allp['fwd_ret'])
    t = np.inf if abs(rho) >= 1 else rho * np.sqrt((n - 2) / (1 - rho ** 2))
    per_window = [_spearman(p['imbalance'], p['fwd_ret']) for p in pairs_list]
    n_same_sign = int(sum(np.sign(r) == np.sign(rho) for r in per_window))
    passes = bool(abs(t) >= T_THRESHOLD and n_same_sign >= MIN_SAME_SIGN_WINDOWS)
    return {'rho': rho, 't': float(t), 'n': n, 'per_window_rho': per_window,
            'n_same_sign': n_same_sign, 'passes': passes}


def check_windows_disjoint(spans):
    """spans: list of (start, end) Timestamps. Raises ValueError if any two overlap."""
    ordered = sorted(spans)
    for (s1, e1), (s2, e2) in zip(ordered, ordered[1:]):
        if s2 < e1:
            raise ValueError(f'Windows overlap: one ends {e1}, the next starts {s2}')


def main():
    if os.path.exists(RESULTS_FILE):
        raise SystemExit(f'{RESULTS_FILE} already exists. This test is run once; '
                         f'see the pre-registration. Not overwriting.')

    pairs_list, spans = [], []
    for name in WINDOW_DIRS:
        path = os.path.join(HERE, name, 'raw_trades.parquet')
        if not os.path.exists(path):
            raise SystemExit(f'{path} not found.')
        raw = pd.read_parquet(path)
        ts = pd.to_datetime(raw['Timestamp'], unit='us')
        spans.append((ts.min(), ts.max()))
        pairs_list.append(window_pairs(block_table(raw)))
        print(f'{name}: {len(raw):,} trades, {len(pairs_list[-1])} block pairs')

    check_windows_disjoint(spans)
    res = pooled_test(pairs_list)

    print('\nPOOLED RESULT')
    print(f"  n pairs        : {res['n']}")
    print(f"  Spearman rho   : {res['rho']:+.4f}")
    print(f"  t statistic    : {res['t']:+.2f}   (threshold +/-{T_THRESHOLD})")
    print(f"  same-sign wins : {res['n_same_sign']} of {len(WINDOW_DIRS)} (need >= {MIN_SAME_SIGN_WINDOWS})")
    print(f"  PRE-REGISTERED RULE MET: {res['passes']}")

    rows = [{'window': name, 'n_pairs': len(p), 'rho': r}
            for name, p, r in zip(WINDOW_DIRS, pairs_list, res['per_window_rho'])]
    rows.append({'window': 'POOLED', 'n_pairs': res['n'], 'rho': res['rho']})
    out = pd.DataFrame(rows)
    out['t'] = np.nan
    out.loc[out['window'] == 'POOLED', 't'] = res['t']
    out['rule_met'] = np.nan
    out.loc[out['window'] == 'POOLED', 'rule_met'] = res['passes']
    out.to_csv(RESULTS_FILE, index=False)
    print(f'\nSaved {RESULTS_FILE}')


if __name__ == '__main__':
    main()
