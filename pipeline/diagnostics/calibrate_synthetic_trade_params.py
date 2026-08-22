"""
calibrate_synthetic_trade_params.py

Purpose
-------
The synthetic order-flow-imbalance edge-detection harness (next-session
build, agreed 2026-08-21) needs a fake raw-trade generator whose BASELINE
statistics -- tick density, per-trade price volatility, average trade
size, and baseline buy/sell imbalance -- match real observed BTC data.
Otherwise any edge-detection result we get is confounded by "the synthetic
tape doesn't look like real data" rather than telling us anything about
the pipeline's real sensitivity.

This script computes those baseline stats from ONE real trade file (either
the March static CSV or a fresh live pull -- pass via --source) and writes
them to a small JSON file the generator will read as its defaults.

This is read-only diagnostic tooling. It does not touch any production
pipeline file.

Usage
-----
    conda activate mlfinlab
    cd C:\ws\AFML
    python pipeline\diagnostics\calibrate_synthetic_trade_params.py --source static
    # or
    python pipeline\diagnostics\calibrate_synthetic_trade_params.py --source live

Output
------
    pipeline/diagnostics/synthetic_trade_baseline_params.json
"""

import argparse
import json
import os
import sys

import numpy as np
import pandas as pd

# LOAD-BEARING (2026-08-22): path portability convention -- derive root via
# __file__, never hardcode an absolute path, per repo convention (decided
# 2026-06-28).
ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '..'))
sys.path.insert(0, os.path.join(ROOT, 'pipeline'))


def load_static_trades():
    """Load the March 2026 BTC/TUSD static CSV used by the original
    chapter work and the original pipeline baseline.

    LOAD-BEARING (2026-08-22): this CSV has NO header row -- confirmed by
    running `pd.read_csv(..., nrows=3)` and seeing row 0's actual data
    misread as column names (e.g. a column literally named '395339315',
    which is really the first TradeId value). Columns are raw Binance
    trade-schema order: TradeId, Price, Volume, QuoteVolume, Timestamp,
    IsBuyerMaker, IsBestMatch. Timestamp is in MICROSECONDS since epoch,
    not milliseconds -- confirmed by digit count (16 digits, e.g.
    1772323209088203; current-day millisecond epoch time is 13 digits,
    e.g. 1772323209088 -- the extra 3 digits are exactly what
    microsecond precision would add). Do not assume this matches
    ingestion.py's live-pull timestamp unit without checking that
    separately.
    """
    path = os.path.join(ROOT, 'input_data', 'BTCTUSD-trades-2026-03.csv')
    if not os.path.exists(path):
        raise FileNotFoundError(
            f"Static trade CSV not found at {path}. If it lives somewhere "
            "else on this machine, edit this function's path or pass the "
            "correct path in directly."
        )
    df = pd.read_csv(
        path,
        header=None,
        names=['TradeId', 'Price', 'Volume', 'QuoteVolume', 'Timestamp',
               'IsBuyerMaker', 'IsBestMatch'],
    )
    df['Timestamp'] = pd.to_datetime(df['Timestamp'], unit='us')
    return df


def load_live_trades():
    """Pull a fresh live BTCUSDT window via the pipeline's own, already-
    tested ingestion module -- reuses pull_recent_trades(), no
    reimplementation. Uses the read-only Binance.US API key already set
    as a session-scoped env var."""
    from orchestration import ingestion  # real module, real function
    # LOAD-BEARING (2026-08-22): 720h window matches the window size used
    # in the 2026-08-21 CUSUM_H staleness audit, for rough comparability
    # with that session's diff-std numbers if useful later.
    df = ingestion.pull_recent_trades(symbol='BTCUSDT', hours=720)
    return df


def compute_baseline_stats(df: pd.DataFrame) -> dict:
    """
    Compute the four baseline statistics the synthetic generator needs:

    1. tick_density_per_sec: median trades/second (tick arrival rate)
    2. price_diff_std: std of trade-to-trade price differences (NOT bar-
       to-bar -- this is raw tick-level jitter, since the generator
       produces raw trades, and rebuild.py's own bar construction will
       aggregate them the same way it aggregates real trades)
    3. avg_trade_size: mean trade volume, for realistic Volume column
    4. baseline_imbalance: fraction of trades with IsBuyerMaker == True,
       BEFORE any edge injection -- this is what "no injected edge"
       should reproduce, and injected edge tilts away from this baseline
       rather than from an arbitrary 0.5.

    All computed directly from the real trade file, no assumptions.
    """
    df = df.copy()
    # Normalize expected column names -- static CSV and live pull may
    # differ slightly; adjust here if the real column names don't match
    # (Ethan: please confirm actual column names when you run this --
    # I'm assuming Price/Volume/Timestamp/IsBuyerMaker per the agreed
    # design, matching Binance trade schema convention).
    df['Timestamp'] = pd.to_datetime(df['Timestamp'])
    df = df.sort_values('Timestamp').reset_index(drop=True)

    time_diffs_sec = df['Timestamp'].diff().dt.total_seconds().dropna()
    tick_density_per_sec = 1.0 / time_diffs_sec.median()

    price_diffs = df['Price'].diff().dropna()
    price_diff_std = price_diffs.std()

    avg_trade_size = df['Volume'].mean()

    baseline_imbalance = df['IsBuyerMaker'].mean()

    n_trades = len(df)
    span_hours = (df['Timestamp'].iloc[-1] - df['Timestamp'].iloc[0]).total_seconds() / 3600.0

    return {
        'n_trades': int(n_trades),
        'span_hours': float(span_hours),
        'tick_density_per_sec': float(tick_density_per_sec),
        'median_inter_trade_sec': float(time_diffs_sec.median()),
        'price_diff_std': float(price_diff_std),
        'price_start': float(df['Price'].iloc[0]),
        'price_end': float(df['Price'].iloc[-1]),
        'avg_trade_size': float(avg_trade_size),
        'baseline_imbalance': float(baseline_imbalance),
    }


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--source', choices=['static', 'live'], required=True)
    args = parser.parse_args()

    if args.source == 'static':
        df = load_static_trades()
    else:
        df = load_live_trades()

    stats = compute_baseline_stats(df)
    stats['source'] = args.source

    print("=== Real trade baseline stats ===")
    for k, v in stats.items():
        print(f"  {k}: {v}")

    out_path = os.path.join(
        ROOT, 'pipeline', 'diagnostics', 'synthetic_trade_baseline_params.json'
    )
    with open(out_path, 'w') as f:
        json.dump(stats, f, indent=2)
    print(f"\nWrote baseline params to {out_path}")


if __name__ == '__main__':
    main()
