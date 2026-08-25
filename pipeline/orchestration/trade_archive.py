"""
pipeline/orchestration/trade_archive.py

Accumulates successive live pulls (ingestion.pull_recent_trades()) into
ONE persistent, de-duplicated raw-trades archive that grows over calendar
time -- rather than each pipeline run only ever seeing whatever fits in
its own LOOKBACK_HOURS window.

Motivation (2026-08-25 session): this project's own 2026-08-19 Detection
Power Calibration Findings flagged "accumulating multiple live pulls into
a combined historical dataset over time (live_run_log.csv's accumulating
structure could eventually support this)" as an unexplored T_effective
lever. Later the same week, the lookback-extension sweep confirmed a
longer window DOES raise T_effective (reaching 255.80 at 2160h/
target_bars=2000, the first result to clear the ~200 DSR-reliability
threshold) -- but also surfaced a real single-window regime-dependency
cost (CALIBRATION_AUDIT.md's "OFI Null Confirmed Real..." section: the
2160h window's own trial grid was uniformly negative-Sharpe, driven by
whatever happened in the earlier part of that ONE 90-day draw). An
accumulated archive addresses both at once: density grows without
needing to trust any single large window's particular historical
sequence, and it's the natural raw-data input a future CPCV-across-paths
evaluation (recommended in that same section, not yet implemented) would
need -- CPCV needs a genuinely long history to split into groups; a
single 720h-2160h pull is not that.

DESIGN: symmetric to ingestion.py's own established conventions --
parquet for storage (this project's standing cross-environment-safe
format; CSV/parquet only, never pickles across environments, per
CLAUDE.md), TradeID as the dedup key (Binance's own unique per-trade
identifier -- more reliable than Timestamp, given ingestion.py's own
documented millisecond-collision issue on this venue), append-safe (an
interrupted accumulation run doesn't corrupt or lose the existing
archive -- same "results written incrementally" discipline as this
project's sweep CSVs).

NOT YET WIRED into run_pipeline_live.py's main flow -- that's a
separate, explicit integration decision (per this project's standing
"confirm design decisions before implementation" convention), same
status as the CPCV-on-longer-window recommendation this accumulation
mechanism exists to eventually feed. This module and its companion
script (accumulate_live_trades.py) are usable standalone starting now,
independent of that larger integration.
"""
import os

import pandas as pd

RAW_TRADE_COLUMNS = [
    'TradeID', 'Price', 'Volume', 'QuoteVolume', 'Timestamp',
    'IsBuyerMaker', 'IsBestMatch',
]


def load_archive(archive_path):
    """Loads the persistent archive if it exists, otherwise returns an
    empty DataFrame with the correct schema (RAW_TRADE_COLUMNS) so
    callers can treat "no archive yet" and "empty archive" identically
    without a separate existence check."""
    if not os.path.exists(archive_path):
        return pd.DataFrame(columns=RAW_TRADE_COLUMNS)
    return pd.read_parquet(archive_path)


def append_to_archive(new_trades, archive_path):
    """Merges `new_trades` (a fresh pull, e.g. from
    ingestion.pull_recent_trades()) into the persistent archive at
    `archive_path`, de-duplicating by TradeID and re-sorting by
    Timestamp ascending, then writes the merged result back.

    Successive live pulls with overlapping LOOKBACK_HOURS windows are
    the expected, normal case -- most of `new_trades` will already be
    in the archive after the first few calls. This is deliberate: it's
    what makes running this on a simple recurring schedule (e.g. once
    per day) safe -- there is no requirement that pulls be
    non-overlapping or precisely timed.

    Parameters
    ----------
    new_trades : pd.DataFrame, columns matching RAW_TRADE_COLUMNS (the
        same schema ingestion.pull_recent_trades() returns)
    archive_path : str, path to the persistent .parquet archive file
        (created if it doesn't exist yet)

    Returns
    -------
    dict with keys:
      'n_new_added'    : int, trades actually new to the archive after
                         dedup (0 on a fully-overlapping pull -- not an
                         error, just nothing new to add)
      'n_duplicates_skipped' : int, trades in `new_trades` that were
                         already in the archive
      'n_total_after'  : int, total distinct trades in the archive after
                         this merge
      'span_start'     : pd.Timestamp, earliest trade in the archive
                         after this merge
      'span_end'       : pd.Timestamp, latest trade in the archive after
                         this merge
      'span_days'      : float, (span_end - span_start) in days
    """
    missing = set(RAW_TRADE_COLUMNS) - set(new_trades.columns)
    if missing:
        raise ValueError(
            f'new_trades is missing required columns: {sorted(missing)} '
            f'-- expected schema {RAW_TRADE_COLUMNS}'
        )

    existing = load_archive(archive_path)
    n_existing = len(existing)

    combined = pd.concat([existing, new_trades[RAW_TRADE_COLUMNS]], ignore_index=True)
    combined = combined.drop_duplicates(subset='TradeID', keep='first')
    combined = combined.sort_values('Timestamp').reset_index(drop=True)

    n_total_after = len(combined)
    n_new_added = n_total_after - n_existing
    n_duplicates_skipped = len(new_trades) - n_new_added

    os.makedirs(os.path.dirname(os.path.abspath(archive_path)), exist_ok=True)
    combined.to_parquet(archive_path)

    if n_total_after > 0:
        span_start = pd.to_datetime(combined['Timestamp'].min(), unit='us')
        span_end = pd.to_datetime(combined['Timestamp'].max(), unit='us')
        span_days = (span_end - span_start).total_seconds() / 86400.0
    else:
        span_start = span_end = None
        span_days = 0.0

    return {
        'n_new_added': n_new_added,
        'n_duplicates_skipped': n_duplicates_skipped,
        'n_total_after': n_total_after,
        'span_start': span_start,
        'span_end': span_end,
        'span_days': span_days,
    }


# ---------------------------------------------------------------------------
# TDD results -- sandbox pre-check (numpy 2.4/pandas 3.0), pending real-
# machine confirmation (mlfinlab env: Python 3.10.20, pandas 1.5.3,
# numpy 1.23.5) -- see test_trade_archive.py for the full suite.
#
# python -m pytest pipeline/orchestration/test_trade_archive.py -v
# ============================== 9 passed in 5.99s ==============================
# (sandbox: Python 3.12.3, pandas/numpy current -- run again on the real
# mlfinlab env below and update this block with that real output before
# considering this closed, per this project's two-pass TDD convention)
# ---------------------------------------------------------------------------
