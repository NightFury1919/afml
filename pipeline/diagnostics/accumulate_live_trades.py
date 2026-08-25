"""
pipeline/diagnostics/accumulate_live_trades.py

Standalone script: pulls the latest LOOKBACK_HOURS of live BTCUSDT trades
(same call as run_pipeline_live.py's own pull) and appends them into the
persistent, de-duplicated archive at pipeline/diagnostics/trade_archive/
raw_trades_archive.parquet, via trade_archive.append_to_archive().

Meant to be run repeatedly over time (manually for now; a natural
candidate for the scheduling/unattended-operation work already flagged
as a separate priority for this week) -- each run's overlapping window
is expected and handled safely by the archive's TradeID-based dedup.
The archive grows in calendar-time span across runs even though each
individual pull only covers LOOKBACK_HOURS.

NOT wired into run_pipeline_live.py -- deliberately separate and
optional, so accumulation can start now without touching the existing,
working live-run flow. Whether/how to fold this into that flow (or into
a future CPCV-based evaluation, per CALIBRATION_AUDIT.md's "OFI Null
Confirmed Real..." section) remains an explicit next-session design
decision, not made here.

Usage
-----
    conda activate mlfinlab
    cd C:\\ws\\AFML
    python pipeline\\diagnostics\\accumulate_live_trades.py
"""
import os
import sys

HERE = os.path.dirname(os.path.abspath(__file__))
PIPELINE_DIR = os.path.abspath(os.path.join(HERE, '..'))
ORCH_DIR = os.path.join(PIPELINE_DIR, 'orchestration')
sys.path.insert(0, ORCH_DIR)

from ingestion import pull_recent_trades              # noqa: E402
from trade_archive import append_to_archive           # noqa: E402

SYMBOL = 'BTCUSDT'          # confirmed 2026-08-25: denser than BTCUSD on
                             # this venue (512 vs. 365 trades/hour on a
                             # same-day comparison) -- no reason to switch
LOOKBACK_HOURS = 720        # same as run_pipeline_live.py's real pull

ARCHIVE_DIR = os.path.join(HERE, 'trade_archive')
ARCHIVE_PATH = os.path.join(ARCHIVE_DIR, 'raw_trades_archive.parquet')


def main():
    api_key = os.environ.get('BINANCE_API_KEY')
    if not api_key:
        raise SystemExit(
            'BINANCE_API_KEY is not set. See ingestion.py\'s module '
            'docstring for how to get a free read-only key.'
        )

    print(f'Pulling last {LOOKBACK_HOURS}h of {SYMBOL} trades from Binance.US...')
    raw_trades = pull_recent_trades(SYMBOL, LOOKBACK_HOURS, api_key)
    print(f'  {len(raw_trades)} raw trades pulled')

    print(f'\nMerging into archive at {ARCHIVE_PATH}...')
    result = append_to_archive(raw_trades, ARCHIVE_PATH)

    print(f"\n  {result['n_new_added']} new trades added "
          f"({result['n_duplicates_skipped']} were already archived from "
          f"a prior pull)")
    print(f"  Archive now holds {result['n_total_after']} total distinct trades")
    if result['span_start'] is not None:
        print(f"  Archive span: {result['span_start']} to "
              f"{result['span_end']} ({result['span_days']:.1f} days)")

    print('\nRun this again periodically (daily is a reasonable cadence, '
          'given the 720h pull window) to keep extending the archive\'s '
          'span. Overlapping pulls are expected and safe.')


if __name__ == '__main__':
    main()
