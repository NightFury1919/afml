"""
pipeline/diagnostics/refresh_snapshot_to_now.py

log_live_prediction.py's first real run showed a 73.8h staleness gap --
the model's underlying signal was 3+ days old while being compared to a
real-time price. That's not a meaningful forward test: a real chunk of
the price move already happened during a window the model never saw.

Fix: pull ONLY the gap (new trades since the snapshot's last bar) and
append -- not a full 30-45 minute 720h re-pull. pull_recent_trades_kraken
pages FORWARD from a `since` cursor to the present (see its own
docstring) -- setting lookback_hours to just the real gap size, with no
end_time_unix override, fetches exactly the missing span up to now.

Writes a NEW refreshed snapshot dir (does not overwrite the original --
same non-destructive convention as every capture script today), with
the same full-row dedup + re-verified contiguity check
pool_kraken_raw_trades.py already established are the correct way to
combine real trade data, reused here rather than reimplemented.

Run:
    conda activate mlfinlab
    cd C:\\ws\\AFML
    python pipeline\\diagnostics\\refresh_snapshot_to_now.py <snapshot_dir>
"""
import os
import sys
import time
from datetime import date

import pandas as pd

HERE = os.path.dirname(os.path.abspath(__file__))
ORCH = os.path.join(HERE, '..', 'orchestration')
sys.path.insert(0, ORCH)

from ingestion_kraken import pull_recent_trades_kraken   # noqa: E402

PAIR = 'XBTUSD'
GAP_BUFFER_HOURS = 0.25   # small overlap margin so the new pull's own
                           # `since` cursor definitely covers back past
                           # the old snapshot's actual last trade, not
                           # just up to it -- dedup handles any overlap
MAX_ACCEPTABLE_GAP_SECONDS = 1800   # same real-world-calibrated value
                                      # pool_kraken_raw_trades.py uses


def main():
    if len(sys.argv) != 2:
        raise SystemExit('Usage: python refresh_snapshot_to_now.py <snapshot_dir>')
    snapshot_dir = sys.argv[1]
    raw_trades_path = os.path.join(snapshot_dir, 'raw_trades.parquet')
    if not os.path.exists(raw_trades_path):
        raise SystemExit(f'{raw_trades_path} not found -- wrong snapshot dir?')

    old_trades = pd.read_parquet(raw_trades_path)
    old_max_ts = old_trades['Timestamp'].max()
    old_max_time = pd.to_datetime(old_max_ts, unit='us')
    now = time.time()
    gap_hours = (now - old_max_ts / 1_000_000) / 3600.0 + GAP_BUFFER_HOURS

    print(f'Existing snapshot: {len(old_trades)} trades, last trade {old_max_time}')
    print(f'Real gap to now: {gap_hours - GAP_BUFFER_HOURS:.2f}h '
          f'(pulling {gap_hours:.2f}h with a {GAP_BUFFER_HOURS}h overlap buffer)')

    if gap_hours - GAP_BUFFER_HOURS < 0.05:
        print('Gap is under 3 minutes -- snapshot is already essentially current, '
              'not pulling anything.')
        return

    est_rate_per_hour = len(old_trades) / (
        (old_trades['Timestamp'].max() - old_trades['Timestamp'].min()) / 1_000_000 / 3600
    )
    est_trades = est_rate_per_hour * gap_hours
    max_calls = max(int(est_trades / 1000) + 10, 50) * 2
    print(f'Estimated ~{est_trades:,.0f} new trades (using this snapshot\'s own '
          f'observed rate, {est_rate_per_hour:.1f}/hour), max_calls={max_calls}')

    print('\nPulling the gap...')
    t0 = time.time()
    new_trades = pull_recent_trades_kraken(PAIR, gap_hours, max_calls=max_calls)
    elapsed_min = (time.time() - t0) / 60.0
    print(f'  {len(new_trades)} trades pulled in {elapsed_min:.1f} min, '
          f'{pd.to_datetime(new_trades["Timestamp"].min(), unit="us")} to '
          f'{pd.to_datetime(new_trades["Timestamp"].max(), unit="us")}')

    combined = pd.concat([old_trades, new_trades], ignore_index=True)
    combined = combined.sort_values('Timestamp').reset_index(drop=True)
    n_before = len(combined)
    combined = combined.drop_duplicates()
    n_dupes = n_before - len(combined)
    print(f'\nCombined: {len(combined)} total trades '
          f'(dropped {n_dupes} overlap duplicates from the buffer region)')

    gaps = combined['Timestamp'].diff().dropna() / 1_000_000
    max_gap = gaps.max()
    n_large_gaps = (gaps > MAX_ACCEPTABLE_GAP_SECONDS).sum()
    print(f'Contiguity check: max gap = {max_gap:.1f}s '
          f'(threshold {MAX_ACCEPTABLE_GAP_SECONDS}s)')
    if n_large_gaps > 0:
        raise SystemExit(
            f'{n_large_gaps} gap(s) exceed the threshold -- the pulled range may '
            'not have actually covered the full gap. Investigate before using '
            'this for a live prediction (a real hole here would make the '
            '"model saw everything up to now" claim false).'
        )
    print('  PASS: genuinely continuous, no real gap between old and new data.')

    out_dir = f'{snapshot_dir.rstrip(chr(92)).rstrip("/")}_refreshed_{date.today().isoformat()}'
    if os.path.exists(out_dir):
        raise SystemExit(f'{out_dir} already exists -- refusing to overwrite.')
    os.makedirs(out_dir)
    out_path = os.path.join(out_dir, 'raw_trades.parquet')
    combined.to_parquet(out_path)

    new_max_time = pd.to_datetime(combined['Timestamp'].max(), unit='us')
    print(f'\nWritten to {out_dir}')
    print(f'Snapshot now current through {new_max_time} '
          f'(was {old_max_time} -- staleness reduced from '
          f'{gap_hours - GAP_BUFFER_HOURS:.1f}h to essentially zero)')
    print(f'\nUse this refreshed snapshot for log_live_prediction.py instead of '
          'the original:')
    print(f'  python pipeline\\diagnostics\\log_live_prediction.py {out_dir} <target_bars>')


if __name__ == '__main__':
    main()
