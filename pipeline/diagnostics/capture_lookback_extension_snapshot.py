"""
pipeline/diagnostics/capture_lookback_extension_snapshot.py

Follow-on to 2026-08-25's OFI trace: at edge_strength=1.0 (this
generator's strongest tested signal), the classifier demonstrably
extracts real out-of-sample skill (OOS accuracy 0.575, corr(prob,label)
+0.284) -- but DSR still only reads 0.6079 at T_effective=97.10, which
2026-08-19's Detection Power Calibration Findings already predicted:
at this T, the fat-tailed-regime null baseline itself sits around
DSR~0.573, so 0.6079 is barely distinguishable from noise. Meaningful
DSR discrimination needs T_effective~200-1000 (same document).

Ethan's question: does extending LOOKBACK_HOURS help reach that range?
The 2026-08-19 lever sweep (CALIBRATION_AUDIT.md's "T_effective Lever
Sweep Findings") already established LOOKBACK_HOURS is a red herring
WITHIN a fixed target_bars: compute_dynamic_threshold() rescales the
dollar-bar threshold to hit ~target_bars bars regardless of pull length,
so a longer pull alone doesn't raise T_raw. But target_bars=750->1000
plateaued (CALIBRATION_AUDIT.md's "target_bars=1000" section) because
"the pulled window's raw trade count... starting to become a binding
constraint on how many additional bars a larger target_bars can
extract" -- i.e. the plateau's real cause was running out of raw trades
to subdivide, not target_bars itself failing. A LONGER pull gives more
raw trades to subdivide, which may let target_bars climb further before
hitting that same constraint. This is untested -- this script and its
companion (calibrate_lookback_extension.py) test it directly.

DESIGN: pulls ONCE at the longest lookback to be tested (2160h / 90
days), rather than three separate live pulls at 720h/1440h/2160h. Since
Binance's historicalTrades pages BACKWARD from the most recent trade,
one 2160h pull already CONTAINS the most recent 720h and 1440h as exact
subsets -- slicing downstream (calibrate_lookback_extension.py) gives an
apples-to-apples comparison against the SAME underlying market data for
all three window lengths, avoiding the cross-pull drift confound this
project's frozen-snapshot convention exists to prevent (same discipline
as capture_t_effective_snapshot.py).

COST NOTE: 2160h at this pipeline's observed live trade density
(~150-165 trades/hour, per recent pulls) is roughly 325,000-355,000
trades -- pull_recent_trades()'s defaults (limit_per_call=1000,
max_calls=500) cap a single pull at 500,000 trades, so this should fit,
but is closer to that ceiling than any prior pull in this project. If
this raises ValueError before covering the window, rerun with
max_calls raised (pull_recent_trades's own documented behavior: treat
this as "increase max_calls," not silently truncate).

Run ONCE before calibrate_lookback_extension.py.

Usage
-----
    conda activate mlfinlab
    cd C:\\ws\\AFML
    python pipeline\\diagnostics\\capture_lookback_extension_snapshot.py
"""
import os
import sys
from datetime import date

HERE = os.path.dirname(os.path.abspath(__file__))
ORCH = os.path.join(HERE, '..', 'orchestration')
sys.path.insert(0, ORCH)

from ingestion import pull_recent_trades           # noqa: E402

# Longest window to be tested by calibrate_lookback_extension.py -- the
# 720h and 1440h windows are sliced from this same pull, not pulled
# separately (see module docstring).
LOOKBACK_HOURS = 2160   # 90 days

SNAPSHOT_DIR = os.path.join(
    HERE, f'lookback_extension_snapshot_{date.today().isoformat()}'
)


def main():
    api_key = os.environ.get('BINANCE_API_KEY')
    if not api_key:
        raise SystemExit(
            'BINANCE_API_KEY is not set. See ingestion.py\'s module '
            'docstring for how to get a free read-only key.'
        )

    if os.path.exists(SNAPSHOT_DIR):
        raise SystemExit(
            f'{SNAPSHOT_DIR} already exists -- refusing to overwrite a '
            'snapshot the lookback-extension sweep may already be using. '
            'Delete it manually first if you really want a fresh capture '
            'today.'
        )
    os.makedirs(SNAPSHOT_DIR)

    print(f'Pulling last {LOOKBACK_HOURS}h ({LOOKBACK_HOURS/24:.0f} days) of '
          f'BTCUSDT trades from Binance.US...')
    print('This is a much larger pull than this project\'s usual 720h -- '
          'expect this to take noticeably longer and use more API calls.')
    raw_trades = pull_recent_trades(
        'BTCUSDT', LOOKBACK_HOURS, api_key, max_calls=600,
        # max_calls raised from the default 500 -- see module docstring's
        # COST NOTE on why 2160h may approach the default ceiling.
    )
    print(f'  {len(raw_trades)} raw trades pulled')

    raw_trades.to_parquet(os.path.join(SNAPSHOT_DIR, 'raw_trades.parquet'))

    print(f'\nSnapshot frozen to {SNAPSHOT_DIR}')
    print('calibrate_lookback_extension.py will slice this into 720h/1440h/')
    print('2160h windows and test target_bars={1000,1500,2000} on each --')
    print('do not re-run this script mid-sweep.')


if __name__ == '__main__':
    main()
