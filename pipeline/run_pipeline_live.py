"""
pipeline/run_pipeline_live.py

Phase 3: live-data counterpart to run_pipeline.py. Chains ingestion ->
rebuild -> features -> live_staging -> Ch11's real trial construction
(via stages.run_live_trials, live data monkeypatched in without touching
chapter_11_backtest_dangers.py) -> the SAME evaluate_overfitting/
latest_bet_signal/build_report used by the static-data pipeline.

Requires BINANCE_API_KEY (see ingestion.py's module docstring for setup).

Usage
-----
    conda activate mlfinlab
    cd C:\\ws\\AFML
    $env:BINANCE_API_KEY = 'your-key-here'
    python pipeline\\run_pipeline_live.py
"""
import os
import sys

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.abspath(os.path.join(HERE, '..'))
LIVE_STAGING_DIR = os.path.join(HERE, 'live_staging_data')
LIVE_HERE_DIR = os.path.join(HERE, 'live_run_output')

sys.path.insert(0, os.path.join(HERE, 'orchestration'))

from ingestion import pull_recent_trades              # noqa: E402
from rebuild import build_bars_and_labels              # noqa: E402
from features import build_enriched_events              # noqa: E402
from live_staging import stage_live_training_tables      # noqa: E402
from stages import (                                    # noqa: E402
    load_ch11_driver, run_live_trials, evaluate_overfitting,
    latest_bet_signal,
)
from report import build_report                          # noqa: E402

LOOKBACK_HOURS = 720   # 30-day minimum, LIVE-CONFIRMED 2026-08-13 (see
                        # ingestion.py/rebuild.py -- shorter windows leave
                        # get_daily_vol() empty or produce too few events)


def main():
    api_key = os.environ.get('BINANCE_API_KEY')
    if not api_key:
        raise SystemExit(
            'BINANCE_API_KEY is not set. See ingestion.py\'s module '
            'docstring for how to get a free read-only key.'
        )

    print(f'Pulling last {LOOKBACK_HOURS}h of BTCUSDT trades from Binance.US...')
    raw_trades = pull_recent_trades('BTCUSDT', LOOKBACK_HOURS, api_key)
    print(f'  {len(raw_trades)} raw trades pulled')

    rebuild_result = build_bars_and_labels(raw_trades)
    print(f"  {len(rebuild_result['bars'])} bars, "
          f"{len(rebuild_result['events'])} triple-barrier events, "
          f"threshold=${rebuild_result['threshold']:,.2f}")

    enriched_result = build_enriched_events(
        raw_trades, rebuild_result['threshold'], rebuild_result['events'],
    )
    print(f"  {enriched_result['n_events_after']}/"
          f"{enriched_result['n_events_before']} events survived "
          f"feature enrichment (fracdiff d={enriched_result['fracdiff_d']})")

    staged = stage_live_training_tables(
        rebuild_result, enriched_result, LIVE_STAGING_DIR,
    )
    print(f"  staged {staged['n_events']} enriched events to "
          f"{staged['enriched_csv_path']}")

    ch11 = load_ch11_driver()
    M, meta = run_live_trials(ch11, LIVE_STAGING_DIR, LIVE_HERE_DIR)

    eval_result = evaluate_overfitting(M, meta, ch11, S=8)
    signal = latest_bet_signal(
        eval_result['best_trial'], meta, ch11, LIVE_STAGING_DIR,
    )

    caveats = []
    if enriched_result['fracdiff_d'] == 0:
        caveats.append(
            "This run's fracdiff feature used d=0 -- an UNRESOLVED, "
            "marginal ADF finding on live data's small sample (see "
            "features.py's own LOAD-BEARING note). If d=0 is a small-"
            "sample artifact rather than a real finding, this feature is "
            "effectively just log(price) -- non-stationary, the exact "
            "failure mode Chapter 5's frac-diff exists to prevent. Treat "
            "any signal from this run with that caveat in mind."
        )

    report = build_report(eval_result, signal, asset_label='BTC/USDT (live, Binance.US)')
    if caveats:
        report += '\n\n' + '\n\n'.join(caveats)
    print(report)

    out_path = os.path.join(HERE, 'latest_live_report.txt')
    with open(out_path, 'w') as f:
        f.write(report)
    print(f'\n[live report written to {out_path}]')


if __name__ == '__main__':
    main()