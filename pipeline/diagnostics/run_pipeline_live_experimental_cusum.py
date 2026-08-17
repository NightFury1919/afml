"""
run_pipeline_live_experimental_cusum.py -- ONE-OFF EXPERIMENT, not part of
the committed pipeline.

Runs the exact same chain as pipeline/run_pipeline_live.py (ingestion ->
rebuild -> features -> live_staging -> Ch11 real trials -> risk_context ->
report -> oversight), with ONE difference: rebuild_module.CUSUM_H is
monkeypatched to a looser value for the duration of this run, then
restored -- mirroring stages.py's own established INPUT/HERE monkeypatch-
and-restore-in-finally pattern for ch11, rather than editing rebuild.py's
committed CUSUM_H=500 constant.

Writes to SEPARATE output locations (experimental_cusum_staging_data/,
experimental_cusum_output/, experimental_cusum_report.txt) so this never
overwrites the real live_staging_data/live_run_output/latest_live_report.txt
from a genuine run_pipeline_live.py call.

Purpose: see what a MUCH larger real sample (h=100 -> ~150+ raw CUSUM
events, vs. h=500's ~45-49) does to PBO/DSR/OTR findings, before deciding
whether a permanent CUSUM_H redesign (see rebuild.py's own KNOWN OPEN
QUESTION) is worth the book-fidelity/event-density tradeoff discussed
2026-08-16.

Requires BINANCE_API_KEY, same as run_pipeline_live.py.

Usage
-----
    conda activate mlfinlab
    cd C:\\ws\\AFML
    $env:BINANCE_API_KEY = 'your-key-here'
    python run_pipeline_live_experimental_cusum.py           # h=100 default
    python run_pipeline_live_experimental_cusum.py 75         # custom h
"""
import os
import sys

HERE = os.path.dirname(os.path.abspath(__file__))
PIPELINE_DIR = os.path.join(HERE, 'pipeline')
EXPERIMENTAL_STAGING_DIR = os.path.join(PIPELINE_DIR, 'experimental_cusum_staging_data')
EXPERIMENTAL_HERE_DIR = os.path.join(PIPELINE_DIR, 'experimental_cusum_output')
EXPERIMENTAL_REPORT_PATH = os.path.join(PIPELINE_DIR, 'experimental_cusum_report.txt')

sys.path.insert(0, os.path.join(PIPELINE_DIR, 'orchestration'))
sys.path.insert(0, os.path.join(HERE, 'portfolio_oversight'))

from ingestion import pull_recent_trades              # noqa: E402
import rebuild as rebuild_module                        # noqa: E402
from rebuild import build_bars_and_labels               # noqa: E402
from features import build_enriched_events              # noqa: E402
from live_staging import stage_live_training_tables      # noqa: E402
from stages import (                                    # noqa: E402
    load_ch11_driver, run_live_trials, evaluate_overfitting,
    latest_bet_signal,
)
from risk_context import (                              # noqa: E402
    compute_otr_finding, compute_strategy_risk, compute_pt_sl_context,
)
from report import build_report                          # noqa: E402
from oversight import build_oversight_section             # noqa: E402

LOOKBACK_HOURS = 720
PAPER_CAPITAL_USD = 10_000.0
DEFAULT_EXPERIMENTAL_CUSUM_H = 100  # crosses min_reliable_T=150 in raw
                                     # CUSUM terms per 2026-08-16 scan on
                                     # today's staged close series (153
                                     # events at h=100 vs 47 at h=500)


def main():
    experimental_h = (
        float(sys.argv[1]) if len(sys.argv) > 1 else DEFAULT_EXPERIMENTAL_CUSUM_H
    )

    api_key = os.environ.get('BINANCE_API_KEY')
    if not api_key:
        raise SystemExit(
            'BINANCE_API_KEY is not set. See ingestion.py\'s module '
            'docstring for how to get a free read-only key.'
        )

    print(f'*** EXPERIMENTAL RUN: CUSUM_H={experimental_h} (committed '
          f'default is {rebuild_module.CUSUM_H}) ***')
    print(f'Pulling last {LOOKBACK_HOURS}h of BTCUSDT trades from Binance.US...')
    raw_trades = pull_recent_trades('BTCUSDT', LOOKBACK_HOURS, api_key)
    print(f'  {len(raw_trades)} raw trades pulled')

    original_cusum_h = rebuild_module.CUSUM_H
    try:
        rebuild_module.CUSUM_H = experimental_h
        rebuild_result = build_bars_and_labels(raw_trades)
    finally:
        rebuild_module.CUSUM_H = original_cusum_h

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
        rebuild_result, enriched_result, EXPERIMENTAL_STAGING_DIR,
    )
    print(f"  staged {staged['n_events']} enriched events to "
          f"{staged['enriched_csv_path']}")

    ch11 = load_ch11_driver()
    M, meta = run_live_trials(ch11, EXPERIMENTAL_STAGING_DIR, EXPERIMENTAL_HERE_DIR)

    eval_result = evaluate_overfitting(M, meta, ch11, S=8)
    signal = latest_bet_signal(
        eval_result['best_trial'], meta, ch11, EXPERIMENTAL_STAGING_DIR,
    )

    print('  computing risk context (Ch13 OTR, Ch15 strategy risk, PT/SL)...')
    otr_result = compute_otr_finding(rebuild_result)
    print(f"    OTR: phi_hat={otr_result['phi_hat']:.4f}, "
          f"stationary={otr_result['stationary']}")
    strategy_risk_result = compute_strategy_risk(rebuild_result, eval_result['sr_hat'])
    print(f"    Ch15: P[fail]={strategy_risk_result['p_fail']:.4f} "
          f"at tSR=sr_hat={eval_result['sr_hat']:.4f}")
    pt_sl_result = compute_pt_sl_context(rebuild_result, rebuild_module.PT_SL)
    print(f"    PT/SL: +{pt_sl_result['implied_pt_pct']:.2%} / "
          f"-{pt_sl_result['implied_sl_pct']:.2%}")

    caveats = [
        f"*** EXPERIMENTAL RUN (2026-08-16): CUSUM_H={experimental_h} "
        f"(NOT the committed default of {original_cusum_h}). This report "
        f"is a one-off diagnostic to see how findings change with a much "
        f"larger real sample -- it is NOT the pipeline's established "
        f"result and should not be compared apples-to-apples with prior "
        f"handoffs' tracking tables without this caveat."
    ]
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

    report = build_report(
        eval_result, signal,
        asset_label=f'BTC/USDT (EXPERIMENTAL live, CUSUM_H={experimental_h})',
        otr_result=otr_result, strategy_risk_result=strategy_risk_result,
        pt_sl_result=pt_sl_result,
    )
    report += '\n\n' + '\n\n'.join(caveats)

    oversight_section = build_oversight_section(
        signal, eval_result, strategy_risk_result=strategy_risk_result,
        paper_capital_usd=PAPER_CAPITAL_USD,
    )
    report += '\n\n' + oversight_section

    print(report)

    with open(EXPERIMENTAL_REPORT_PATH, 'w') as f:
        f.write(report)
    print(f'\n[EXPERIMENTAL live report written to {EXPERIMENTAL_REPORT_PATH}]')
    print(f'[real latest_live_report.txt was NOT touched by this run]')


if __name__ == '__main__':
    main()
