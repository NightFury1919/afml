"""
pipeline/run_pipeline_live.py

Phase 3: live-data counterpart to run_pipeline.py. Chains ingestion ->
rebuild -> features -> live_staging -> Ch11's real trial construction
(via stages.run_live_trials, live data monkeypatched in without touching
chapter_11_backtest_dangers.py) -> the SAME evaluate_overfitting/
latest_bet_signal/build_report used by the static-data pipeline.

Phase 4 (2026-08-15): also computes and reports risk context (Ch13 OTR
re-derived live, Ch15 strategy risk at this run's own sr_hat, rebuild.py's
PT/SL) via risk_context.py -- see that module's own docstring and the
2026-08-14 handoff's Part 5.

Phase 5 (2026-08-15): also appends an EXPERIMENTAL, NOT-FROM-AFML
portfolio-oversight section (portfolio_oversight/oversight.py -- capital-
based position sizing, an informational circuit-breaker flag, a single-
run lifecycle-stage classification) as a clearly separate, clearly
labeled block after the real AFML report. See oversight.py's own module
docstring for why this lives in its own top-level directory and is never
merged into report.py itself.

Phase 6 (2026-08-19): every run now auto-appends one row to
pipeline/diagnostics/live_run_log.csv (see live_run_logger.py) -- same
tidy schema convention as this project's other diagnostics/*.csv files.
Pass --snapshot to ALSO write a narrative README + copy the staged
training table into pipeline/live_run_examples/YYYY-MM-DD/, matching the
existing hand-written 2026-08-14 example's style -- this stays opt-in
per that folder's standing "occasional frozen artifact" convention (see
live_run_logger.py's own module docstring for why the two outputs are
kept separate).

Requires BINANCE_API_KEY (see ingestion.py's module docstring for setup).

Usage
-----
    conda activate mlfinlab
    cd C:\\ws\\AFML
    $env:BINANCE_API_KEY = 'your-key-here'
    python pipeline\\run_pipeline_live.py               # logs CSV row only
    python pipeline\\run_pipeline_live.py --snapshot     # also writes README snapshot
"""
import os
import sys
from datetime import date

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.abspath(os.path.join(HERE, '..'))
LIVE_STAGING_DIR = os.path.join(HERE, 'live_staging_data')
LIVE_HERE_DIR = os.path.join(HERE, 'live_run_output')

sys.path.insert(0, os.path.join(HERE, 'orchestration'))
sys.path.insert(0, os.path.join(ROOT, 'portfolio_oversight'))
sys.path.insert(0, os.path.join(HERE, 'diagnostics'))

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
from oversight import build_oversight_section, classify_lifecycle_stage  # noqa: E402
from live_run_logger import log_live_run, write_snapshot_readme  # noqa: E402

LOOKBACK_HOURS = 720   # 30-day minimum, LIVE-CONFIRMED 2026-08-13 (see
                        # ingestion.py/rebuild.py -- shorter windows leave
                        # get_daily_vol() empty or produce too few events)
PAPER_CAPITAL_USD = 10_000.0   # arbitrary, invented judgment call for the
                                # portfolio_oversight/ add-on ONLY -- see
                                # oversight.py's own module docstring.
                                # Not an AFML parameter.


def main(write_snapshot=False):
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

    # *** LOAD-BEARING (2026-08-17): tw for the DSR uniqueness-weighting
    # fix *** -- rebuild_result['tw'] is indexed to the PRE-enrichment
    # event population; evaluate_overfitting() needs it aligned to the
    # population that actually fed the trial grid (enriched_result's
    # post-fracdiff-dropna subset). Same reindex-and-fail-loud pattern
    # live_staging.py already uses for 'w'.
    tw_aligned = rebuild_result['tw'].reindex(enriched_result['enriched_events'].index)
    if tw_aligned.isna().any():
        raise ValueError(
            "tw has NaN after reindexing to the enriched event index -- an "
            "enriched event has no matching rebuild.py tw value, which "
            "should be impossible since build_enriched_events() only DROPS "
            "rows from rebuild.py's events, never adds new ones (same "
            "invariant live_staging.py already relies on for w). "
            "Investigate before evaluating overfitting on it."
        )

    eval_result = evaluate_overfitting(M, meta, ch11, S=12, tw=tw_aligned)  # S=12 per calibrate_pbo_precision.py, 2026-08-18
    signal = latest_bet_signal(
        eval_result['best_trial'], meta, ch11, LIVE_STAGING_DIR,
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

    report = build_report(
        eval_result, signal, asset_label='BTC/USDT (live, Binance.US)',
        otr_result=otr_result, strategy_risk_result=strategy_risk_result,
        pt_sl_result=pt_sl_result,
    )
    if caveats:
        report += '\n\n' + '\n\n'.join(caveats)

    oversight_section = build_oversight_section(
        signal, eval_result, strategy_risk_result=strategy_risk_result,
        paper_capital_usd=PAPER_CAPITAL_USD,
    )
    report += '\n\n' + oversight_section

    # *** Phase 6 (2026-08-19): auto-log this run *** -- classify_lifecycle_stage
    # is called again here (build_oversight_section already called it
    # internally to build oversight_section's text) since it's a pure
    # function of eval_result and its return dict isn't otherwise exposed
    # to this scope. Cheap, no side effects, safe to call twice.
    lifecycle = classify_lifecycle_stage(eval_result)
    row = {
        'run_date': date.today().isoformat(),
        'n_raw_trades': len(raw_trades),
        'n_bars': len(rebuild_result['bars']),
        'n_events': len(rebuild_result['events']),
        'n_events_enriched': enriched_result['n_events_after'],
        'fracdiff_d': enriched_result['fracdiff_d'],
        'S': eval_result['S'],
        'n_trials': eval_result['n_trials'],
        'T_raw': eval_result['T_raw'],
        'tw_mean': eval_result['tw_mean'],
        'T_effective': eval_result['T'],
        'best_trial': eval_result['best_trial'],
        'best_sharpe': eval_result['sr_hat'],
        'pbo': eval_result['prob_overfit'],
        'dsr': eval_result['dsr'],
        'skew': eval_result['skew'],
        'kurtosis': eval_result['kurtosis'],
        'phi_hat': otr_result['phi_hat'],
        'phi_stationary': otr_result['stationary'],
        'half_life': otr_result.get('half_life'),
        'p_fail': strategy_risk_result['p_fail'],
        'realized_precision': strategy_risk_result['p_bar'],
        'freq_real': strategy_risk_result['freq_real'],
        'lifecycle_stage': lifecycle['stage'],
        'position_size': signal if signal is not None else 0.0,
        'notes': '; '.join(caveats) if caveats else '',
    }
    log_live_run(os.path.join(HERE, 'diagnostics', 'live_run_log.csv'), row)

    if write_snapshot:
        import shutil
        snapshot_dir = os.path.join(HERE, 'live_run_examples', row['run_date'])
        os.makedirs(snapshot_dir, exist_ok=True)
        files_written = []
        enriched_dst = os.path.join(snapshot_dir, 'ch07_training_table_enriched.csv')
        shutil.copy(staged['enriched_csv_path'], enriched_dst)
        files_written.append((
            'ch07_training_table_enriched.csv',
            'staged live training table (12 real features + fracdiff, '
            't1/bin/w -- trgt/ret deliberately excluded)',
        ))
        write_snapshot_readme(snapshot_dir, row, files_written)
        print(f'[snapshot written to {snapshot_dir}]')

    print(report)

    out_path = os.path.join(HERE, 'latest_live_report.txt')
    with open(out_path, 'w') as f:
        f.write(report)
    print(f'\n[live report written to {out_path}]')


if __name__ == '__main__':
    main(write_snapshot='--snapshot' in sys.argv)