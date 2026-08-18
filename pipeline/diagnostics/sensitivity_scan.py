"""
pipeline/diagnostics/sensitivity_scan.py

Tier-3 constant sensitivity sweep: ROLL_WINDOW, VPIN_WINDOW, FFD_THRES, S.
Runs every sweep point against the SAME frozen live-data snapshot (see
capture_sensitivity_snapshot.py) so differences in output are attributable
to the constant being varied, not to live data drift between pulls.

*** LOAD-BEARING (2026-08-18): explicit keyword overrides, NOT monkeypatching
the module constants ***
features.py's compute_fracdiff_feature(close, thres=FFD_THRES) and
compute_ch19_features(trades, bars_vol, roll_window=ROLL_WINDOW,
vpin_window=VPIN_WINDOW) bind these constants as DEFAULT ARGUMENT VALUES at
function-definition time (import time), not as live global lookups at call
time. Monkeypatching features.ROLL_WINDOW after import has NO effect on
these functions' behavior -- their defaults are already baked in. This
script therefore never touches the features module's globals; it calls
compute_fracdiff_feature()/compute_ch19_features() directly with explicit
keyword overrides, replicating build_enriched_events()'s own body rather
than calling it as a black box. features.py itself is never edited.

Usage
-----
    conda activate mlfinlab
    cd C:\ws\AFML
    python pipeline\diagnostics\sensitivity_scan.py
"""
import csv
import os
import sys
from datetime import date

import numpy as np
import pandas as pd

HERE = os.path.dirname(os.path.abspath(__file__))
ORCH = os.path.join(HERE, '..', 'orchestration')
sys.path.insert(0, ORCH)

import features as features_module                     # noqa: E402
from features import compute_fracdiff_feature, compute_ch19_features  # noqa: E402
from live_staging import stage_live_training_tables      # noqa: E402
from stages import load_ch11_driver, run_live_trials, evaluate_overfitting  # noqa: E402

# Reuse features.py's own private helpers rather than re-deriving them.
_retag_trades_with_bar_id = features_module._retag_trades_with_bar_id
_build_bars_with_volume = features_module._build_bars_with_volume

DEFAULT_ROLL_WINDOW = features_module.ROLL_WINDOW      # 20
DEFAULT_VPIN_WINDOW = features_module.VPIN_WINDOW       # 10
DEFAULT_FFD_THRES = features_module.FFD_THRES            # 0.01

SNAPSHOT_DIR = None   # set in main() once we know today's/the target date
SWEEP_STAGING_DIR = None
SWEEP_PLOTS_DIR = None
RESULTS_CSV = os.path.join(HERE, 'sensitivity_scan.csv')

CSV_FIELDS = [
    'run_date', 'constant', 'value', 'n_events_after', 'fracdiff_d',
    'T_raw', 'tw_mean', 'T_effective', 'dsr', 'pbo', 'skew', 'kurtosis',
    'notes',
]


def build_enriched_events_override(raw_trades, threshold, events,
                                     ffd_thres, roll_window, vpin_window):
    """Same body as features.py's build_enriched_events(), but passing
    ffd_thres/roll_window/vpin_window explicitly instead of relying on
    that function's own default-bound constants. See module LOAD-BEARING
    note above for why this replication is necessary."""
    trades = _retag_trades_with_bar_id(raw_trades, threshold)
    bars_vol = _build_bars_with_volume(trades)

    fd = compute_fracdiff_feature(bars_vol['Close'], thres=ffd_thres)
    ch19_features = compute_ch19_features(
        trades, bars_vol, roll_window=roll_window, vpin_window=vpin_window,
    )

    table = ch19_features.copy()
    if fd['d'] is not None:
        table = table.join(fd['fracdiff'], how='left')
    else:
        table['fracdiff'] = np.nan

    enriched = events.join(table, how='left')
    n_before = len(enriched)
    feature_cols = list(table.columns)
    enriched = enriched.dropna(subset=feature_cols)
    n_after = len(enriched)

    return {
        'enriched_events': enriched,
        'fracdiff_d': fd['d'],
        'n_events_before': n_before,
        'n_events_after': n_after,
        'feature_table': table,
    }


def load_snapshot(snapshot_dir):
    raw_trades = pd.read_parquet(os.path.join(snapshot_dir, 'raw_trades.parquet'))
    bars = pd.read_parquet(os.path.join(snapshot_dir, 'bars.parquet'))
    close = pd.read_parquet(os.path.join(snapshot_dir, 'close.parquet'))['close']
    events = pd.read_parquet(os.path.join(snapshot_dir, 'events.parquet'))
    w = pd.read_parquet(os.path.join(snapshot_dir, 'w.parquet'))['w']
    tw = pd.read_parquet(os.path.join(snapshot_dir, 'tw.parquet'))['tw']
    with open(os.path.join(snapshot_dir, 'threshold.txt')) as f:
        threshold = float(f.read())

    rebuild_result = {
        'bars': bars, 'close': close, 'threshold': threshold,
        'events': events, 'w': w, 'tw': tw,
    }
    return raw_trades, rebuild_result


def run_one_sweep_point(raw_trades, rebuild_result, ffd_thres, roll_window,
                          vpin_window, S, reuse_trials=None):
    """Runs feature enrichment -> staging -> Ch11 trials -> evaluate_overfitting
    for one (ffd_thres, roll_window, vpin_window, S) combination.

    reuse_trials: optional (M, meta, tw_aligned, enriched_result) tuple to
    skip re-running feature engineering/staging/trials entirely -- used for
    the S sweep, since S only affects evaluate_overfitting, not anything
    upstream of it.
    """
    ch11 = load_ch11_driver()

    if reuse_trials is not None:
        M, meta, tw_aligned, enriched_result = reuse_trials
    else:
        enriched_result = build_enriched_events_override(
            raw_trades, rebuild_result['threshold'], rebuild_result['events'],
            ffd_thres=ffd_thres, roll_window=roll_window, vpin_window=vpin_window,
        )
        stage_live_training_tables(rebuild_result, enriched_result, SWEEP_STAGING_DIR)

        M, meta = run_live_trials(ch11, SWEEP_STAGING_DIR, SWEEP_PLOTS_DIR)

        tw_aligned = rebuild_result['tw'].reindex(
            enriched_result['enriched_events'].index)
        if tw_aligned.isna().any():
            raise ValueError(
                'tw has NaN after reindexing to this sweep point\'s '
                'enriched event index -- investigate before evaluating.'
            )

    eval_result = evaluate_overfitting(M, meta, ch11, S=S, tw=tw_aligned)
    return eval_result, enriched_result, (M, meta, tw_aligned, enriched_result)


def log_failed_row(constant, value, error_message):
    """Logs a sweep point that raised an exception instead of crashing the
    whole sweep. Used specifically for FFD_THRES values whose implied FFD
    weight window is wider than this pipeline's live series has bars for
    (see 2026-08-18 CALIBRATION_AUDIT.md note -- e.g. thres=1e-5 at d=0.1
    needs ~4,075 weights against a ~238-bar series, so frac_diff_ffd()
    produces zero valid rows and adfuller() gets an empty array). This is
    a genuine finding about that constant's practical range on THIS data,
    not a bug to silently swallow -- hence logged to the CSV with notes,
    not just skipped."""
    row = {
        'run_date': date.today().isoformat(),
        'constant': constant,
        'value': value,
        'n_events_after': '', 'fracdiff_d': '',
        'T_raw': '', 'tw_mean': '', 'T_effective': '',
        'dsr': '', 'pbo': '', 'skew': '', 'kurtosis': '',
        'notes': f'FAILED: {error_message[:200]}',
    }
    file_exists = os.path.exists(RESULTS_CSV)
    with open(RESULTS_CSV, 'a', newline='') as f:
        writer = csv.DictWriter(f, fieldnames=CSV_FIELDS)
        if not file_exists:
            writer.writeheader()
        writer.writerow(row)
    print(f"  logged FAILURE: {constant}={value}  ({error_message[:120]})")


def log_row(constant, value, eval_result, enriched_result, notes=''):
    row = {
        'run_date': date.today().isoformat(),
        'constant': constant,
        'value': value,
        'n_events_after': enriched_result['n_events_after'],
        'fracdiff_d': enriched_result['fracdiff_d'],
        'T_raw': eval_result['T_raw'],
        'tw_mean': eval_result['tw_mean'],
        'T_effective': eval_result['T'],
        'dsr': eval_result['dsr'],
        'pbo': eval_result['prob_overfit'],
        'skew': eval_result['skew'],
        'kurtosis': eval_result['kurtosis'],
        'notes': notes,
    }
    file_exists = os.path.exists(RESULTS_CSV)
    with open(RESULTS_CSV, 'a', newline='') as f:
        writer = csv.DictWriter(f, fieldnames=CSV_FIELDS)
        if not file_exists:
            writer.writeheader()
        writer.writerow(row)
    print(f"  logged: {constant}={value}  "
          f"T_eff={row['T_effective']:.2f}  dsr={row['dsr']:.4f}  "
          f"pbo={row['pbo']:.4f}  n_events={row['n_events_after']}")


def main():
    global SNAPSHOT_DIR, SWEEP_STAGING_DIR, SWEEP_PLOTS_DIR

    if len(sys.argv) > 1:
        snapshot_date = sys.argv[1]
    else:
        snapshot_date = date.today().isoformat()
    SNAPSHOT_DIR = os.path.join(HERE, f'sensitivity_snapshot_{snapshot_date}')
    if not os.path.isdir(SNAPSHOT_DIR):
        raise SystemExit(
            f'{SNAPSHOT_DIR} not found. Run capture_sensitivity_snapshot.py '
            f'first, or pass the snapshot date as an argument, e.g.\n'
            f'  python sensitivity_scan.py 2026-08-18'
        )
    SWEEP_STAGING_DIR = os.path.join(SNAPSHOT_DIR, 'sweep_staging')
    SWEEP_PLOTS_DIR = os.path.join(SNAPSHOT_DIR, 'sweep_plots')

    print(f'Loading frozen snapshot from {SNAPSHOT_DIR}...')
    raw_trades, rebuild_result = load_snapshot(SNAPSHOT_DIR)
    print(f"  {len(raw_trades)} raw trades, {len(rebuild_result['bars'])} bars, "
          f"{len(rebuild_result['events'])} events (all frozen)")

    # --- Baseline (current production defaults, S=8) ---
    print('\n[baseline] ROLL_WINDOW=20 VPIN_WINDOW=10 FFD_THRES=0.01 S=8')
    baseline_eval, baseline_enriched, baseline_trials = run_one_sweep_point(
        raw_trades, rebuild_result,
        ffd_thres=DEFAULT_FFD_THRES, roll_window=DEFAULT_ROLL_WINDOW,
        vpin_window=DEFAULT_VPIN_WINDOW, S=8,
    )
    log_row('baseline', 'default', baseline_eval, baseline_enriched)

    # --- S sweep: reuse baseline's M/meta/tw_aligned, evaluate-only ---
    print('\n[S sweep] S=4 (reusing baseline trial grid, no retrain)')
    s4_eval, s4_enriched, _ = run_one_sweep_point(
        raw_trades, rebuild_result,
        ffd_thres=DEFAULT_FFD_THRES, roll_window=DEFAULT_ROLL_WINDOW,
        vpin_window=DEFAULT_VPIN_WINDOW, S=4, reuse_trials=baseline_trials,
    )
    log_row('S', 4, s4_eval, s4_enriched)
    log_row('S', 8, baseline_eval, baseline_enriched,
            notes='same run as baseline row, logged separately for S comparison')

    # --- ROLL_WINDOW sweep (others at default) ---
    for val in (10, 40):
        print(f'\n[ROLL_WINDOW={val}] (VPIN_WINDOW=10 FFD_THRES=0.01 S=8)')
        eval_result, enriched_result, _ = run_one_sweep_point(
            raw_trades, rebuild_result,
            ffd_thres=DEFAULT_FFD_THRES, roll_window=val,
            vpin_window=DEFAULT_VPIN_WINDOW, S=8,
        )
        log_row('ROLL_WINDOW', val, eval_result, enriched_result)

    # --- VPIN_WINDOW sweep (others at default) ---
    for val in (5, 20):
        print(f'\n[VPIN_WINDOW={val}] (ROLL_WINDOW=20 FFD_THRES=0.01 S=8)')
        eval_result, enriched_result, _ = run_one_sweep_point(
            raw_trades, rebuild_result,
            ffd_thres=DEFAULT_FFD_THRES, roll_window=DEFAULT_ROLL_WINDOW,
            vpin_window=val, S=8,
        )
        log_row('VPIN_WINDOW', val, eval_result, enriched_result)

    # --- FFD_THRES sweep (others at default) ---
    # *** LOAD-BEARING (2026-08-18): try/except around this leg only ***
    # A too-small thres can require an FFD weight window wider than this
    # pipeline's live series has bars for, leaving frac_diff_ffd() with
    # zero valid rows and adfuller() an empty array -- ValueError, not a
    # bug in ch05's book-fidelity code. Logged as a documented failure
    # (log_failed_row) rather than silently skipped or left to crash the
    # whole sweep, since which values are too small IS the finding.
    for val in (1e-5, 0.05):
        print(f'\n[FFD_THRES={val}] (ROLL_WINDOW=20 VPIN_WINDOW=10 S=8)')
        try:
            eval_result, enriched_result, _ = run_one_sweep_point(
                raw_trades, rebuild_result,
                ffd_thres=val, roll_window=DEFAULT_ROLL_WINDOW,
                vpin_window=DEFAULT_VPIN_WINDOW, S=8,
            )
            log_row('FFD_THRES', val, eval_result, enriched_result)
        except ValueError as e:
            log_failed_row('FFD_THRES', val, str(e))

    print(f'\nAll sweep points appended to {RESULTS_CSV}')


if __name__ == '__main__':
    main()