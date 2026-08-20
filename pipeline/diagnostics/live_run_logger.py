"""
pipeline/diagnostics/live_run_logger.py

Auto-logs every run_pipeline_live.py execution into two places:
  1. live_run_log.csv -- one appended row per run, same tidy schema as
     this project's other diagnostics/*.csv files (sensitivity_scan.csv,
     pbo_precision_calibration.csv): header row once, comma-separated
     floats after, freeform 'notes' column last.
  2. live_run_examples/YYYY-MM-DD/README.md -- narrative snapshot in the
     same style as the hand-written 2026-08-14 example, auto-generated
     instead. Only written when the caller explicitly opts in (see
     run_pipeline_live.py's --snapshot flag) -- per the standing
     live_run_examples/ convention, these are OCCASIONAL frozen
     artifacts, not a write-every-run log. That's what live_run_log.csv
     is for.

Both functions take the SAME `row` dict, so the CSV row and the README's
numbers can never drift out of sync with each other.

Date added: 2026-08-19.
"""
import csv
import os

LIVE_RUN_LOG_COLUMNS = [
    'run_date', 'n_raw_trades', 'n_bars', 'n_events', 'n_events_enriched',
    'fracdiff_d', 'S', 'n_trials', 'T_raw', 'tw_mean', 'T_effective',
    'best_trial', 'best_sharpe', 'pbo', 'dsr', 'skew', 'kurtosis',
    'phi_hat', 'phi_stationary', 'half_life', 'p_fail',
    'realized_precision', 'freq_real', 'lifecycle_stage', 'position_size',
    'notes',
]


def log_live_run(csv_path, row):
    """Append one row to live_run_log.csv, writing the header first if the
    file doesn't exist yet.

    `row` : dict. Missing keys are written as '' rather than raising --
    the run still deserves to be logged even if e.g. risk_context.py
    failed and one field (say, half_life) is unavailable. Extra keys in
    `row` not in LIVE_RUN_LOG_COLUMNS are silently ignored.

    *** LOAD-BEARING (2026-08-19): if the file already exists, its header
    is checked against LIVE_RUN_LOG_COLUMNS before appending. A mismatch
    raises rather than silently writing misaligned columns into an
    existing log -- same fail-loud principle as this project's tw/w
    reindex checks elsewhere. If you deliberately change the schema,
    either migrate the old file's header by hand or archive it and start
    a new one -- don't let this check start silently passing on a
    genuinely different schema. ***
    """
    file_exists = os.path.exists(csv_path)

    if file_exists:
        with open(csv_path, 'r', newline='') as f:
            existing_header = next(csv.reader(f), None)
        if existing_header != LIVE_RUN_LOG_COLUMNS:
            raise ValueError(
                "live_run_log.csv's existing header does not match this "
                "module's LIVE_RUN_LOG_COLUMNS -- refusing to append "
                f"misaligned columns.\n  existing: {existing_header}\n"
                f"  expected: {LIVE_RUN_LOG_COLUMNS}\n"
                "Migrate the old file's header or archive it before "
                "changing the schema."
            )

    with open(csv_path, 'a', newline='') as f:
        writer = csv.writer(f)
        if not file_exists:
            writer.writerow(LIVE_RUN_LOG_COLUMNS)
        writer.writerow([row.get(col, '') for col in LIVE_RUN_LOG_COLUMNS])


def _fmt(x, nd=4):
    return f'{x:.{nd}f}' if isinstance(x, (int, float)) else str(x)


def _fmt_pct(x, nd=2):
    return f'{x * 100:.{nd}f}%' if isinstance(x, (int, float)) else str(x)


def write_snapshot_readme(snapshot_dir, row, files_written):
    """Write a README.md into snapshot_dir, narrating `row` in the same
    style as the hand-written live_run_examples/2026-08-14/README.md.

    Parameters
    ----------
    snapshot_dir : str, e.g. pipeline/live_run_examples/2026-08-19/
    row : dict, the SAME dict passed to log_live_run() for this run
    files_written : list of (filename, one-line description) tuples for
        the "Files:" section. Passed explicitly by the caller (which
        knows exactly what it copied into snapshot_dir) rather than
        os.listdir()'d, so a stray file in the directory can't leak into
        the README as if it were a deliberate part of the snapshot.

    Returns
    -------
    str, the path written.
    """
    os.makedirs(snapshot_dir, exist_ok=True)
    run_date = row.get('run_date', '')

    lines = [
        f'# Live pipeline run -- {run_date}',
        '',
        'Frozen snapshot of one real end-to-end run of '
        '`pipeline/run_pipeline_live.py` against a live 720h (30-day) '
        'Binance.US BTCUSDT pull. This is NOT live/current data -- it is '
        'a point-in-time example proving the live pipeline (ingestion -> '
        "rebuild -> features -> Ch11's real trial construction -> "
        'PBO/DSR -> report) runs end-to-end on real, independently-pulled '
        'data. See pipeline/README.md for how to run a fresh live '
        'pipeline yourself.',
        '',
        'Real numbers from this specific run:',
        f"  {row.get('n_raw_trades', '?')} raw trades, "
        f"{row.get('n_bars', '?')} bars, {row.get('n_events', '?')} "
        'triple-barrier events',
        f"  {row.get('n_events_enriched', '?')}/{row.get('n_events', '?')} "
        f"events survived feature enrichment (fracdiff "
        f"d={row.get('fracdiff_d', '?')})",
        f"  S={row.get('S', '?')}, n_trials={row.get('n_trials', '?')}, "
        f"T_effective={_fmt(row.get('T_effective'))} "
        f"(T_raw={row.get('T_raw', '?')} x "
        f"tw_mean={_fmt(row.get('tw_mean'))})",
        f"  best trial: {row.get('best_trial', '?')}, "
        f"sharpe={_fmt(row.get('best_sharpe'))}",
        f"  PBO {_fmt_pct(row.get('pbo'))}, DSR {_fmt(row.get('dsr'))} -- "
        f"lifecycle stage: {row.get('lifecycle_stage', '?')}",
        f"  OTR: phi_hat={_fmt(row.get('phi_hat'))}, stationary="
        f"{row.get('phi_stationary', '?')}"
        + (f", half-life={_fmt(row.get('half_life'), 1)} bars"
           if row.get('half_life') not in (None, '') else ''),
        f"  Ch15: P[fail]={_fmt(row.get('p_fail'))}, realized precision="
        f"{_fmt_pct(row.get('realized_precision'))} on "
        f"{row.get('n_events', '?')} real bets, annualized to "
        f"~{_fmt(row.get('freq_real'), 1)} bets/year",
    ]
    if row.get('notes'):
        lines += ['', f"  {row['notes']}"]
    lines += ['', 'Files:']
    for fname, desc in files_written:
        lines.append(f'  {fname}  {desc}')
    lines.append('')

    readme_path = os.path.join(snapshot_dir, 'README.md')
    with open(readme_path, 'w', encoding='utf-8', newline='\n') as f:
        f.write('\n'.join(lines))
    return readme_path