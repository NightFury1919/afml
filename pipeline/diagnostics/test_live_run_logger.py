"""
pipeline/diagnostics/test_live_run_logger.py

TDD test suite for live_run_logger.py (added 2026-08-19). Two functions
under test, both driven from the same `row` dict per the module's own
docstring ("can never drift out of sync"):

  - log_live_run(csv_path, row): append-only CSV writer with a fail-loud
    header-mismatch guard.
  - write_snapshot_readme(snapshot_dir, row, files_written): narrative
    README generator.

Two of write_snapshot_readme's lines were real bugs caught and fixed
during this session (2026-08-19), by cross-checking the first generated
README against actual console report text, not assumed correct:
  1. The Ch15 events-survived line used n_events_enriched where it should
     use n_events (the raw triple-barrier count) for the denominator/
     "on N real bets" figure.
  2. freq_real was interpolated raw instead of through _fmt().
Both are pinned here as explicit regression tests so a future refactor
can't silently reintroduce either.

Run (two-pass, per project convention):
    From repo root:               pytest pipeline/diagnostics/test_live_run_logger.py -v
    From pipeline/diagnostics:    pytest test_live_run_logger.py -v
"""
import csv
import os
import sys

import pytest

HERE = os.path.dirname(os.path.abspath(__file__))
if HERE not in sys.path:
    sys.path.insert(0, HERE)

import live_run_logger  # real module under test


# ---------------------------------------------------------------------
# Shared hand-traced fixture -- a full, realistic row dict matching every
# key LIVE_RUN_LOG_COLUMNS and write_snapshot_readme's f-strings read
# ---------------------------------------------------------------------

def _make_row(**overrides):
    row = {
        'run_date': '2026-08-19',
        'n_raw_trades': 99878,
        'n_bars': 238,
        'n_events': 48,
        'n_events_enriched': 47,
        'fracdiff_d': 0.3,
        'S': 12,
        'n_trials': 20,
        'T_raw': 156,
        'tw_mean': 0.4252,
        'T_effective': 66.3266,
        'best_trial': 'C0.1_s0.05',
        'best_sharpe': -0.0049,
        'pbo': 0.4784,
        'dsr': 0.2136,
        'skew': -0.531,
        'kurtosis': 8.1,
        'phi_hat': 0.9673,
        'phi_stationary': True,
        'half_life': 20.9,
        'p_fail': 0.4654,
        'realized_precision': 0.5417,
        'freq_real': 628.3775591952092,
        'lifecycle_stage': 'EMBARGO',
        'position_size': 0.0,
        'notes': '',
    }
    row.update(overrides)
    return row


# =======================================================================
# log_live_run
# =======================================================================

def test_writes_header_and_row_when_file_does_not_exist(tmp_path):
    csv_path = str(tmp_path / 'live_run_log.csv')
    live_run_logger.log_live_run(csv_path, _make_row())

    with open(csv_path, newline='') as f:
        rows = list(csv.reader(f))

    assert rows[0] == live_run_logger.LIVE_RUN_LOG_COLUMNS
    assert len(rows) == 2  # header + one data row


def test_appends_without_duplicating_header_when_file_exists(tmp_path):
    csv_path = str(tmp_path / 'live_run_log.csv')
    live_run_logger.log_live_run(csv_path, _make_row(run_date='2026-08-19'))
    live_run_logger.log_live_run(csv_path, _make_row(run_date='2026-08-20'))

    with open(csv_path, newline='') as f:
        rows = list(csv.reader(f))

    assert rows[0] == live_run_logger.LIVE_RUN_LOG_COLUMNS
    assert len(rows) == 3  # header + two data rows
    run_date_idx = live_run_logger.LIVE_RUN_LOG_COLUMNS.index('run_date')
    assert rows[1][run_date_idx] == '2026-08-19'
    assert rows[2][run_date_idx] == '2026-08-20'


def test_column_order_matches_row_values_exactly(tmp_path):
    """Hand-trace: row['n_events']=48 must land in the 'n_events' column
    position, not shifted by a dict-ordering assumption -- writer must
    build the row via [row.get(col, '') for col in LIVE_RUN_LOG_COLUMNS],
    not row.values()."""
    csv_path = str(tmp_path / 'live_run_log.csv')
    live_run_logger.log_live_run(csv_path, _make_row())

    with open(csv_path, newline='') as f:
        rows = list(csv.reader(f))
    header, data = rows[0], rows[1]

    n_events_idx = header.index('n_events')
    dsr_idx = header.index('dsr')
    assert data[n_events_idx] == '48'
    assert data[dsr_idx] == '0.2136'


def test_missing_keys_written_as_empty_string(tmp_path):
    """A run can still deserve logging even if e.g. risk_context.py
    failed and half_life is unavailable (module docstring)."""
    csv_path = str(tmp_path / 'live_run_log.csv')
    row = _make_row()
    del row['half_life']
    live_run_logger.log_live_run(csv_path, row)

    with open(csv_path, newline='') as f:
        rows = list(csv.reader(f))
    header, data = rows[0], rows[1]
    assert data[header.index('half_life')] == ''


def test_extra_keys_in_row_are_silently_ignored(tmp_path):
    csv_path = str(tmp_path / 'live_run_log.csv')
    row = _make_row()
    row['some_future_field_not_in_schema_yet'] = 'surprise'
    live_run_logger.log_live_run(csv_path, row)  # must not raise

    with open(csv_path, newline='') as f:
        rows = list(csv.reader(f))
    assert len(rows[1]) == len(live_run_logger.LIVE_RUN_LOG_COLUMNS)


# ---------------------------------------------------------------------
# LOAD-BEARING (2026-08-19): fail-loud header mismatch guard
# ---------------------------------------------------------------------

def test_raises_on_header_mismatch_with_existing_file(tmp_path):
    csv_path = str(tmp_path / 'live_run_log.csv')
    with open(csv_path, 'w', newline='') as f:
        csv.writer(f).writerow(['run_date', 'some_old_schema_column'])

    with pytest.raises(ValueError, match='does not match'):
        live_run_logger.log_live_run(csv_path, _make_row())


def test_header_mismatch_raise_does_not_corrupt_existing_file(tmp_path):
    csv_path = str(tmp_path / 'live_run_log.csv')
    with open(csv_path, 'w', newline='') as f:
        csv.writer(f).writerow(['run_date', 'some_old_schema_column'])
        csv.writer(f).writerow(['2026-08-01', 'preexisting_value'])

    with pytest.raises(ValueError):
        live_run_logger.log_live_run(csv_path, _make_row())

    with open(csv_path, newline='') as f:
        rows = list(csv.reader(f))
    assert rows == [
        ['run_date', 'some_old_schema_column'],
        ['2026-08-01', 'preexisting_value'],
    ]


def test_raises_on_preexisting_empty_file(tmp_path):
    """A 0-byte file that already exists (e.g. an accidental touch/
    New-Item with no content -- exactly the failure mode found and fixed
    this same session with CALIBRATION_AUDIT.md) must be treated as a
    header mismatch (existing_header=None != LIVE_RUN_LOG_COLUMNS), not
    silently written into as if it were fresh."""
    csv_path = str(tmp_path / 'live_run_log.csv')
    open(csv_path, 'w').close()  # create a real, genuinely empty file
    assert os.path.getsize(csv_path) == 0

    with pytest.raises(ValueError, match='does not match'):
        live_run_logger.log_live_run(csv_path, _make_row())


def test_matching_existing_header_does_not_raise(tmp_path):
    csv_path = str(tmp_path / 'live_run_log.csv')
    with open(csv_path, 'w', newline='') as f:
        csv.writer(f).writerow(live_run_logger.LIVE_RUN_LOG_COLUMNS)

    live_run_logger.log_live_run(csv_path, _make_row())  # must not raise
    with open(csv_path, newline='') as f:
        rows = list(csv.reader(f))
    assert len(rows) == 2


# =======================================================================
# _fmt / _fmt_pct helpers
# =======================================================================

def test_fmt_default_four_decimals():
    assert live_run_logger._fmt(0.2136) == '0.2136'


def test_fmt_respects_nd_argument():
    """LOAD-BEARING regression (bug #2, 2026-08-19): freq_real must go
    through _fmt(x, 1), not be interpolated raw -- 628.3775591952092
    must render as '628.4', not the full float repr."""
    assert live_run_logger._fmt(628.3775591952092, 1) == '628.4'


def test_fmt_passes_through_non_numeric_as_str():
    assert live_run_logger._fmt('EMBARGO') == 'EMBARGO'
    assert live_run_logger._fmt(None) == 'None'


def test_fmt_pct_default_two_decimals():
    assert live_run_logger._fmt_pct(0.4784) == '47.84%'


def test_fmt_pct_passes_through_non_numeric_as_str():
    assert live_run_logger._fmt_pct('?') == '?'


# =======================================================================
# write_snapshot_readme
# =======================================================================

def test_creates_snapshot_dir_if_missing(tmp_path):
    snapshot_dir = str(tmp_path / 'not_created_yet' / '2026-08-19')
    assert not os.path.exists(snapshot_dir)

    live_run_logger.write_snapshot_readme(snapshot_dir, _make_row(), [])
    assert os.path.exists(snapshot_dir)
    assert os.path.exists(os.path.join(snapshot_dir, 'README.md'))


def test_returns_the_path_written(tmp_path):
    snapshot_dir = str(tmp_path / '2026-08-19')
    returned = live_run_logger.write_snapshot_readme(snapshot_dir, _make_row(), [])
    assert returned == os.path.join(snapshot_dir, 'README.md')


def test_header_line_uses_run_date(tmp_path):
    snapshot_dir = str(tmp_path / '2026-08-19')
    live_run_logger.write_snapshot_readme(snapshot_dir, _make_row(), [])
    with open(os.path.join(snapshot_dir, 'README.md'), encoding='utf-8') as f:
        lines = f.read().split('\n')
    assert lines[0] == '# Live pipeline run -- 2026-08-19'


# ---------------------------------------------------------------------
# LOAD-BEARING (2026-08-19): bug #1 regression -- the Ch15 line's event
# count and denominator must use n_events (raw), NOT n_events_enriched
# ---------------------------------------------------------------------

def test_ch15_line_uses_raw_n_events_not_enriched(tmp_path):
    """Hand-traced: row has n_events=48, n_events_enriched=47 (deliberately
    different, matching the real 47/48 split observed 2026-08-19). The
    Ch15 line's '... on N real bets' must read 48, not 47 -- Ch15's
    P[fail]/realized_precision are computed over the raw triple-barrier
    event count, not the post-fracdiff-dropna enriched subset."""
    snapshot_dir = str(tmp_path / '2026-08-19')
    live_run_logger.write_snapshot_readme(snapshot_dir, _make_row(), [])
    with open(os.path.join(snapshot_dir, 'README.md'), encoding='utf-8') as f:
        content = f.read()

    ch15_line = next(l for l in content.split('\n') if l.strip().startswith('Ch15:'))
    assert 'on 48 real bets' in ch15_line
    assert 'on 47 real bets' not in ch15_line


def test_events_survived_enrichment_line_hand_traced(tmp_path):
    """The OTHER real line that legitimately uses n_events_enriched --
    distinguishing it from bug #1 above confirms the fix targeted the
    right line rather than removing n_events_enriched everywhere."""
    snapshot_dir = str(tmp_path / '2026-08-19')
    live_run_logger.write_snapshot_readme(snapshot_dir, _make_row(), [])
    with open(os.path.join(snapshot_dir, 'README.md'), encoding='utf-8') as f:
        content = f.read()

    enrichment_line = next(
        l for l in content.split('\n') if 'survived' in l
    )
    assert enrichment_line.strip().startswith('47/48 events survived')
    assert 'fracdiff d=0.3' in enrichment_line


# ---------------------------------------------------------------------
# LOAD-BEARING (2026-08-19): bug #2 regression -- freq_real must be
# formatted through _fmt(), not interpolated raw
# ---------------------------------------------------------------------

def test_freq_real_line_is_formatted_not_raw(tmp_path):
    """row['freq_real']=628.3775591952092 (a real observed unformatted
    value from this session) must render as '628.4', never the raw
    float repr."""
    snapshot_dir = str(tmp_path / '2026-08-19')
    live_run_logger.write_snapshot_readme(snapshot_dir, _make_row(), [])
    with open(os.path.join(snapshot_dir, 'README.md'), encoding='utf-8') as f:
        content = f.read()

    ch15_line = next(l for l in content.split('\n') if l.strip().startswith('Ch15:'))
    assert '~628.4 bets/year' in ch15_line
    assert '628.3775591952092' not in content


# ---------------------------------------------------------------------
# T_effective / tw_mean / sharpe / dsr / phi_hat lines -- hand-traced
# ---------------------------------------------------------------------

def test_t_effective_line_hand_traced(tmp_path):
    snapshot_dir = str(tmp_path / '2026-08-19')
    live_run_logger.write_snapshot_readme(snapshot_dir, _make_row(), [])
    with open(os.path.join(snapshot_dir, 'README.md'), encoding='utf-8') as f:
        content = f.read()

    line = next(l for l in content.split('\n') if l.strip().startswith('S='))
    assert line.strip() == (
        'S=12, n_trials=20, T_effective=66.3266 (T_raw=156 x tw_mean=0.4252)'
    )


def test_best_trial_line_hand_traced(tmp_path):
    snapshot_dir = str(tmp_path / '2026-08-19')
    live_run_logger.write_snapshot_readme(snapshot_dir, _make_row(), [])
    with open(os.path.join(snapshot_dir, 'README.md'), encoding='utf-8') as f:
        content = f.read()

    line = next(l for l in content.split('\n') if l.strip().startswith('best trial:'))
    assert line.strip() == 'best trial: C0.1_s0.05, sharpe=-0.0049'


def test_pbo_dsr_line_hand_traced(tmp_path):
    snapshot_dir = str(tmp_path / '2026-08-19')
    live_run_logger.write_snapshot_readme(snapshot_dir, _make_row(), [])
    with open(os.path.join(snapshot_dir, 'README.md'), encoding='utf-8') as f:
        content = f.read()

    line = next(l for l in content.split('\n') if l.strip().startswith('PBO'))
    assert line.strip() == 'PBO 47.84%, DSR 0.2136 -- lifecycle stage: EMBARGO'


# ---------------------------------------------------------------------
# half_life: present vs absent
# ---------------------------------------------------------------------

def test_otr_line_includes_half_life_when_present(tmp_path):
    snapshot_dir = str(tmp_path / '2026-08-19')
    live_run_logger.write_snapshot_readme(snapshot_dir, _make_row(half_life=20.9), [])
    with open(os.path.join(snapshot_dir, 'README.md'), encoding='utf-8') as f:
        content = f.read()

    line = next(l for l in content.split('\n') if l.strip().startswith('OTR:'))
    assert line.strip() == 'OTR: phi_hat=0.9673, stationary=True, half-life=20.9 bars'


def test_otr_line_omits_half_life_when_none(tmp_path):
    snapshot_dir = str(tmp_path / '2026-08-19')
    live_run_logger.write_snapshot_readme(
        snapshot_dir, _make_row(half_life=None), []
    )
    with open(os.path.join(snapshot_dir, 'README.md'), encoding='utf-8') as f:
        content = f.read()

    line = next(l for l in content.split('\n') if l.strip().startswith('OTR:'))
    assert line.strip() == 'OTR: phi_hat=0.9673, stationary=True'
    assert 'half-life' not in line


def test_otr_line_omits_half_life_when_empty_string(tmp_path):
    """row.get('half_life') can legitimately be '' (e.g. logged via
    log_live_run's missing-key convention) -- must be treated the same
    as None, not rendered as 'half-life= bars'."""
    snapshot_dir = str(tmp_path / '2026-08-19')
    live_run_logger.write_snapshot_readme(
        snapshot_dir, _make_row(half_life=''), []
    )
    with open(os.path.join(snapshot_dir, 'README.md'), encoding='utf-8') as f:
        content = f.read()

    line = next(l for l in content.split('\n') if l.strip().startswith('OTR:'))
    assert 'half-life' not in line


# ---------------------------------------------------------------------
# notes: present vs absent
# ---------------------------------------------------------------------

def test_notes_line_included_when_present(tmp_path):
    snapshot_dir = str(tmp_path / '2026-08-19')
    live_run_logger.write_snapshot_readme(
        snapshot_dir, _make_row(notes='fracdiff d=0 this run, see caveat'), []
    )
    with open(os.path.join(snapshot_dir, 'README.md'), encoding='utf-8') as f:
        content = f.read()
    assert 'fracdiff d=0 this run, see caveat' in content


def test_notes_line_omitted_when_empty(tmp_path):
    """Hand-traced: with notes='', the Ch15 line must be followed by
    exactly one blank line and then 'Files:' -- no extra notes block
    inserted in between."""
    snapshot_dir = str(tmp_path / '2026-08-19')
    live_run_logger.write_snapshot_readme(snapshot_dir, _make_row(notes=''), [])
    with open(os.path.join(snapshot_dir, 'README.md'), encoding='utf-8') as f:
        lines = f.read().split('\n')

    ch15_idx = next(i for i, l in enumerate(lines) if l.strip().startswith('Ch15:'))
    assert lines[ch15_idx + 1] == ''
    assert lines[ch15_idx + 2] == 'Files:'


# ---------------------------------------------------------------------
# Files: section -- explicit list, never os.listdir()'d
# ---------------------------------------------------------------------

def test_files_section_lists_exactly_passed_tuples(tmp_path):
    snapshot_dir = str(tmp_path / '2026-08-19')
    files_written = [
        ('ch07_training_table_enriched.csv', 'staged live training table'),
        ('extra_artifact.png', 'a diagnostic plot'),
    ]
    live_run_logger.write_snapshot_readme(snapshot_dir, _make_row(), files_written)
    with open(os.path.join(snapshot_dir, 'README.md'), encoding='utf-8') as f:
        content = f.read()

    assert '  ch07_training_table_enriched.csv  staged live training table' in content
    assert '  extra_artifact.png  a diagnostic plot' in content


def test_files_section_ignores_stray_files_actually_on_disk(tmp_path):
    """Module docstring: passed explicitly 'so a stray file in the
    directory can't leak into the README as if it were a deliberate
    part of the snapshot' -- a real extra file sitting in snapshot_dir
    that ISN'T in files_written must not appear."""
    snapshot_dir = tmp_path / '2026-08-19'
    snapshot_dir.mkdir(parents=True)
    (snapshot_dir / 'accidental_leftover.tmp').write_text('junk')

    live_run_logger.write_snapshot_readme(str(snapshot_dir), _make_row(), [])
    with open(snapshot_dir / 'README.md', encoding='utf-8') as f:
        content = f.read()
    assert 'accidental_leftover.tmp' not in content


def test_files_section_empty_list_still_writes_files_header(tmp_path):
    snapshot_dir = str(tmp_path / '2026-08-19')
    live_run_logger.write_snapshot_readme(snapshot_dir, _make_row(), [])
    with open(os.path.join(snapshot_dir, 'README.md'), encoding='utf-8') as f:
        content = f.read()
    assert 'Files:' in content

# =============================================================================
# TDD VERIFICATION -- pytest results, real-machine-confirmed 2026-08-19
# (mlfinlab env: Python 3.10.20, pytest 9.0.3)
# =============================================================================
# Two-pass run (per project convention):
#
# PASS 1 -- from repo root (pytest pipeline\diagnostics\test_live_run_logger.py -v):
#   test_writes_header_and_row_when_file_does_not_exist PASSED
#   test_appends_without_duplicating_header_when_file_exists PASSED
#   test_column_order_matches_row_values_exactly PASSED
#   test_missing_keys_written_as_empty_string PASSED
#   test_extra_keys_in_row_are_silently_ignored PASSED
#   test_raises_on_header_mismatch_with_existing_file PASSED
#   test_header_mismatch_raise_does_not_corrupt_existing_file PASSED
#   test_raises_on_preexisting_empty_file PASSED
#   test_matching_existing_header_does_not_raise PASSED
#   test_fmt_default_four_decimals PASSED
#   test_fmt_respects_nd_argument PASSED
#   test_fmt_passes_through_non_numeric_as_str PASSED
#   test_fmt_pct_default_two_decimals PASSED
#   test_fmt_pct_passes_through_non_numeric_as_str PASSED
#   test_creates_snapshot_dir_if_missing PASSED
#   test_returns_the_path_written PASSED
#   test_header_line_uses_run_date PASSED
#   test_ch15_line_uses_raw_n_events_not_enriched PASSED
#   test_events_survived_enrichment_line_hand_traced PASSED
#   test_freq_real_line_is_formatted_not_raw PASSED
#   test_t_effective_line_hand_traced PASSED
#   test_best_trial_line_hand_traced PASSED
#   test_pbo_dsr_line_hand_traced PASSED
#   test_otr_line_includes_half_life_when_present PASSED
#   test_otr_line_omits_half_life_when_none PASSED
#   test_otr_line_omits_half_life_when_empty_string PASSED
#   test_notes_line_included_when_present PASSED
#   test_notes_line_omitted_when_empty PASSED
#   test_files_section_lists_exactly_passed_tuples PASSED
#   test_files_section_ignores_stray_files_actually_on_disk PASSED
#   test_files_section_empty_list_still_writes_files_header PASSED
#   31 passed in 0.45s
#
# PASS 2 -- from pipeline\diagnostics\ (pytest test_live_run_logger.py -v):
#   Same 31 tests, all PASSED, 31 passed in 0.31s
#
# No bugs found in live_run_logger.py during this test suite's real-machine
# confirmation -- both real bugs this module had (n_events_enriched vs
# n_events on the Ch15 line; freq_real unformatted) were already caught and
# fixed during today's earlier auto-logging deployment work, before this
# test suite was written. This suite pins both as explicit regressions
# (test_ch15_line_uses_raw_n_events_not_enriched,
# test_freq_real_line_is_formatted_not_raw) so a future refactor can't
# silently reintroduce either.
# =============================================================================