"""
pipeline/orchestration/test_live_staging.py

TDD test suite for live_staging.py. This module is pure glue (one function,
stage_live_training_tables) around real upstream results from rebuild.py
and features.py -- so, following this project's established philosophy for
thin orchestration wrappers (see test_risk_context.py), tests hand-trace
THIS module's own logic exactly: the trgt/ret exclusion, the w reindex-
by-label (not position) onto the post-enrichment index, the NaN-after-
reindex guard, the exact CSV filenames/schema Ch11's part_c_build_trials()
hard-loads, and that ch05_features.csv's 'close' column is the FULL
rebuild.py close series, not reindexed down to the enriched subset.

rebuild_result/enriched_result inputs are minimal hand-built dicts with
only the keys stage_live_training_tables() actually reads ('w', 'close'
from rebuild_result; 'enriched_events', 'feature_table' from
enriched_result) -- not full rebuild.py/features.py outputs, matching
this project's principle of isolating the module under test.

Run (two-pass, per project convention):
    From repo root:              pytest pipeline/orchestration/test_live_staging.py -v
    From pipeline/orchestration: pytest test_live_staging.py -v
"""
import os
import sys

import pandas as pd
import pytest

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.abspath(os.path.join(HERE, '..', '..'))
if ROOT not in sys.path:
    sys.path.insert(0, ROOT)
if HERE not in sys.path:
    sys.path.insert(0, HERE)

import live_staging  # real module under test


# ---------------------------------------------------------------------
# Shared hand-traced fixture
# ---------------------------------------------------------------------

def _make_inputs():
    """3 enriched events (2026-01-01 00:00/01:00/02:00), 2 real feature
    columns (featA, featB), t1/trgt/ret/bin exactly as rebuild.py +
    features.py's join would produce. rebuild_result['w'] is indexed to
    a 4-timestamp PRE-enrichment superset (00:00-03:00), deliberately in
    SHUFFLED order and with an EXTRA row (03:00) that enrichment's dropna
    would have dropped -- this is the real shape: build_enriched_events()
    only ever DROPS rows relative to rebuild.py's own event index, and
    w's positional order has no guaranteed relationship to enriched's
    index order. rebuild_result['close'] is a SEPARATE, longer (5-bar)
    series spanning 00:00-04:00, since ch05_features.csv's 'close' column
    is documented to come from the full bar series, not the event subset.
    """
    enriched_idx = pd.to_datetime(
        ['2026-01-01 00:00', '2026-01-01 01:00', '2026-01-01 02:00']
    )
    enriched_events = pd.DataFrame(
        {
            't1': pd.to_datetime(
                ['2026-01-01 02:00', '2026-01-01 03:00', '2026-01-01 04:00']
            ),
            'trgt': [0.01, 0.02, 0.015],
            'ret': [0.005, -0.01, 0.02],
            'bin': [1, 0, 1],
            'featA': [10.0, 20.0, 30.0],
            'featB': [100.0, 200.0, 300.0],
        },
        index=enriched_idx,
    )
    feature_table = pd.DataFrame(
        {'featA': [1.0, 2.0], 'featB': [3.0, 4.0]}
    )  # only .columns is read by the module under test; values unused

    # shuffled order + one extra pre-enrichment row (03:00) that dropna
    # would have removed -- tests label-based reindex, not positional
    w = pd.Series(
        [0.7, 0.5, 0.9, 0.6],
        index=pd.to_datetime(
            ['2026-01-01 02:00', '2026-01-01 00:00',
             '2026-01-01 03:00', '2026-01-01 01:00']
        ),
    )

    close = pd.Series(
        [100.0, 101.0, 102.0, 103.0, 104.0],
        index=pd.to_datetime([
            '2026-01-01 00:00', '2026-01-01 01:00', '2026-01-01 02:00',
            '2026-01-01 03:00', '2026-01-01 04:00',
        ]),
    )

    rebuild_result = {'w': w, 'close': close}
    enriched_result = {
        'enriched_events': enriched_events,
        'feature_table': feature_table,
    }
    return rebuild_result, enriched_result


# ---------------------------------------------------------------------
# Filenames, shapes, return dict
# ---------------------------------------------------------------------

def test_writes_exact_filenames_ch11_hard_loads(tmp_path):
    """Ch11's part_c_build_trials() hard-loads these two exact filenames
    from its own module-level INPUT constant (see stages.py docstring) --
    any deviation would silently break run_live_trials()."""
    rebuild_result, enriched_result = _make_inputs()
    out = live_staging.stage_live_training_tables(
        rebuild_result, enriched_result, str(tmp_path)
    )

    assert out['enriched_csv_path'] == str(tmp_path / 'ch07_training_table_enriched.csv')
    assert out['features_csv_path'] == str(tmp_path / 'ch05_features.csv')
    assert os.path.exists(out['enriched_csv_path'])
    assert os.path.exists(out['features_csv_path'])


def test_return_dict_n_events_and_feature_cols(tmp_path):
    rebuild_result, enriched_result = _make_inputs()
    out = live_staging.stage_live_training_tables(
        rebuild_result, enriched_result, str(tmp_path)
    )
    assert out['n_events'] == 3
    assert out['feature_cols'] == ['featA', 'featB']


def test_creates_out_dir_if_missing(tmp_path):
    rebuild_result, enriched_result = _make_inputs()
    missing_dir = tmp_path / 'not_created_yet' / 'nested'
    assert not missing_dir.exists()

    live_staging.stage_live_training_tables(
        rebuild_result, enriched_result, str(missing_dir)
    )
    assert missing_dir.exists()
    assert (missing_dir / 'ch07_training_table_enriched.csv').exists()


# ---------------------------------------------------------------------
# LOAD-BEARING (2026-08-14): trgt/ret dropped, hand-traced column set
# ---------------------------------------------------------------------

def test_training_table_columns_are_exactly_t1_bin_features_w(tmp_path):
    """Deliberate exclusion of trgt/ret (module's own LOAD-BEARING note):
    ch11's stages.py derives feature_cols as everything NOT in
    ('bin','w','t1') -- if trgt/ret leaked through, they'd be silently
    fed into the SVC as bogus features."""
    rebuild_result, enriched_result = _make_inputs()
    out = live_staging.stage_live_training_tables(
        rebuild_result, enriched_result, str(tmp_path)
    )
    df = pd.read_csv(out['enriched_csv_path'], index_col=0)

    assert list(df.columns) == ['t1', 'bin', 'featA', 'featB', 'w']
    assert 'trgt' not in df.columns
    assert 'ret' not in df.columns


def test_training_table_hand_traced_values(tmp_path):
    """Full round-trip check against hand-derived expected values --
    t1/bin/features carried through unchanged, w correctly reindexed by
    LABEL (not position) from the shuffled pre-enrichment Series onto
    the enriched (00:00, 01:00, 02:00) index: expected [0.5, 0.6, 0.7]."""
    rebuild_result, enriched_result = _make_inputs()
    out = live_staging.stage_live_training_tables(
        rebuild_result, enriched_result, str(tmp_path)
    )
    df = pd.read_csv(out['enriched_csv_path'], index_col=0, parse_dates=True)

    assert df['bin'].tolist() == [1, 0, 1]
    assert df['featA'].tolist() == [10.0, 20.0, 30.0]
    assert df['featB'].tolist() == [100.0, 200.0, 300.0]
    # label-based reindex: w's shuffled index [02:00,00:00,03:00,01:00]
    # with values [0.7,0.5,0.9,0.6] -> onto enriched's [00:00,01:00,02:00]
    # order must give [0.5, 0.6, 0.7], NOT a positional slice like
    # [0.7, 0.5, 0.9] (the first 3 values in w's own stored order)
    assert df['w'].tolist() == pytest.approx([0.5, 0.6, 0.7])
    expected_t1 = pd.to_datetime(
        ['2026-01-01 02:00', '2026-01-01 03:00', '2026-01-01 04:00']
    )
    assert list(pd.to_datetime(df['t1'])) == list(expected_t1)


# ---------------------------------------------------------------------
# LOAD-BEARING (2026-08-14): w reindexed to POST-enrichment index; NaN
# after reindex must raise loudly, not stage silently-misaligned weights
# ---------------------------------------------------------------------

def test_raises_when_w_missing_a_label_in_enriched_index(tmp_path):
    """An enriched event with no matching rebuild.py event should be
    impossible per the module's own docstring -- if it happens anyway
    (e.g. an upstream index bug), this must raise loudly rather than
    silently stage a NaN sample weight."""
    rebuild_result, enriched_result = _make_inputs()
    # drop the 01:00 label from w entirely -> reindex produces NaN there
    rebuild_result['w'] = rebuild_result['w'].drop(
        pd.Timestamp('2026-01-01 01:00')
    )

    with pytest.raises(ValueError, match='NaN after reindexing'):
        live_staging.stage_live_training_tables(
            rebuild_result, enriched_result, str(tmp_path)
        )


def test_no_files_left_behind_shape_after_raise(tmp_path):
    """Not a hard requirement of the module (no cleanup logic exists),
    but documents current behavior: the ValueError fires before either
    CSV is written, since the NaN check happens before both to_csv calls."""
    rebuild_result, enriched_result = _make_inputs()
    rebuild_result['w'] = rebuild_result['w'].drop(
        pd.Timestamp('2026-01-01 01:00')
    )
    with pytest.raises(ValueError):
        live_staging.stage_live_training_tables(
            rebuild_result, enriched_result, str(tmp_path)
        )
    assert not (tmp_path / 'ch07_training_table_enriched.csv').exists()
    assert not (tmp_path / 'ch05_features.csv').exists()


# ---------------------------------------------------------------------
# ch05_features.csv: full rebuild.py close series, NOT the enriched
# (post-dropna) subset
# ---------------------------------------------------------------------

def test_features_csv_uses_full_close_not_enriched_subset(tmp_path):
    """rebuild_result['close'] spans 5 bars (00:00-04:00); enriched_events
    only has 3 rows (00:00-02:00, warmup/NaN rows already dropped by
    features.py). ch05_features.csv must carry all 5 close values --
    Ch11 only ever reads feats['close'] from it (module docstring) and
    the static baseline's ch05_features.csv is bar-indexed, not event-
    indexed."""
    rebuild_result, enriched_result = _make_inputs()
    out = live_staging.stage_live_training_tables(
        rebuild_result, enriched_result, str(tmp_path)
    )
    feats = pd.read_csv(out['features_csv_path'], index_col=0, parse_dates=True)

    assert len(feats) == 5
    assert list(feats.columns) == ['close']
    assert feats['close'].tolist() == [100.0, 101.0, 102.0, 103.0, 104.0]


def test_features_csv_single_column_only(tmp_path):
    """Module docstring: 'Ch11 only ever reads feats['close']' -- no
    other columns should leak in even though rebuild_result carries
    more keys than just 'close'."""
    rebuild_result, enriched_result = _make_inputs()
    out = live_staging.stage_live_training_tables(
        rebuild_result, enriched_result, str(tmp_path)
    )
    feats = pd.read_csv(out['features_csv_path'], index_col=0)
    assert feats.shape[1] == 1


# ---------------------------------------------------------------------
# feature_cols ordering preserved from feature_table.columns
# ---------------------------------------------------------------------

def test_feature_cols_order_matches_feature_table_column_order(tmp_path):
    """feature_cols = list(feature_table.columns) -- order must be
    preserved (not sorted/reordered), since it feeds directly into the
    training_table column selection."""
    rebuild_result, enriched_result = _make_inputs()
    # reverse the feature_table's column order relative to enriched_events'
    enriched_result['feature_table'] = enriched_result['feature_table'][
        ['featB', 'featA']
    ]
    out = live_staging.stage_live_training_tables(
        rebuild_result, enriched_result, str(tmp_path)
    )
    assert out['feature_cols'] == ['featB', 'featA']

    df = pd.read_csv(out['enriched_csv_path'], index_col=0)
    assert list(df.columns) == ['t1', 'bin', 'featB', 'featA', 'w']


# =============================================================================
# TDD VERIFICATION -- pytest results, real-machine-confirmed 2026-08-16
# (mlfinlab env: Python 3.10.20, pandas 1.5.3, numpy 1.23.5, pytest 9.0.3)
# =============================================================================
# Two-pass run (per project convention):
#
# PASS 1 -- from repo root (pytest pipeline\orchestration\test_live_staging.py -v):
#   test_writes_exact_filenames_ch11_hard_loads PASSED
#   test_return_dict_n_events_and_feature_cols PASSED
#   test_creates_out_dir_if_missing PASSED
#   test_training_table_columns_are_exactly_t1_bin_features_w PASSED
#   test_training_table_hand_traced_values PASSED
#   test_raises_when_w_missing_a_label_in_enriched_index PASSED
#   test_no_files_left_behind_shape_after_raise PASSED
#   test_features_csv_uses_full_close_not_enriched_subset PASSED
#   test_features_csv_single_column_only PASSED
#   test_feature_cols_order_matches_feature_table_column_order PASSED
#   10 passed in 1.59s
#
# PASS 2 -- from pipeline\orchestration\ (pytest test_live_staging.py -v):
#   Same 10 tests, all PASSED, 10 passed in 0.91s
#
# No bugs found in live_staging.py during real-machine confirmation --
# sandbox (Python 3.12.3/pandas 2.x, dry-run only) and real-machine
# (mlfinlab pinned versions) behavior matched exactly.
# =============================================================================
