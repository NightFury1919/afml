"""
pipeline/orchestration/live_staging.py

Phase 3: writes rebuild.py + features.py's live output to CSVs in the
EXACT schema/filenames ch11's chapter_11_backtest_dangers.py expects
(ch07_training_table_enriched.csv, ch05_features.csv), so
part_c_build_trials() -- which hard-loads those two filenames from its
own module-level INPUT constant, with no parameters -- can run against
live data unmodified. See stages.py's run_live_trials() for how INPUT is
pointed at this module's output directory via monkeypatching, rather than
editing chapter_11_backtest_dangers.py itself.

*** LOAD-BEARING (2026-08-14): trgt/ret are DROPPED, not carried through
***
features.build_enriched_events()'s 'enriched_events' still carries
rebuild.py's 't1', 'trgt', 'ret', 'bin' columns alongside the 12 real
features (Ch19's 11 + fracdiff). Ch11's part_c_build_trials() derives
feature_cols as everything NOT named 'bin'/'w'/'t1' -- it does NOT know
about 'trgt'/'ret' by name. Staged naively, both would be silently fed
into the SVC as bogus features. The static ch07_training_table_enriched
.csv does not carry them (confirmed by inspecting chapter_11_backtest_
dangers.py's own working feature_cols count against it) -- this module
replicates that same shape deliberately, not by accident.

*** LOAD-BEARING (2026-08-14): 'w' is a separate Series in rebuild.py's
output, re-reindexed here to the POST-enrichment index ***
rebuild.py's 'w' (return-attribution sample weight) is indexed to its
own pre-enrichment event count. features.build_enriched_events() drops
warmup rows (dropna on feature_cols) when joining the Ch19/fracdiff
table, so the final enriched_events index is a SUBSET of rebuild.py's
events index. Staging 'w' without reindexing to the post-dropna index
would silently misalign sample weights with the events they're supposed
to weight -- w must be reindexed here, not assumed to already match.
"""
import os

import pandas as pd


def stage_live_training_tables(rebuild_result, enriched_result, out_dir):
    """Write ch07_training_table_enriched.csv and ch05_features.csv into
    out_dir, matching the exact filenames/schema chapter_11_backtest_
    dangers.py's part_c_build_trials() hard-loads.

    Parameters
    ----------
    rebuild_result : dict, rebuild.py's build_bars_and_labels() output
        (needs 'events', 'w', 'close')
    enriched_result : dict, features.py's build_enriched_events() output
        (needs 'enriched_events', 'feature_table')
    out_dir : str, live staging directory (created if missing)

    Returns
    -------
    dict with keys 'enriched_csv_path', 'features_csv_path', 'n_events',
    'feature_cols' -- for the caller to log/inspect what was staged.
    """
    os.makedirs(out_dir, exist_ok=True)

    enriched = enriched_result['enriched_events']
    feature_cols = list(enriched_result['feature_table'].columns)

    w_aligned = rebuild_result['w'].reindex(enriched.index)
    if w_aligned.isna().any():
        raise ValueError(
            'Sample weight w has NaN after reindexing to the post-'
            'enrichment event index -- an enriched event has no matching '
            'rebuild.py event, which should be impossible since '
            'build_enriched_events() only DROPS rows from rebuild.py\'s '
            'events, never adds new ones. Investigate before staging.'
        )

    # ONLY t1, bin, w, and the real feature columns -- trgt/ret
    # deliberately excluded (see module LOAD-BEARING note above)
    training_table = enriched[['t1', 'bin'] + feature_cols].copy()
    training_table['w'] = w_aligned

    enriched_csv_path = os.path.join(out_dir, 'ch07_training_table_enriched.csv')
    training_table.to_csv(enriched_csv_path)

    # Ch11 only ever reads feats['close'] from ch05_features.csv
    features_csv_path = os.path.join(out_dir, 'ch05_features.csv')
    pd.DataFrame({'close': rebuild_result['close']}).to_csv(features_csv_path)

    return {
        'enriched_csv_path': enriched_csv_path,
        'features_csv_path': features_csv_path,
        'n_events': len(training_table),
        'feature_cols': feature_cols,
    }