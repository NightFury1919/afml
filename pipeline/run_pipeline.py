"""
pipeline/run_pipeline.py

Driver script: chains real AFML chapter modules end-to-end on the existing
March 2026 BTC/TUSD real dataset. Phase 1b reuses Ch11's own real,
established 20-trial bar-level trial-construction pipeline directly (see
orchestration/stages.py's module docstring for why Phase 1a's smaller,
event-level version was replaced), then computes PBO/DSR overfitting
diagnostics and a plain-English evidence-based assessment report.

Phase 1 scope (per project handoff, 2026-08-12): proves the orchestration +
report layer against EXISTING static real data. Phase 2 will replace the
static-artifact load with a live Binance pull feeding the SAME downstream
bar/feature/label pipeline.

*** LOAD-BEARING (2026-08-17): loads and aligns tw for the DSR fix ***
evaluate_overfitting() now REQUIRES a uniqueness-weighted tw (see
stages.py's own LOAD-BEARING note on this). This script owns sourcing it
for the static pipeline: ch04_weights.csv's real 'tw' column, reindexed to
ch07_training_table_enriched.csv's real event population (the same
population Ch11's part_c_build_trials() trains the 20-trial grid on) --
mirrors live_staging.py's existing reindex-and-fail-loud pattern for 'w'.

Usage
-----
    conda activate mlfinlab
    cd C:\\ws\\AFML
    python pipeline\\run_pipeline.py
"""
import os
import sys

import pandas as pd

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.abspath(os.path.join(HERE, '..'))
INPUT_DATA = os.path.join(ROOT, 'input_data')

sys.path.insert(0, os.path.join(HERE, 'orchestration'))

from stages import (                       # noqa: E402
    load_ch11_driver, run_real_trials, evaluate_overfitting,
    latest_bet_signal,
)
from report import build_report            # noqa: E402


def load_aligned_tw(input_data_dir):
    """Real Ch04 average uniqueness, cached in ch04_weights.csv, reindexed
    to the enriched training table's real event population. Raises loudly
    on NaN after reindexing (same convention live_staging.py already uses
    for 'w') -- an unaligned tw would silently corrupt the DSR fix."""
    weights = pd.read_csv(
        os.path.join(input_data_dir, 'ch04_weights.csv'),
        index_col=0, parse_dates=True,
    )
    enriched_index = pd.read_csv(
        os.path.join(input_data_dir, 'ch07_training_table_enriched.csv'),
        index_col=0, parse_dates=True,
    ).index
    tw = weights['tw'].reindex(enriched_index)
    if tw.isna().any():
        raise ValueError(
            "tw has NaN after reindexing to the enriched training table's "
            "event index -- ch04_weights.csv may be stale relative to "
            "ch07_training_table_enriched.csv. Investigate before running "
            "the pipeline."
        )
    return tw


def main():
    ch11 = load_ch11_driver()
    M, meta = run_real_trials(ch11)
    tw = load_aligned_tw(INPUT_DATA)

    eval_result = evaluate_overfitting(M, meta, ch11, S=8, tw=tw)
    signal = latest_bet_signal(eval_result['best_trial'], meta, ch11, INPUT_DATA)

    report = build_report(eval_result, signal, asset_label='BTC/TUSD')
    print(report)

    out_path = os.path.join(HERE, 'latest_report.txt')
    with open(out_path, 'w') as f:
        f.write(report)
    print(f"\n[report written to {out_path}]")


if __name__ == '__main__':
    main()
