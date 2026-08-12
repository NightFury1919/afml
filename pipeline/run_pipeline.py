"""
pipeline/run_pipeline.py

Driver script: chains real AFML chapter modules end-to-end on the existing
March 2026 BTC/TUSD real dataset, from the already-built enriched feature
table through multi-trial purged CV, PBO/DSR overfitting diagnostics, and a
plain-English evidence-based assessment report.

Phase 1 scope (per project handoff, 2026-08-12): proves the orchestration +
report layer against EXISTING static real data
(ch07_training_table_enriched.csv, ch03_events.csv, both already
real-machine confirmed by Ch04/05/19). Phase 2 will replace the static-
artifact load with a live Binance pull feeding the SAME downstream bar/
feature/label pipeline -- everything in orchestration/stages.py past
load_enriched_table() is already asset- and data-source-agnostic.

Usage
-----
    conda activate mlfinlab
    cd C:\\ws\\AFML
    python pipeline\\run_pipeline.py
"""
import os
import sys

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.abspath(os.path.join(HERE, '..'))
INPUT_DATA = os.path.join(ROOT, 'input_data')

sys.path.insert(0, os.path.join(HERE, 'orchestration'))

from stages import (                       # noqa: E402
    load_enriched_table, default_trials, assemble_pnl_matrix,
    evaluate_overfitting, latest_bet_signal,
)
from report import build_report            # noqa: E402


def main():
    X, y, w, t1, ret = load_enriched_table(INPUT_DATA)
    trials = default_trials()

    M, trial_probs = assemble_pnl_matrix(X, y, w, t1, ret, trials)
    eval_result = evaluate_overfitting(M, S=8)
    signal = latest_bet_signal(eval_result['best_trial'], trial_probs, t1)

    report = build_report(eval_result, signal, asset_label='BTC/TUSD')
    print(report)

    out_path = os.path.join(HERE, 'latest_report.txt')
    with open(out_path, 'w') as f:
        f.write(report)
    print(f"\n[report written to {out_path}]")


if __name__ == '__main__':
    main()
