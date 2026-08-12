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
    load_ch11_driver, run_real_trials, evaluate_overfitting,
    latest_bet_signal,
)
from report import build_report            # noqa: E402


def main():
    ch11 = load_ch11_driver()
    M, meta = run_real_trials(ch11)

    eval_result = evaluate_overfitting(M, meta, ch11, S=8)
    signal = latest_bet_signal(eval_result['best_trial'], meta, ch11, INPUT_DATA)

    report = build_report(eval_result, signal, asset_label='BTC/TUSD')
    print(report)

    out_path = os.path.join(HERE, 'latest_report.txt')
    with open(out_path, 'w') as f:
        f.write(report)
    print(f"\n[report written to {out_path}]")


if __name__ == '__main__':
    main()
