# Live pipeline run -- 2026-08-14

Frozen snapshot of one real end-to-end run of `pipeline/run_pipeline_live.py`
against a live 720h (30-day) Binance.US BTCUSDT pull. This is NOT live/current
data -- it is a point-in-time example proving the live pipeline (ingestion ->
rebuild -> features -> Ch11's real trial construction -> PBO/DSR -> report)
runs end-to-end on real, independently-pulled data. See pipeline/README.md
for how to run a fresh live pipeline yourself.

Real numbers from this specific run:
  240 bars, 48 triple-barrier events (threshold=$308,241.55)
  48/48 events survived Ch19+Ch05 feature enrichment
  fracdiff d=0.1 (contrast with the 2026-08-13 session's d=0 on a
    different 30-day window -- one data point toward the still-open
    fracdiff d=0 investigation, not a resolution of it)
  PBO 78.57%, DSR 0.3706 -- directionally consistent with this project's
    established static-data convergent-null finding (Ch11-15)
  Report flagged its own SAMPLE SIZE WARNING (146 obs, 20 trials, below
    the 150-obs reliability threshold) -- correctly, this run's numbers
    should NOT be read as reliable evidence of edge either way.

Files:
  ch07_training_table_enriched.csv  staged live training table (12 real
    features + fracdiff, t1/bin/w -- trgt/ret deliberately excluded, see
    live_staging.py's own LOAD-BEARING note on why)
  ch05_features.csv                 staged close-price series
  ch11_trial_sharpes.png            Ch11's real trial-comparison plot,
    generated from this live run (NOT ch11/'s own committed static-data
    plot -- that one is untouched, see stages.py's run_live_trials()
    docstring on why HERE is monkeypatched)
  live_report.txt                   the plain-English report this run
    produced
