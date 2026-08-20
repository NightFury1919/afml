# Live pipeline run -- 2026-08-19

Frozen snapshot of one real end-to-end run of `pipeline/run_pipeline_live.py` against a live 720h (30-day) Binance.US BTCUSDT pull. This is NOT live/current data -- it is a point-in-time example proving the live pipeline (ingestion -> rebuild -> features -> Ch11's real trial construction -> PBO/DSR -> report) runs end-to-end on real, independently-pulled data. See pipeline/README.md for how to run a fresh live pipeline yourself.

Real numbers from this specific run:
  99878 raw trades, 238 bars, 48 triple-barrier events
  47/48 events survived feature enrichment (fracdiff d=0.30000000000000004)
  S=12, n_trials=20, T_effective=66.3266 (T_raw=156 x tw_mean=0.4252)
  best trial: C0.1_s0.05, sharpe=-0.0049
  PBO 47.84%, DSR 0.2136 -- lifecycle stage: EMBARGO
  OTR: phi_hat=0.9673, stationary=True, half-life=20.9 bars
  Ch15: P[fail]=0.4654, realized precision=54.17% on 48 real bets, annualized to ~615.6 bets/year

Files:
  ch07_training_table_enriched.csv  staged live training table (12 real features + fracdiff, t1/bin/w -- trgt/ret deliberately excluded)
