# Live pipeline run -- 2026-08-21

Frozen snapshot of one real end-to-end run of `pipeline/run_pipeline_live.py` against a live 720h (30-day) Binance.US BTCUSDT pull. This is NOT live/current data -- it is a point-in-time example proving the live pipeline (ingestion -> rebuild -> features -> Ch11's real trial construction -> PBO/DSR -> report) runs end-to-end on real, independently-pulled data. See pipeline/README.md for how to run a fresh live pipeline yourself.

Real numbers from this specific run:
  114116 raw trades, 855 bars, 182 triple-barrier events
  182/182 events survived feature enrichment (fracdiff d=0.6000000000000001)
  S=12, n_trials=20, T_effective=137.2352 (T_raw=696 x tw_mean=0.1972)
  best trial: C1_s0.05, sharpe=0.0637
  PBO 3.57%, DSR 0.5784 -- lifecycle stage: PAPER_TRADING
  OTR: phi_hat=1.0141, stationary=False, half-life=nan bars
  Ch15: P[fail]=0.2457, realized precision=71.98% on 182 real bets, annualized to ~2334.1 bets/year

Files:
  ch07_training_table_enriched.csv  staged live training table (12 real features + fracdiff, t1/bin/w -- trgt/ret deliberately excluded)
