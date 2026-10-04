"""Effective breadth of the 48-ETF universe, and what it implies for detectability.

Run from the portfolio folder in the mlfinlab env:
    python report_effective_breadth.py
    python report_effective_breadth.py prices_daily_asof_2026-10-02.csv

Only prices are used. No features, labels or model results are looked at.

Link to detectability (my arithmetic, an approximation, not a pipeline result):
    fundamental law  IR = TC * IC * sqrt(independent bets per year)
    bets per year    = (252 / horizon) * ENB          (ENB measured on returns vs the median)
    detectable IR    = z / sqrt(years), with z = 2.9 for 50% power and 3.8 for 80% power
                       (the thresholds from the project's crypto power study)
TC (transfer coefficient) is how much of the signal survives portfolio constraints;
a long-only portfolio keeps less of it. TC = 0.5 below is an illustration, not a measurement.
"""
import os
import sys

import numpy as np
import pandas as pd

from effective_breadth import enb_from_returns, nonoverlapping_sum, relative_to_median, rolling_enb
from panel_data import drop_hedge, load_prices

HORIZON = 21
YEARS_FULL_BREADTH = 17.8   # 2008-12-18 to 2026-10-02: all 48 ETFs and a 252-day warm-up
Z_50, Z_80 = 2.9, 3.8


def main(path):
    px = drop_hedge(load_prices(path))
    full = np.log(px).diff().dropna(how="any")
    print(f"Full-universe return sample: {full.index.min().date()} -> {full.index.max().date()}, "
          f"{len(full)} rows, {full.shape[1]} ETFs")

    corr = np.corrcoef(full.to_numpy(), rowvar=False)
    top_share = np.linalg.eigvalsh(corr)[-1] / full.shape[1]
    raw_enb = enb_from_returns(full)
    rel = relative_to_median(full)
    rel_enb = enb_from_returns(rel)
    print(f"\nA) raw daily returns:                 ENB = {raw_enb:5.1f}  (largest factor = {top_share:.0%} of variance)")
    print(f"B) daily returns vs median (PRIMARY): ENB = {rel_enb:5.1f}")

    m = nonoverlapping_sum(rel, HORIZON)
    noise = np.mean([enb_from_returns(pd.DataFrame(np.random.default_rng(s).normal(size=(len(m), 48))),
                                      min_obs=100) for s in range(50)])
    print(f"C) {HORIZON}-day non-overlapping vs median: ENB = {enb_from_returns(m, min_obs=100):5.1f}  "
          f"({len(m)} blocks; pure noise at this size would show {noise:.1f})")

    roll = rolling_enb(rel, window=756, step=63)
    print(f"D) rolling 3-year windows:            min {roll.min():.1f}, median {roll.median():.1f}, max {roll.max():.1f}")

    ir50 = Z_50 / np.sqrt(YEARS_FULL_BREADTH)
    ir80 = Z_80 / np.sqrt(YEARS_FULL_BREADTH)
    bets = (252 / HORIZON) * rel_enb
    print(f"\nDetectable strategy IR over {YEARS_FULL_BREADTH} years: {ir50:.2f} (50% power), {ir80:.2f} (80% power)")
    print(f"Independent bets per year at {HORIZON}-day horizon: {bets:.0f}")
    for tc in (1.0, 0.5):
        print(f"  TC={tc}: IC needed = {ir50 / (tc * np.sqrt(bets)):.3f} (50%), {ir80 / (tc * np.sqrt(bets)):.3f} (80%)")
    for ic in (0.02, 0.03, 0.05):
        for tc in (1.0, 0.5):
            ir = tc * ic * np.sqrt(bets)
            print(f"  true IC={ic:.2f}, TC={tc}: IR = {ir:.2f}, years to detect at 50% power = {(Z_50 / ir) ** 2:.0f}")


if __name__ == "__main__":
    default = os.path.join(os.path.dirname(os.path.abspath(__file__)), "prices_daily_asof_2026-10-02.csv")
    main(sys.argv[1] if len(sys.argv) > 1 else default)

# ---------------------------------------------------------------------------
# REAL-DATA RUN (mlfinlab env, 2026-10-04, prices_daily_asof_2026-10-02.csv)
# $ python report_effective_breadth.py
#
# Full-universe return sample: 2007-12-20 -> 2026-10-02, 4725 rows, 48 ETFs
#
# A) raw daily returns:                 ENB =   9.0  (largest factor = 51% of variance)
# B) daily returns vs median (PRIMARY): ENB =  16.5
# C) 21-day non-overlapping vs median: ENB =  14.9  (225 blocks; pure noise at this size would show 43.2)
# D) rolling 3-year windows:            min 13.3, median 15.2, max 17.5
#
# Detectable strategy IR over 17.8 years: 0.69 (50% power), 0.90 (80% power)
# Independent bets per year at 21-day horizon: 198
#   TC=1.0: IC needed = 0.049 (50%), 0.064 (80%)
#   TC=0.5: IC needed = 0.098 (50%), 0.128 (80%)
#   true IC=0.02, TC=1.0: IR = 0.28, years to detect at 50% power = 106
#   true IC=0.02, TC=0.5: IR = 0.14, years to detect at 50% power = 424
#   true IC=0.03, TC=1.0: IR = 0.42, years to detect at 50% power = 47
#   true IC=0.03, TC=0.5: IR = 0.21, years to detect at 50% power = 188
#   true IC=0.05, TC=1.0: IR = 0.70, years to detect at 50% power = 17
#   true IC=0.05, TC=0.5: IR = 0.35, years to detect at 50% power = 68
# ---------------------------------------------------------------------------
