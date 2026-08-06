"""
Chapter 18, Part C -- Portfolio Concentration (Section 18.8.3)
=================================================================

Wires portfolio_concentration.py against Chapter 16's REAL 6-commodity
covariance matrix and HRP/IVP allocation vectors. This was deferred when
Ch18 was first scoped ("conceptual only, no usable data yet") -- Ch16
later built exactly the prerequisite this needed (a real covariance matrix
+ real allocation vectors), so per the project's standing "revisit once a
prerequisite exists" rule, it's implemented now.

REAL BUG CAUGHT DURING WIRING (not hypothetical -- verified with a
synthetic mismatched-order reproduction before touching real data):
chapter_16_hrp.py's part_b_real_data() returns hrp/ivp sorted
ALPHABETICALLY by asset name (`.sort_index()`), but cov = ret_df.cov()
keeps the COMMODITIES dict's insertion order (gold, crude_oil, corn,
live_hogs, tbonds, gbp -- not alphabetical). portfolio_concentration.py's
functions are plain-numpy / position-based, not label-aware, so passing
hrp.values or ivp.values directly against cov.values would silently pair
each weight with the WRONG asset's variance/covariance row. Both weight
vectors are explicitly reindexed to cov's column order below before
being passed in -- do not remove this reindex step.

Run this AFTER chapter_16_hrp.py's Part B has been run at least once
(needs the real commodity txt files under ch16/input_data/, which only
exist on the real machine, not in this sandbox) -- it re-runs Part B
itself internally to get real hrp/ivp/cov, so no separate artifact file
is required.
"""
import os
import sys

import numpy as np
import pandas as pd

HERE = os.path.dirname(os.path.abspath(__file__))
AFML_ROOT = os.path.abspath(os.path.join(HERE, '..'))

# Reuse Ch16's real driver code directly, rather than recomputing the
# covariance matrix / HRP / IVP independently -- avoids any risk of this
# script's version of that logic drifting from Ch16's.
sys.path.insert(0, os.path.join(AFML_ROOT, 'ch16'))
from chapter_16_hrp import part_b_real_data  # noqa: E402

sys.path.insert(0, os.path.join(HERE, 'entropy_features'))
from portfolio_concentration import compute_portfolio_concentration  # noqa: E402


def part_c_portfolio_concentration():
    print()
    print("=" * 78)
    print("PART C -- Portfolio Concentration (18.8.3), Ch16's real HRP/IVP weights")
    print("=" * 78)

    hrp, ivp, corr, ret_df = part_b_real_data()
    cov = ret_df.cov()

    # --- Alignment fix: see module docstring for why this is required ---
    assets = cov.index.tolist()
    hrp_aligned = hrp.reindex(assets)
    ivp_aligned = ivp.reindex(assets)
    assert not hrp_aligned.isna().any(), (
        "HRP weights missing for one or more cov assets after reindex -- "
        "asset name mismatch between Ch16's hrp Series and cov's columns"
    )
    assert not ivp_aligned.isna().any(), (
        "IVP weights missing for one or more cov assets after reindex -- "
        "asset name mismatch between Ch16's ivp Series and cov's columns"
    )

    H_hrp, theta_hrp = compute_portfolio_concentration(cov.values, hrp_aligned.values)
    H_ivp, theta_ivp = compute_portfolio_concentration(cov.values, ivp_aligned.values)

    theta_hrp_series = pd.Series(theta_hrp, index=assets).sort_values(ascending=False)
    theta_ivp_series = pd.Series(theta_ivp, index=assets).sort_values(ascending=False)

    print(f"\nHRP portfolio concentration H = {H_hrp:.4f}")
    print("Risk contribution per principal component (theta), HRP:")
    print(theta_hrp_series.to_string())

    print(f"\nIVP portfolio concentration H = {H_ivp:.4f}")
    print("Risk contribution per principal component (theta), IVP:")
    print(theta_ivp_series.to_string())

    print(
        f"\nH is bounded in [0, 1 - 1/N] for N={len(assets)} assets, "
        f"so [0, {1 - 1 / len(assets):.4f}] here. "
        "0 = risk spread perfectly evenly across principal components "
        "(maximally diversified); the upper bound = all risk concentrated "
        "in a single principal component."
    )
    if H_hrp < H_ivp:
        print(
            f"\nHRP is LESS concentrated than IVP ({H_hrp:.4f} < {H_ivp:.4f}) -- "
            "HRP's correlation-aware allocation spreads risk across more "
            "principal components than IVP's variance-only weighting."
        )
    elif H_hrp > H_ivp:
        print(
            f"\nHRP is MORE concentrated than IVP ({H_hrp:.4f} > {H_ivp:.4f})."
        )
    else:
        print(f"\nHRP and IVP have equal concentration ({H_hrp:.4f}).")

    return H_hrp, theta_hrp_series, H_ivp, theta_ivp_series


if __name__ == '__main__':
    part_c_portfolio_concentration()
