import numpy as np
import pandas as pd
import sys
import os

# Add project root (C:\ws\AFML) to path so we can import ch04 as a package
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '..')))

from ch04.sample_weights.indicator_matrix      import get_ind_matrix
from ch04.sample_weights.avg_uniqueness_matrix import get_avg_uniqueness
from ch04.sample_weights.sequential_bootstrap  import seq_bootstrap

# Example of Sequential Bootstrap — AFML Chapter 4, Snippet 4.6, page 65-66
#
# A small, fully worked demonstration comparing standard bootstrap (random
# sampling with replacement, equal probability every draw) against sequential
# bootstrap (probability-weighted toward unique observations) on the exact
# 3-observation example used throughout Section 4.5.3.
#
# --- The setup ---
# Three feature observations:
#   obs0: outcome determined using bars 0 through 2  (t0=0, t1=2)
#   obs1: outcome determined using bars 2 through 3  (t0=2, t1=3)
#   obs2: outcome determined using bars 4 through 5  (t0=4, t1=5)
#
# obs0 and obs1 overlap at bar 2. obs2 never overlaps with anything.
#
# --- What this script demonstrates ---
# Run BOTH bootstrap methods on the same indicator matrix, then compare the
# resulting average uniqueness of each sample. Sequential bootstrap should,
# on average, produce a HIGHER uniqueness sample than standard bootstrap —
# this is the entire point of Sections 4.5.1-4.5.4. A single run is just one
# data point (the real comparison needs the Monte Carlo experiment from
# Snippets 4.7-4.9), but this demo shows the mechanics end-to-end.


def main():
    # t1: index = event start (t0), values = event end (t1)
    # This is the exact example from Section 4.5.3.
    t1 = pd.Series([2, 3, 5], index=[0, 2, 4])

    # Index of all bars spanned by any observation
    bar_ix = range(t1.max() + 1)

    # Build the indicator matrix (Snippet 4.3) — shows which bars each
    # observation touches
    ind_m = get_ind_matrix(bar_ix, t1)
    print("Indicator matrix:")
    print(ind_m)
    print()

    # -------------------------------------------------------------------
    # Standard bootstrap: draw with replacement, EQUAL probability always
    # -------------------------------------------------------------------
    phi_standard = np.random.choice(ind_m.columns, size=ind_m.shape[1])
    print("Standard bootstrap sample:", list(phi_standard))
    print("Standard uniqueness:", get_avg_uniqueness(ind_m[phi_standard]).mean())
    print()

    # -------------------------------------------------------------------
    # Sequential bootstrap: draw with replacement, probability WEIGHTED
    # toward whichever candidate is most unique given the current sample
    # -------------------------------------------------------------------
    phi_sequential = seq_bootstrap(ind_m)
    print("Sequential bootstrap sample:", phi_sequential)
    print("Sequential uniqueness:", get_avg_uniqueness(ind_m[phi_sequential]).mean())


if __name__ == '__main__':
    main()
