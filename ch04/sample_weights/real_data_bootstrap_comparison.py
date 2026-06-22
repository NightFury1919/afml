import numpy as np
import pandas as pd
import sys
import os

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '..')))
from ch04.sample_weights.indicator_matrix      import get_ind_matrix
from ch04.sample_weights.avg_uniqueness_matrix import get_avg_uniqueness
from ch04.sample_weights.sequential_bootstrap  import seq_bootstrap

# Real-Data Bootstrap Comparison — companion to the Monte Carlo experiment
# in monte_carlo.py (Snippets 4.7-4.9).
#
# --- Why this file exists ---
# main_mc() (in monte_carlo.py) answers a GENERAL question: "does sequential
# bootstrap beat standard bootstrap on average, across many different random
# overlap scenarios?" It uses synthetic, randomly generated observations
# precisely because it needs to vary the overlap structure across thousands
# of trials to get a statistically stable answer.
#
# This file answers a different, more PRACTICAL question that matters more
# to a student building a real strategy: "given the ACTUAL overlap structure
# in MY OWN labeled events, how much does sequential bootstrap help me?"
# Instead of generating new synthetic data every trial, we build ONE
# indicator matrix from real triple-barrier events and bootstrap repeatedly
# from that same real structure.
#
# --- Why we subsample down to a small number of events ---
# seq_bootstrap is inherently expensive: for every single observation it
# draws, it must evaluate EVERY candidate's hypothetical uniqueness before
# choosing one. That's roughly O(n^2) work per trial, and real-world event
# sets can have hundreds or thousands of rows — running this on the full
# dataset could take minutes or hours. For a teaching demonstration, we cap
# the event count low enough (12-20) that the whole comparison finishes in
# a few seconds, while still being built from 100% REAL data rather than a
# synthetic stand-in.


def compare_bootstrap_on_real_events(close, events, max_events=12, n_trials=15, seed=None):
    # Run standard vs sequential bootstrap repeatedly on a SUBSAMPLE of a
    # real events DataFrame, and return both sets of uniqueness scores for
    # plotting/comparison.
    #
    # --- Inputs ---
    # close      : pd.Series — the ACTUAL price bar series (e.g. your dollar
    #              bar closes) that events['t1'] was computed against. We
    #              build the indicator matrix using THIS index, not a
    #              guessed date range — this is critical for performance:
    #              using the real bar index keeps the matrix exactly as
    #              large as it needs to be (one row per actual bar), instead
    #              of accidentally creating tens of thousands of empty rows
    #              by guessing a frequency like 'minute' over a long date span.
    # events     : pd.DataFrame — real triple barrier events, must have a
    #              't1' column (output of get_events() from Chapter 3)
    # max_events : int — how many events to subsample down to (default 12).
    #              Keep this small (≤ ~20) — seq_bootstrap's cost grows
    #              roughly quadratically with this number.
    # n_trials   : int — how many bootstrap comparisons to run (default 15)
    # seed       : int or None — random seed for reproducibility
    #
    # --- Output ---
    # dict with keys:
    #   'std_vals'  : list of standard bootstrap uniqueness scores, one per trial
    #   'seq_vals'  : list of sequential bootstrap uniqueness scores, one per trial
    #   'n_events'  : actual number of events used (after subsampling)
    #   'n_bars'    : number of bars in the indicator matrix (for sanity-checking
    #                 that this stayed small enough to run quickly)
    #   'ind_m'     : the indicator matrix built from the subsampled events

    if seed is not None:
        np.random.seed(seed)

    events = events.dropna(subset=['t1'])

    # Subsample down to max_events for tractable runtime.
    #
    # IMPORTANT: we don't subsample uniformly at random across the entire
    # date range. A uniform random subsample can happen to pick events
    # scattered far apart in time, which forces bar_ix to span that entire
    # range (potentially thousands of bars) even though we only kept a
    # handful of events — that defeats the purpose of subsampling for speed.
    #
    # Instead we pick one random CONTIGUOUS block of max_events consecutive
    # events (after sorting chronologically). This keeps the resulting
    # bar_ix span small and predictable, while still giving a genuine
    # cross-section of the student's real, overlapping event structure.
    if len(events) > max_events:
        events_sorted = events.sort_index()
        max_start = len(events_sorted) - max_events
        start_idx = np.random.randint(0, max_start + 1)
        events_sub = events_sorted.iloc[start_idx:start_idx + max_events]
    else:
        events_sub = events.sort_index()

    t1_sub = events_sub['t1']

    # Build bar_ix from the REAL underlying bar series, restricted to just
    # the span this subsample of events actually covers. This guarantees
    # bar_ix has exactly as many rows as real bars exist in that window —
    # no guessed frequency, no risk of an accidentally huge matrix.
    bar_ix = close.index[
        (close.index >= t1_sub.index.min()) & (close.index <= t1_sub.max())
    ]

    ind_m = get_ind_matrix(bar_ix, t1_sub)

    std_vals, seq_vals = [], []
    for _ in range(n_trials):
        # Standard bootstrap: equal probability, random draw with replacement
        phi_std = np.random.choice(ind_m.columns, size=ind_m.shape[1])
        std_vals.append(get_avg_uniqueness(ind_m[phi_std]).mean())

        # Sequential bootstrap: probability weighted toward uniqueness
        phi_seq = seq_bootstrap(ind_m)
        seq_vals.append(get_avg_uniqueness(ind_m[phi_seq]).mean())

    return {
        'std_vals': std_vals,
        'seq_vals': seq_vals,
        'n_events': len(events_sub),
        'n_bars':   len(bar_ix),
        'ind_m':    ind_m,
    }
