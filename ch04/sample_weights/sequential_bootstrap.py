import numpy as np
import pandas as pd
import sys
import os

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '..')))
from .avg_uniqueness_matrix import get_avg_uniqueness

# Return Sample from Sequential Bootstrap — AFML Chapter 4, Snippet 4.5, page 65
#
# The actual fix to the problem described in Section 4.5.1: instead of
# drawing observations purely at random (standard bootstrap), sequentially
# draw observations with probability WEIGHTED toward whichever candidate
# would be most unique given what's already been drawn.
#
# --- The core idea, traced through the book's own example (Section 4.5.3) ---
# Three observations: obs0 spans bars 0-2, obs1 spans bars 2-3, obs2 spans
# bars 4-5 (obs1 and obs0 overlap at bar 2; obs2 never overlaps with anything).
#
# Draw 1: nothing picked yet, so all three get equal probability (1/3 each).
#         Suppose obs1 gets drawn. phi = [1].
#
# Before draw 2: for EACH candidate i, ask "if phi were [1, i], how unique
#         would i be (the just-added observation)?"
#   i=0: phi_hypothetical=[1,0] → uniqueness of column 0 (the last one added)
#   i=1: phi_hypothetical=[1,1] → uniqueness of drawing obs1 AGAIN (very low,
#        since it's now fully redundant with itself)
#   i=2: phi_hypothetical=[1,2] → uniqueness of column 2 (no overlap with 1 → high)
#
# Normalize these three uniqueness scores into a probability distribution.
# The observation already drawn (obs1) ends up with the LOWEST probability of
# being drawn again. The observation with NO overlap to what's drawn so far
# (obs2) gets the HIGHEST probability. This is the key mechanism: redundant
# observations get suppressed, unique ones get favored, at every single draw.
#
# --- Why this beats standard bootstrap ---
# Standard bootstrap draws with EQUAL probability every time, regardless of
# what's already in the sample. It happily draws the same overlapping cluster
# of observations over and over. Sequential bootstrap actively steers away
# from redundancy, producing samples with higher average uniqueness overall
# (Section 4.5.4's Monte Carlo experiment shows median uniqueness 0.7 vs 0.6).


def seq_bootstrap(ind_m, s_length=None):
    # Generate a sample via sequential bootstrap — AFML Snippet 4.5
    #
    # --- Inputs ---
    # ind_m    : pd.DataFrame — output of get_ind_matrix() (Snippet 4.3)
    #            shape (num_bars, num_events)
    # s_length : int or None — number of observations to draw
    #            Defaults to the total number of events (same draw count
    #            as a standard bootstrap would use)
    #
    # --- Output ---
    # list of column labels (event identifiers) — the sequentially
    # bootstrapped sample, drawn WITH replacement (can contain duplicates,
    # but far fewer than standard bootstrap would produce)

    if s_length is None:
        s_length = ind_m.shape[1]

    phi = []  # running list of drawn observations

    while len(phi) < s_length:
        # For every possible next candidate, score how unique it WOULD be
        # if appended to the current sample phi.
        avg_u = pd.Series(dtype=float)

        for i in ind_m:
            # Build a hypothetical sample: everything drawn so far, PLUS
            # candidate i. Reduce the indicator matrix to just those columns.
            ind_m_ = ind_m[phi + [i]]

            # Average uniqueness of every column in this hypothetical sample,
            # then take the LAST one — that's the uniqueness of i itself,
            # given everything already in phi.
            avg_u.loc[i] = get_avg_uniqueness(ind_m_).iloc[-1]

        # Normalize uniqueness scores into a probability distribution.
        # Higher uniqueness (given current sample) → higher draw probability.
        prob = avg_u / avg_u.sum()

        # Draw one observation weighted by these probabilities, and add it
        # to phi. Note: this IS sampling with replacement — i could already
        # be in phi, and could be drawn again (though its probability will
        # typically be low if it overlaps heavily with itself).
        phi += [np.random.choice(ind_m.columns, p=prob)]

    return phi
