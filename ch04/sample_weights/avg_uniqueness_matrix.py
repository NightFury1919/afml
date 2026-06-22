import numpy as np
import pandas as pd

# Compute Average Uniqueness — AFML Chapter 4, Snippet 4.4, page 65
#
# Computes the average uniqueness of every event directly from the indicator
# matrix (Snippet 4.3). This is the matrix-based equivalent of mp_sample_tw
# (Snippet 4.2) — same underlying concept, different implementation strategy.
#
# --- Why two different implementations of the same idea? ---
# mp_sample_tw (uniqueness.py) is the production version: it works bar-by-bar
# without ever building a full dense matrix, which is essential when you have
# millions of bars and thousands of events — a full matrix would be enormous.
#
# get_avg_uniqueness (this file) works directly off the indicator matrix,
# which is wasteful for large real datasets, but is exactly what the
# sequential bootstrap algorithm (Snippet 4.5) needs internally — it has to
# repeatedly ask "what would the average uniqueness be if I added observation
# i to my current sample?", which is a fast column-slice operation on a matrix
# but awkward to do bar-by-bar.
#
# --- The math ---
# concurrency_t = sum across all events of indM[t, :]   (row sum)
# uniqueness[t, i] = indM[t, i] / concurrency_t          (element-wise divide)
# avg_uniqueness[i] = mean of uniqueness[t, i] over all t where indM[t, i] > 0
#                      (only average over bars this event actually touches)


def get_avg_uniqueness(ind_m):
    # Compute average uniqueness from the indicator matrix — AFML Snippet 4.4
    #
    # --- Inputs ---
    # ind_m : pd.DataFrame — output of get_ind_matrix() (Snippet 4.3)
    #         shape (num_bars, num_events), binary 0/1 entries
    #
    # --- Output ---
    # pd.Series indexed by event (column), values = average uniqueness ∈ (0, 1]

    # Concurrency at each bar: how many events touch this bar?
    # Row sum across all event columns.
    c = ind_m.sum(axis=1)

    # Uniqueness contribution of each event at each bar it touches.
    # Divide every column by the SAME concurrency series (broadcast across columns).
    # Where ind_m is 0, this stays 0 (event doesn't touch this bar).
    # Where ind_m is 1, this becomes 1/concurrency (the event's "share" of this bar).
    u = ind_m.div(c, axis=0)

    # Average uniqueness per event: average only over the bars this event
    # actually touches (u > 0) — bars it doesn't touch shouldn't pull the
    # average toward zero, they're simply not relevant to this event.
    avg_u = u[u > 0].mean()

    return avg_u
