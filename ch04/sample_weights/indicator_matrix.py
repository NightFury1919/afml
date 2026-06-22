import numpy as np
import pandas as pd

# Build an Indicator Matrix — AFML Chapter 4, Snippet 4.3, page 65
#
# Builds a binary matrix showing exactly which bars are touched by which
# events. Row = bar, column = event. A 1 means "this event's outcome window
# includes this bar."
#
# --- Why do we need this? ---
# Snippets 4.1 and 4.2 (mp_num_co_events, mp_sample_tw) compute the SAME
# information this matrix represents, but do it efficiently bar-by-bar
# without ever materializing the full matrix — important when you have
# millions of bars and events, since a dense matrix would be huge.
#
# This matrix is mainly useful for SMALL, illustrative examples (like the
# book's 3-observation worked example) and for the sequential bootstrap
# algorithm (Snippet 4.5), which explicitly needs the matrix structure to
# test "what if I added observation i to my current sample?"
#
# --- Reading the matrix ---
# indM.loc[bar, event] == 1  →  this bar falls within this event's window
# indM.loc[bar, event] == 0  →  this bar is NOT part of this event's window
#
# Concurrency at a bar = row sum (how many events touch this bar)
# Uniqueness of an event = related to column values divided by concurrency
#   (this is exactly what get_avg_uniqueness, Snippet 4.4, computes next)


def get_ind_matrix(bar_ix, t1):
    # Build the indicator matrix — AFML Snippet 4.3
    #
    # --- Inputs ---
    # bar_ix : pd.Index — the full index of bars (e.g. range(t1.max()+1)
    #          for a simple integer-indexed example, or close.index for
    #          real datetime-indexed price data)
    # t1     : pd.Series — index = event start (t_in), values = event end (t_out)
    #
    # --- Output ---
    # pd.DataFrame — shape (num_bars, num_events)
    #   indM.loc[bar, i] = 1 if bar is in event i's window, else 0

    # Start with an all-zero matrix: one row per bar, one column per event
    ind_m = pd.DataFrame(0, index=bar_ix, columns=range(t1.shape[0]))

    # For each event i, mark every bar between its start (t_in) and
    # end (t_out) with a 1 in that event's column.
    # Note: we rename the loop variables t_in/t_out to avoid shadowing
    # the input parameter t1 (a subtle naming collision in the original
    # book snippet, where the loop variable is also called t1).
    for i, (t_in, t_out) in enumerate(t1.items()):
        ind_m.loc[t_in:t_out, i] = 1.

    return ind_m
