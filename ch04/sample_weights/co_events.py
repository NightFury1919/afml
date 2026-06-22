import numpy as np
import pandas as pd

# Estimating the Uniqueness of a Label — AFML Chapter 4, Snippet 4.1, page 60-61
#
# Computes the number of CONCURRENT events touching each bar in the price series.
# This is the foundation for "average uniqueness" (Chapter 4.5) — before we can
# say how unique a label is, we first need to know, bar by bar, how many other
# labels are also depending on that same bar.
#
# --- Why do we need this? ---
# In Chapter 4.2 we learned that overlapping outcomes share information.
# To quantify HOW MUCH two labels overlap, we first need a bar-by-bar count of
# how many labels are "open" (still accumulating outcome information) at each
# point in time. A bar touched by only 1 event contributes fully to that event's
# uniqueness. A bar touched by 5 events means each of those 5 events only gets
# 1/5 credit for that bar — they're all sharing the same information.
#
# --- What is "molecule"? ---
# Same pattern as Chapter 3's apply_pt_sl_on_t1 — this function is designed to
# be called via mp_pandas_obj for parallelization. 'molecule' is the chunk of
# event start dates assigned to this worker. The book's docstring notes:
#   molecule[0]  = date of the FIRST event this worker computes weights for
#   molecule[-1] = date of the LAST event this worker computes weights for
# Any event that starts before t1[molecule].max() can still impact the count,
# even if that event itself isn't in this molecule's chunk — its bars overlap
# with bars this molecule cares about.


def mp_num_co_events(close_idx, t1, molecule):
    # Compute the number of concurrent events at each bar — AFML Snippet 4.1
    #
    # --- Inputs ---
    # close_idx : pd.DatetimeIndex — full index of all price bars
    # t1        : pd.Series — index = event start dates (t_in)
    #                         values = event end dates (t_out)
    #             This is the same t1 Series produced by add_vertical_barrier()
    #             and get_events() in Chapter 3 — the first-touch timestamps.
    # molecule  : pd.DatetimeIndex — the subset of event start dates this
    #             worker is responsible for (assigned by mp_pandas_obj)
    #
    # --- Output ---
    # pd.Series indexed by bar timestamp, values = number of events open
    #           at that bar (concurrency count)

    # -------------------------------------------------------------------
    # Step 1: Find events that span the period [molecule[0], molecule[-1]]
    # -------------------------------------------------------------------
    # Events still open (t_out is NaT) haven't resolved yet, but they still
    # occupy bars up through the present. Treat their end as the last
    # available bar in the price series — an unresolved event still
    # "uses up" every bar it has touched so far.
    t1 = t1.fillna(close_idx[-1])

    # Drop events that ended BEFORE this molecule's window starts —
    # they can't possibly overlap with any bar we care about here.
    t1 = t1[t1 >= molecule[0]]

    # Drop events that started AFTER this molecule's window ends —
    # same reasoning, they're outside our window of interest.
    t1 = t1.loc[:t1[molecule].max()]

    # -------------------------------------------------------------------
    # Step 2: Count events spanning each bar
    # -------------------------------------------------------------------
    # Find the integer positions in close_idx that bound our period of interest
    iloc = close_idx.searchsorted(np.array([t1.index[0], t1.max()]))

    # Initialise a zero counter across exactly the bars in that range
    count = pd.Series(0, index=close_idx[iloc[0]:iloc[1] + 1])

    # For every surviving event, increment the counter at every bar
    # between its start (t_in) and end (t_out) — this bar was "touched"
    # by this event, so it counts toward concurrency.
    for t_in, t_out in t1.items():
        count.loc[t_in:t_out] += 1

    # -------------------------------------------------------------------
    # Step 3: Return only the slice relevant to this molecule
    # -------------------------------------------------------------------
    # We computed counts across a WIDER range (to correctly account for
    # events that started before/after molecule but still overlap it),
    # but we only return the portion molecule actually asked for.
    return count.loc[molecule[0]:t1[molecule].max()]
