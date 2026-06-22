import numpy as np
import pandas as pd
import sys
import os

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '..')))
from utils.multiprocess import mp_pandas_obj
from .co_events import mp_num_co_events

# Estimating the Average Uniqueness of a Label — AFML Chapter 4, Snippet 4.2, page 61
#
# Computes the average uniqueness of each event over its entire lifespan.
# Uniqueness answers the question: "how much of this label's information is
# NOT shared with other overlapping labels?"
#
# --- The intuition, revisited ---
# Snippet 4.1 (mp_num_co_events) told us, bar by bar, how many events are
# touching each bar (concurrency). This snippet uses that concurrency count
# to compute, for each EVENT, the average of 1/concurrency across every bar
# it touches. If an event never overlaps with anything, every bar it touches
# has concurrency 1, so 1/1 = 1.0 averaged across its whole lifespan — perfectly
# unique. If an event overlaps heavily with others, most of its bars have high
# concurrency, so 1/concurrency is small, and its average uniqueness drops
# below 1.0.
#
# --- Worked example from the book (Section 4.5.3) ---
# Three events: obs1 spans bars 0-2, obs2 spans bars 2-3, obs3 spans bars 4-5.
# Bar 2 is the only bar with concurrency 2 (touched by both obs1 and obs2).
#   ū_1 = (1/1 + 1/1 + 1/2) / 3 = 5/6 ≈ 0.833
#   ū_2 = (1/2 + 1/1) / 2       = 3/4 ... (book example uses a slightly
#          different overlap window — see test file for the exact case verified)
#   ū_3 = (1/1 + 1/1) / 2       = 1.0   (no overlap at all)


def mp_sample_tw(t1, num_co_events, molecule):
    # Derive average uniqueness over the event's lifespan — AFML Snippet 4.2
    #
    # --- Inputs ---
    # t1            : pd.Series — index = event start (t_in), values = event end (t_out)
    # num_co_events : pd.Series — output of mp_num_co_events, indexed by bar,
    #                 values = concurrency count at that bar
    # molecule      : pd.DatetimeIndex — subset of event start dates this
    #                 worker is responsible for
    #
    # --- Output ---
    # pd.Series indexed by event start date, values = average uniqueness ∈ (0, 1]

    wght = pd.Series(index=molecule, dtype=float)

    for t_in, t_out in t1.loc[wght.index].items():
        # For this event, look at concurrency across every bar it touches
        # (from its start t_in to its end t_out), take 1/concurrency at each
        # bar, and average across the event's whole lifespan.
        wght.loc[t_in] = (1. / num_co_events.loc[t_in:t_out]).mean()

    return wght


def get_average_uniqueness(close, events, num_threads=1):
    # Orchestrator — combines Snippet 4.1 and Snippet 4.2 to produce the
    # final per-event uniqueness column ('tW') referenced throughout the book
    # (e.g. max_samples=out['tW'].mean() in Section 4.5.1).
    #
    # This is the function you call directly. It handles all the
    # mp_pandas_obj plumbing internally.
    #
    # --- Inputs ---
    # close       : pd.Series — closing prices indexed by datetime
    # events      : pd.DataFrame — output of get_events() from Chapter 3,
    #               must have a 't1' column (first-touch timestamps)
    # num_threads : int — number of parallel workers (default 1)
    #
    # --- Output ---
    # pd.Series indexed by event start date, values = average uniqueness

    # Step 1: Compute concurrency at every bar (Snippet 4.1), in parallel
    num_co_events = mp_pandas_obj(
        func=mp_num_co_events,
        pd_obj=('molecule', events.index),
        num_threads=num_threads,
        close_idx=close.index,
        t1=events['t1']
    )

    # Step 2: Dedupe overlapping bars computed by multiple workers
    # (different molecules can compute the same boundary bars — keep just one)
    num_co_events = num_co_events.loc[~num_co_events.index.duplicated(keep='last')]

    # Step 3: Expand to the FULL price index, filling untouched bars with 0
    num_co_events = num_co_events.reindex(close.index).fillna(0)

    # Step 4: Compute average uniqueness per event (Snippet 4.2), in parallel
    tw = mp_pandas_obj(
        func=mp_sample_tw,
        pd_obj=('molecule', events.index),
        num_threads=num_threads,
        t1=events['t1'],
        num_co_events=num_co_events
    )

    return tw
