import numpy as np
import pandas as pd
import sys
import os

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '..')))
from utils.multiprocess import mp_pandas_obj
from .co_events import mp_num_co_events

# Determination of Sample Weight by Absolute Return Attribution —
# AFML Chapter 4, Section 4.6, Snippet 4.10, page 69
#
# A SECOND, complementary weighting scheme to sequential bootstrap. Where
# sequential bootstrap (Section 4.5) weights observations by how UNIQUE they
# are (how little they overlap with others), this section weights
# observations by how MUCH PRICE MOVEMENT they're associated with.
#
# --- The core idea ---
# Not all labels are equally informative. A label generated during a huge,
# decisive price move (large |return|) tells the model much more than a label
# generated during a quiet, noisy chop. We want labels associated with bigger
# moves to count for MORE during training — but we still need to account for
# overlap: if 5 events are all "claiming credit" for the same big move at the
# same bar, that move's contribution gets split 5 ways, same discounting
# principle as average uniqueness.
#
# --- Why log-returns specifically? ---
# We need to SUM returns across a range of bars (from event start to event
# end). Simple returns don't add up nicely across multiple bars — a 10% gain
# followed by a 10% loss is NOT a 0% round trip (it's actually a small loss,
# since 1.10 × 0.90 = 0.99). Log-returns fix this: log-return over bars 1→5
# exactly equals the SUM of log-returns over 1→2, 2→3, 3→4, 4→5. This
# "additive" property is essential for summing returns bar-by-bar the way
# this function does.
#
# --- Sequential bootstrap weight vs return-attribution weight ---
# These are TWO DIFFERENT weighting schemes, often used for different purposes:
#   - Average uniqueness (Chapter 4.5): used to control bootstrap SAMPLING
#     (e.g. max_samples in BaggingClassifier, or as the basis for seq_bootstrap)
#   - Return attribution (Chapter 4.6): used as the actual sample_weight
#     passed to a classifier's .fit() method, so the model pays more
#     attention to high-conviction labels during training


def mp_sample_w(t1, num_co_events, close, molecule):
    # Derive sample weight by return attribution — AFML Snippet 4.10
    #
    # --- Inputs ---
    # t1            : pd.Series — index = event start (t_in), values = event end (t_out)
    # num_co_events : pd.Series — output of mp_num_co_events, indexed by bar,
    #                 values = concurrency count at that bar
    # close         : pd.Series — closing prices indexed by datetime
    # molecule      : pd.DatetimeIndex — subset of event start dates this
    #                 worker is responsible for
    #
    # --- Output ---
    # pd.Series indexed by event start date, values = absolute attributed
    # return weight (always >= 0, NOT yet normalized to sum to num_observations
    # — that final rescaling happens in get_sample_weights below)

    # Log-returns, so they're additive across consecutive bars.
    # ret.loc[t] = log(close[t]) - log(close[t-1])
    ret = np.log(close).diff()

    wght = pd.Series(index=molecule, dtype=float)

    for t_in, t_out in t1.loc[wght.index].items():
        # For this event, sum the discounted return across every bar in its
        # lifespan. At each bar, the return is divided by that bar's
        # concurrency — same discounting principle as average uniqueness:
        # if 3 events are all open at this bar, each only gets 1/3 credit
        # for whatever happened there.
        wght.loc[t_in] = (ret.loc[t_in:t_out] / num_co_events.loc[t_in:t_out]).sum()

    # We care about MAGNITUDE of attributed return, not direction — a label
    # tied to a huge move should get a large weight whether that move was
    # up or down.
    return wght.abs()


def get_sample_weights(close, events, num_threads=1):
    # Orchestrator — combines mp_num_co_events (Snippet 4.1) and mp_sample_w
    # (Snippet 4.10) to produce the final, NORMALIZED sample weight column
    # ready to be passed directly as sample_weight to a classifier's .fit().
    #
    # This mirrors the structure of get_average_uniqueness() (uniqueness.py)
    # — same concurrency computation, same mp_pandas_obj plumbing, just a
    # different per-event aggregation (summed attributed return instead of
    # averaged uniqueness).
    #
    # --- Inputs ---
    # close       : pd.Series — closing prices indexed by datetime
    # events      : pd.DataFrame — output of get_events() from Chapter 3,
    #               must have a 't1' column (first-touch timestamps)
    # num_threads : int — number of parallel workers (default 1)
    #
    # --- Output ---
    # pd.Series indexed by event start date, values = sample weight.
    # Weights are rescaled so they SUM to the number of observations —
    # this keeps the average weight at 1.0, a standard normalization
    # convention so weights are comparable across different datasets.

    # Step 1: Compute concurrency at every bar (Snippet 4.1), in parallel
    # (identical to the first step of get_average_uniqueness)
    num_co_events = mp_pandas_obj(
        func=mp_num_co_events,
        pd_obj=('molecule', events.index),
        num_threads=num_threads,
        close_idx=close.index,
        t1=events['t1']
    )
    num_co_events = num_co_events.loc[~num_co_events.index.duplicated(keep='last')]
    num_co_events = num_co_events.reindex(close.index).fillna(0)

    # Step 2: Compute absolute attributed return per event (Snippet 4.10), in parallel
    w = mp_pandas_obj(
        func=mp_sample_w,
        pd_obj=('molecule', events.index),
        num_threads=num_threads,
        t1=events['t1'],
        num_co_events=num_co_events,
        close=close
    )

    # Step 3: Rescale so weights sum to the number of observations
    # (keeps average weight at 1.0 — standard normalization convention)
    w *= w.shape[0] / w.sum()

    return w
