import numpy as np
import pandas as pd

# Fixed-Time Horizon Labeling — AFML Chapter 3, Section 3.2, page 43
#
# Labels each observation in a feature matrix as +1, 0, or -1 based on
# whether the price return over the next h bars exceeds a threshold τ.
#
# --- What problem does this solve? ---
# Machine learning models need labeled training data. For a trading strategy,
# each bar needs a label telling the model whether that was a good entry point.
# The simplest approach is to look h bars into the future and check whether
# the price moved enough in either direction to be considered meaningful.
#
# --- Limitations ---
# This method ignores what happened BETWEEN the entry and the h-th bar.
# A trade that dropped 10% before recovering to +1% over h bars gets labeled
# +1, even though most real traders would have been stopped out. The Triple
# Barrier Method (Section 3.3) addresses this limitation.
#
# --- Formula (page 43) ---
# r_{t_{i,0}, t_{i,0}+h} = P_{t_{i,0}+h} / P_{t_{i,0}} - 1
#
# y_i = -1  if r < -τ    (price fell more than threshold → bad trade)
#         0  if |r| ≤ τ  (price moved less than threshold → neutral)
#         1  if r > τ    (price rose more than threshold → good trade)
#
# where:
#   t_{i,0}   = index of the bar immediately after observation X_i
#   t_{i,0}+h = index of the h-th bar after t_{i,0}
#   τ         = pre-defined constant threshold (e.g. 0.01 = 1%)


def fixed_time_horizon(close, events, h, threshold):
    # Fixed-Time Horizon Labeling — AFML Chapter 3, Section 3.2, page 43
    #
    # Labels each event date as +1, 0, or -1 based on the price return
    # over the next h bars compared to a threshold τ.
    #
    # --- Inputs ---
    # close     : pd.Series — closing prices indexed by datetime
    #             Each value is the closing price of one bar.
    # events    : pd.DatetimeIndex — dates of observations to label
    #             Typically the output of the CUSUM filter (Section 2.5).
    #             Only these dates will be labeled; all others are ignored.
    # h         : int — number of bars to look forward
    #             e.g. h=5 means look 5 bars ahead to evaluate the trade.
    # threshold : float — the τ value, expressed as a decimal return
    #             e.g. 0.01 = 1% threshold. Returns between -1% and +1%
    #             get label 0. Returns above 1% get +1, below -1% get -1.
    #
    # --- Output ---
    # pd.Series indexed by event dates, values in {-1, 0, 1}
    #   +1 = price rose more than threshold over h bars (buy signal confirmed)
    #    0 = price moved less than threshold (neutral, not enough signal)
    #   -1 = price fell more than threshold (sell signal confirmed)

    # Convert close prices to a list-based index so we can look up
    # integer positions (needed to step h bars forward from any date)
    close_index = list(close.index)

    labels = []     # will collect (date, label) pairs
    dates  = []     # event dates that were successfully labeled

    for event_date in events:

        # --- Find the integer position of this event in the price series ---
        # t_{i,0} in the book: the bar immediately after observation X_i.
        # We use the event date itself as t_{i,0} since CUSUM fires on the
        # bar where the threshold was crossed.
        if event_date not in close_index:
            # Skip events that don't have a matching bar in the price series
            # (can happen if events were generated from a different dataset)
            continue

        t0 = close_index.index(event_date)  # integer position of entry bar

        # --- Check that h bars ahead exists in the series ---
        # If the event is too close to the end of the data, we can't look
        # h bars forward, so we skip it.
        t1 = t0 + h                         # integer position of exit bar
        if t1 >= len(close_index):
            continue                         # not enough future data

        # --- Compute the price return over h bars ---
        # r = P_{t0+h} / P_{t0} - 1
        # This is the percentage change from entry price to exit price.
        p0 = close.iloc[t0]     # entry price (price at event date)
        p1 = close.iloc[t1]     # exit price (price h bars later)
        r  = p1 / p0 - 1        # percentage return

        # --- Assign label based on threshold τ ---
        # Formula (page 43):
        #   y_i = -1  if r < -τ
        #          0  if |r| ≤ τ
        #          1  if r > τ
        if r > threshold:
            label = 1
        elif r < -threshold:
            label = -1
        else:
            label = 0

        labels.append(label)
        dates.append(event_date)

    return pd.Series(labels, index=pd.DatetimeIndex(dates), name='label')
