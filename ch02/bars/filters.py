import numpy as np
import pandas as pd

def cusum_filter(df, h):
    # Symmetric CUSUM Filter — AFML Chapter 2, Section 2.5.2, page 39
    # (Snippet 2.4 in the book)
    #
    # A quality-control method that detects when a price series has drifted
    # significantly in either direction from a reset level of zero.
    # Fires an event when cumulative drift exceeds threshold h, then resets.
    # Avoids triggering multiple events when price hovers near a threshold
    # (unlike Bollinger Bands).
    #
    # --- What problem does this solve? ---
    # Many trading strategies generate candidate signals at every bar. But if
    # the price hasn't moved meaningfully, acting on those signals wastes
    # transaction costs and statistical power. The CUSUM filter acts as a
    # "pre-filter" — it outputs only those dates where the price has drifted
    # by at least h away from the level it was at the LAST TIME it fired.
    #
    # Unlike a Bollinger Band (which fires whenever price is outside a fixed band
    # and can fire on many consecutive bars in a row), the CUSUM filter RESETS
    # after firing. It then has to accumulate a full h of drift before firing
    # again. This prevents repeated signals during a slow trending move.
    #
    # --- Visual intuition ---
    # Imagine a ball on a rubber band anchored at 0. Every price uptick stretches
    # the band upward (S+_t grows); every downtick compresses it (or grows S-_t).
    # When the stretch exceeds h, the rubber band SNAPS (event fires, reset to 0),
    # and the process begins again from scratch.
    #
    # Formulas (page 39):
    #   S+_t = max{0, S+_{t-1} + y_t - E_{t-1}[y_t]},  S+_0 = 0
    #   S-_t = min{0, S-_{t-1} + y_t - E_{t-1}[y_t]},  S-_0 = 0
    #   S_t  = max{S+_t, -S-_t}
    #
    # An event fires at t if S_t >= h, at which point S+_t and S-_t are reset to 0.
    # Here E_{t-1}[y_t] = y_{t-1} (previous price), so y_t - E_{t-1}[y_t] = Δp_t

    events = []   # list of dates where a signal fired
    s_pos = 0     # S+_t accumulator (tracks upward drift from last reset)
    s_neg = 0     # S-_t accumulator (tracks downward drift from last reset)

    # Compute Δp_t — price change between consecutive bars
    # diff() subtracts each row from the next: diff[i] = Price[i] - Price[i-1]
    # The very first row has no predecessor so diff[0] = NaN (handled below)
    prices = df['Price']
    diff = prices.diff()

    for date, delta in zip(df['Date'], diff):
        if pd.isna(delta):  # skip the first row (no previous price to diff against)
            continue

        # Update the upward drift accumulator S+_t.
        # S+_t = max{0, S+_{t-1} + Δp_t}
        #   - If price went up (delta > 0): S+ grows (upward drift is accumulating).
        #   - If price went down (delta < 0): S+ shrinks, but never goes below 0.
        #     (max{0, ...} is the "reset floor" — we don't let S+ go negative.)
        s_pos = max(0, s_pos + delta)

        # Update the downward drift accumulator S-_t.
        # S-_t = min{0, S-_{t-1} + Δp_t}
        #   - If price went down (delta < 0): S- grows more negative (downward drift).
        #   - If price went up (delta > 0): S- shrinks toward 0, but never exceeds 0.
        #     (min{0, ...} is the "reset ceiling" — we don't let S- go positive.)
        s_neg = min(0, s_neg + delta)

        if s_pos >= h:
            # The upward cumulative drift has exceeded threshold h.
            # An upward move of magnitude h has occurred since the last reset.
            # Fire an event at this date and reset S+ to 0 (the "snap").
            # S- is left unchanged — it keeps tracking any residual downward drift.
            s_pos = 0
            events.append(date)

        elif abs(s_neg) >= h:
            # The downward cumulative drift has exceeded threshold h in magnitude.
            # abs(s_neg) converts S- (which is ≤ 0) to a positive number for comparison.
            # Fire an event and reset S- to 0.
            # We use elif (not if) so only ONE event fires per bar — never both.
            s_neg = 0
            events.append(date)

    # Return the event dates as a DatetimeIndex so they can be used directly
    # to slice other time-indexed dataframes (e.g. to label training samples).
    return pd.DatetimeIndex(events)
