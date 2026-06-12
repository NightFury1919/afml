import numpy as np
import pandas as pd

def ewma(arr, window):
    # Exponentially Weighted Moving Average (EWMA)
    # Used throughout chapter 2 to adaptively update expected number of ticks
    # and expected imbalance after each bar closes. Referenced on pages 31-32.
    #
    # Formula: ewma_t = alpha * x_t + (1 - alpha) * ewma_{t-1}
    # where alpha = 2 / (window + 1)
    #
    # --- What is an EWMA and why use it? ---
    # A regular average treats every data point equally.
    # An EWMA gives MORE weight to recent observations and LESS weight to older ones.
    # This means the average "follows" recent data more closely — ideal for
    # financial data where recent behavior is more predictive than distant history.
    #
    # alpha = 2 / (window + 1) is the "smoothing factor":
    #   - alpha close to 1  → nearly all weight on the newest value (very reactive)
    #   - alpha close to 0  → nearly all weight on history    (very smooth / slow)
    #
    # Example with window=3:  alpha = 2/(3+1) = 0.5
    #   If arr = [10, 12, 8]:
    #     ewma_0 = 10                          (start with first value)
    #     ewma_1 = 0.5*12 + 0.5*10 = 11.0
    #     ewma_2 = 0.5*8  + 0.5*11 = 9.5
    #
    # This is used in bar construction to ask: "how long were recent bars?"
    # and adaptively set the threshold for the next bar accordingly.

    if len(arr) == 0:
        # If the input list is empty there is nothing to average — return 0
        return 0

    # alpha is the smoothing factor — controls how fast old values decay
    alpha = 2 / (window + 1)

    # Seed the running average with the very first value in the array.
    # There is no "previous" average yet, so we use the first data point.
    ewma_val = arr[0]

    for i in range(1, len(arr)):
        # Each step: blend the new value (arr[i]) with the current running average.
        # New value gets weight alpha; running history gets weight (1 - alpha).
        ewma_val = alpha * arr[i] + (1 - alpha) * ewma_val

    return ewma_val


def delta(df):
    # Computes Δp_t — the price change between consecutive ticks.
    # Used as input to the Tick Rule (page 29).
    #
    # --- Why do we need price differences? ---
    # The Tick Rule (below) needs to know whether the price WENT UP or DOWN
    # at each trade. To find that out, we subtract the previous price from
    # the current price. The sign of that difference (+/-) tells us the direction.
    #
    # np.diff() computes arr[i] - arr[i-1] for each i, producing an array that is
    # one element shorter than the original. We insert a 0 at the front so the
    # output stays the same length as the input (the very first trade has no
    # "previous" price, so its price change is defined as 0).

    a = np.diff(df['Price'])        # array of price changes, length = n-1
    a = np.insert(a, 0, 0)         # prepend 0 → now length = n, aligned with rows
    df['Delta'] = a                 # store as a new column in the dataframe
    return df


def tick_rule(df):
    # Tick Rule — AFML Chapter 2, page 29
    # Assigns a direction b_t to each trade:
    #   b_t = b_{t-1}         if Δp_t = 0  (price unchanged, carry forward)
    #   b_t = |Δp_t| / Δp_t  if Δp_t ≠ 0  (gives +1 for uptick, -1 for downtick)
    # b_t is used as a proxy for trade direction (buy = +1, sell = -1)
    #
    # --- Why do we need trade direction labels? ---
    # In real exchanges, individual trades are often NOT labelled as buyer-initiated
    # or seller-initiated in public data. The Tick Rule is a simple heuristic:
    #   - If the price ROSE since the last trade → the buyer was aggressive → b = +1
    #   - If the price FELL since the last trade → the seller was aggressive → b = -1
    #   - If the price DID NOT CHANGE → we cannot tell, so we carry forward the
    #     previous label (the last known direction is our best guess).
    #
    # |Δp_t| / Δp_t is just a compact way to write the SIGN of Δp_t:
    #   positive Δp_t → |Δp_t| / Δp_t = Δp_t / Δp_t = +1
    #   negative Δp_t → |Δp_t| / Δp_t = (-Δp_t) / Δp_t = -1
    #
    # These labels feed directly into imbalance bars and run bars.

    # Start every trade with label +1 as a default (will be overwritten)
    b = np.ones(len(df['Price']))

    for i, delta in enumerate(df['Delta']):
        if i > 0:                           # skip the first row (nothing to compare to)
            if delta == 0:
                b[i] = b[i-1]              # price flat → carry forward previous direction
            else:
                b[i] = abs(delta) / delta  # price moved → +1 (up) or -1 (down)

    df['Label'] = b     # store direction labels as a new column
    return df


def estimate_buy_sell_probs(df):
    prob = pd.DataFrame(pd.pivot_table(df, index='Label', values='Price', aggfunc='count'))
    prob = np.array(prob)

    # Original code assumes both -1 and +1 rows always exist — crashes if all buys or all sells
    # Fix: check how many rows the pivot produced
    if len(prob) == 1:
        # Only one label present — check which one
        only_label = df['Label'].iloc[0]
        if only_label == 1:
            return 1.0, 0.0   # all buys
        else:
            return 0.0, 1.0   # all sells

    p_b = prob[1] / (prob[0] + prob[1])
    p_s = prob[0] / (prob[0] + prob[1])
    return p_b, p_s