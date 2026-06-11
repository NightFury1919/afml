import numpy as np
import pandas as pd
from .utils import ewma

# Information-Driven Imbalance Bars — AFML Chapter 2, Section 2.3.2
# Imbalance bars close when the accumulated signed imbalance exceeds
# an adaptive expected threshold, causing more frequent sampling during
# periods of informed trading and less frequent sampling during quiet periods.
#
# --- What is an "imbalance" and why does it matter? ---
# Every trade is either buyer-initiated (b_t = +1) or seller-initiated (b_t = -1).
# If the market is perfectly balanced, buy and sell orders cancel out and
# the "imbalance" (their running sum) hovers near zero.
#
# But when a large informed trader is active — someone who KNOWS something the
# market doesn't — they tend to trade in ONE direction repeatedly. This causes
# the running sum of +1s and -1s to drift strongly away from zero. That drift
# is our signal of informed trading activity.
#
# Imbalance bars close precisely when this drift becomes unexpectedly large,
# so each bar represents the same AMOUNT OF INFORMATION, not the same amount
# of time or volume. During normal markets, bars are infrequent; during
# bursts of informed trading, bars are very frequent.


def tick_imbalance_bars(df, expected_num_ticks_init=10, num_prev_bars=3):
    # Tick Imbalance Bars (TIBs) — AFML Chapter 2, page 29
    # Uses tick direction only (v_t = 1 for every trade).
    #
    # --- Conceptual overview ---
    # We accumulate θ_T = Σ b_t (sum of +1s and -1s as trades arrive).
    # The bar closes when |θ_T| — the absolute magnitude of this running sum —
    # exceeds the EXPECTED magnitude given a "normal" bar of this length.
    #
    # Expected magnitude: E[T] * |2*P[b_t=1] - 1|
    #   E[T]             = expected number of ticks in a bar (EWMA of past bar lengths)
    #   P[b_t=1]         = probability a tick is a buy (EWMA of past proportions)
    #   |2*P[b_t=1] - 1| = expected imbalance per tick (0 if 50/50, 1 if all buys/sells)
    #
    # If the market is perfectly balanced (50% buys, 50% sells):
    #   |2*0.5 - 1| = 0 → expected imbalance is 0 → threshold is effectively 0
    #   → every tick would close a bar (useless). In practice, the threshold is
    #     clipped to a minimum (1e-6) to avoid division-by-zero.
    #
    # If 80% of ticks are buys:
    #   |2*0.8 - 1| = 0.6 → the expected imbalance per tick is 0.6
    #   → bars close less frequently because drift is "expected"

    cum_theta = 0       # θ_T: running sum of b_t values within the current bar
    collector = []      # list of prices within the current bar
    bars = []           # completed bars, each stored as a tuple

    imbalance_array = []    # ALL b_t values ever seen (grows across all bars)
                            # used to compute the rolling EWMA of imbalance
    bar_lengths = []        # number of ticks in each completed bar

    num_ticks = 0                               # tick counter for the current bar
    expected_num_ticks = expected_num_ticks_init  # E[T]: initial guess for bar length
    expected_imbalance = 0                      # E[θ]: starts at 0 until warmup finishes

    for i, (label, price, date) in enumerate(zip(df['Label'], df['Price'], df['Date'])):

        # Tick Imbalance Accumulation — page 29
        # θ_T = Σ b_t, from t = 1 to T
        # Each tick contributes +1 (buy) or -1 (sell) to the running total
        imbalance = label           # b_t (for tick bars, volume weight v_t = 1)
        imbalance_array.append(imbalance)   # store globally for EWMA lookback
        cum_theta += imbalance      # θ_T = running sum of b_t; grows toward ±∞

        collector.append(price)
        num_ticks += 1

        # --- Warmup period ---
        # We can't compute a meaningful expected imbalance until we have seen
        # at least expected_num_ticks_init trades. Before that, no bar closes.
        # Once we have enough data, we use EWMA over all seen imbalances
        # to estimate the typical per-tick imbalance magnitude.
        if len(bars) == 0 and len(imbalance_array) >= expected_num_ticks_init:
            expected_imbalance = max(
                ewma(imbalance_array, expected_num_ticks_init),
                1e-6    # floor to avoid threshold = 0 (would close every tick)
            )

        # Tick Imbalance Bar Stopping Formula — page 29
        # T* = arg min_T { |θ_T| >= E0[T] * |2P[b_t=1] - 1| }
        # Close the bar when actual imbalance exceeds the expected imbalance.
        #
        # In plain English: "close the bar as soon as the signed running sum of
        # trade directions is larger (in absolute value) than what we would expect
        # from a randomly ordered sequence of E[T] ticks."
        #
        # expected_num_ticks * abs(expected_imbalance) is the full threshold:
        #   E[T]       = typical number of ticks per bar
        #   |E[θ/tick]| = expected imbalance per tick
        #   product    = expected total imbalance in a typical bar
        if expected_imbalance != 0 and abs(cum_theta) >= expected_num_ticks * abs(expected_imbalance):

            # Compute bar OHLC
            open_p  = collector[0]
            high_p  = np.max(collector)
            low_p   = np.min(collector)
            close_p = collector[-1]

            bars.append((date, i, open_p, low_p, high_p, close_p))
            bar_lengths.append(num_ticks)   # record how long this bar was

            # --- Reset accumulators for the next bar ---
            cum_theta = 0
            collector = []
            num_ticks = 0

            # Tick Imbalance Bar Expected Imbalance update — page 29
            # After closing each bar, update both estimates using EWMA:
            #   expected_num_ticks: EWMA of the last `num_prev_bars` bar lengths
            #   expected_imbalance: EWMA over the last (num_prev_bars * E[T]) ticks
            #
            # Using num_prev_bars * expected_num_ticks as the EWMA window ensures
            # we look back across approximately num_prev_bars worth of bars of data.
            expected_num_ticks = ewma(bar_lengths, num_prev_bars)
            expected_imbalance = max(
                ewma(
                    imbalance_array,
                    max(1, int(num_prev_bars * expected_num_ticks))
                ),
                1e-6
            )

    cols = ['Date', 'Index', 'Open', 'Low', 'High', 'Close']
    return pd.DataFrame(bars, columns=cols)


def volume_imbalance_bars(df, expected_num_ticks_init=10, num_prev_bars=3):
    # Volume/Dollar Imbalance Bars (VIBs/DIBs) — AFML Chapter 2, page 30
    # Extends tick imbalance bars by weighting each tick by its volume (v_t).
    #
    # --- Difference from tick imbalance bars ---
    # In tick imbalance bars, every trade counts as ±1 regardless of size.
    # Here, a trade of 100 units counts as ±100. This means large trades have
    # proportionally larger impact on the imbalance, which is more realistic —
    # a single 10,000-share block trade from an institution carries much more
    # information than 10,000 individual 1-share retail trades.
    #
    # The stopping rule is otherwise identical:
    #   θ_T = Σ b_t * v_t  (volume-weighted signed sum)
    #   Close when |θ_T| >= E[T] * |2v+ - E[v_t]|
    #   where v+ is the expected volume of a buy trade.

    cum_theta = 0       # θ_T: running volume-weighted imbalance sum
    cumm_vol  = 0       # total volume in current bar (for VWAP denominator)
    vol_price = 0       # Σ(price × volume) in current bar (for VWAP numerator)

    collector = []
    bars = []

    imbalance_array = []    # all signed volume imbalances b_t * v_t seen so far
    bar_lengths = []        # number of ticks in each completed bar

    num_ticks = 0
    expected_num_ticks = expected_num_ticks_init
    expected_imbalance = 0

    for i, (label, price, date, volume) in enumerate(zip(df['Label'], df['Price'], df['Date'], df['Volume'])):

        # Volume/Dollar Imbalance Accumulation — page 30
        # θ_T = Σ b_t * v_t, from t = 1 to T
        # Each tick contributes its signed volume to the running total.
        # label = +1 (buy) or -1 (sell), volume = units traded in this tick.
        imbalance = label * volume      # b_t * v_t: signed volume contribution
        imbalance_array.append(imbalance)
        cum_theta += imbalance          # accumulate signed volume

        cumm_vol  += volume             # track total volume for VWAP
        vol_price += price * volume     # track Σ(p*v) for VWAP
        collector.append(price)
        num_ticks += 1

        # Warmup: wait until we have seen enough ticks before starting to close bars.
        # Once warmed up, use EWMA of all signed volumes to estimate E[b_t * v_t].
        if len(bars) == 0 and len(imbalance_array) >= expected_num_ticks_init:
            expected_imbalance = max(
                ewma(imbalance_array, expected_num_ticks_init),
                1e-6
            )

        # V/D Imbalance Bar Stopping Rule — page 30
        # T* = arg min_T { |θ_T| >= E0[T] * |2v+ - E0[v_t]| }
        # Close the bar when actual volume imbalance exceeds expected.
        #
        # The threshold now scales with VOLUME rather than just tick count,
        # so a bar during high-volume informed trading closes after fewer
        # ticks than a bar during low-volume noise trading.
        if expected_imbalance != 0 and abs(cum_theta) >= expected_num_ticks * abs(expected_imbalance):

            open_p  = collector[0]
            high_p  = np.max(collector)
            low_p   = np.min(collector)
            close_p = collector[-1]

            # VWAP = Σ(price × volume) / Σ(volume)
            # Gives the "true average transaction price" for the bar,
            # weighted by how much was traded at each level.
            vwap = vol_price / cumm_vol

            bars.append((date, i, open_p, low_p, high_p, close_p, vwap))
            bar_lengths.append(num_ticks)

            # Reset all accumulators for the next bar
            cum_theta = 0
            cumm_vol  = 0
            vol_price = 0
            collector = []
            num_ticks = 0

            # V/D Expected Imbalance Equation update — page 30
            # E0[θ_T] = E0[T] * (2v+ - E0[v_t])
            # Update E0[T] and (2v+ - E0[v_t]) using EWMA of prior bar values.
            # Window for imbalance EWMA = num_prev_bars * E[T] ticks, so we
            # always look back across a fixed number of recent bars worth of ticks.
            expected_num_ticks = ewma(bar_lengths, num_prev_bars)
            expected_imbalance = max(
                ewma(
                    imbalance_array,
                    max(1, int(num_prev_bars * expected_num_ticks))
                ),
                1e-6
            )

    cols = ['Date', 'Index', 'Open', 'Low', 'High', 'Close', 'Vwap']
    result = pd.DataFrame(bars, columns=cols)
    result['Date'] = pd.to_datetime(result['Date'])
    return result
