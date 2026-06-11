import numpy as np
import pandas as pd
from .utils import ewma

# Information-Driven Run Bars — AFML Chapter 2, Section 2.3.2
# Run bars monitor sequences of trades in the same direction (runs),
# sampling more frequently when large traders are sweeping the order book.
# Unlike imbalance bars which track net imbalance, run bars track the
# maximum of cumulative buys vs cumulative sells independently.
#
# --- Imbalance bars vs. Run bars: what is the difference? ---
# Imbalance bars compute θ_T = Σ b_t (buys MINUS sells).
#   If 60 buys and 40 sells arrive, θ_T = 60 - 40 = +20.
#
# Run bars compute θ_T = max(Σ buys, Σ sells) independently.
#   Same 60 buys and 40 sells → θ_T = max(60, 40) = 60.
#
# The key insight: in a run bar, the sell-side counter NEVER cancels the
# buy-side counter. Even if the market alternates buy/sell/buy/sell, the
# run bar still sees the maximum one-sided accumulation growing.
#
# Why might this be useful? A large trader who keeps buying will create
# a large buy run. Detecting this "sweep" of the book — where one side
# dominates — can signal aggressive institutional order flow earlier
# than a net-imbalance signal would.


def tick_run_bars(df, expected_num_ticks_init=10, num_prev_bars=3):
    # Tick Run Bars (TRBs) — AFML Chapter 2, page 31
    # Counts raw ticks on each side (buy ticks vs sell ticks).
    #
    # --- Stopping rule ---
    # θ_T = max(cumulative buy ticks, cumulative sell ticks)
    # Close the bar when θ_T >= E[T] * max(P[b_t=1], 1 - P[b_t=1])
    #
    # E[T]         = expected number of ticks in a bar (EWMA of past bar lengths)
    # P[b_t=1]     = expected fraction of buys (EWMA of buy proportions from past bars)
    # max(p, 1-p)  = the probability of the DOMINANT side
    #                (e.g. if 70% buys → max(0.7, 0.3) = 0.7)
    #
    # product E[T] * max(p, 1-p) = expected size of the dominant run in a typical bar
    #
    # So we close the bar when either the buy run OR the sell run grows unexpectedly
    # large relative to what we'd expect by chance.

    pos_run = 0     # cumulative count of buy ticks in current bar (Σ b_t where b_t=+1)
    neg_run = 0     # cumulative count of sell ticks in current bar (-Σ b_t where b_t=-1)
    cumm_vol  = 0   # total volume in current bar (for VWAP)
    vol_price = 0   # Σ(price × volume) in current bar (for VWAP)
    collector = []  # list of prices in current bar
    bars = []       # completed bars

    bar_lengths = []            # number of ticks in each completed bar
    buy_tick_proportions = []   # fraction of ticks that were buys in each completed bar
    num_ticks = 0               # tick counter for the current bar

    expected_num_ticks = expected_num_ticks_init    # E[T]: initial guess for bar length
    expected_p_buy = 0.5        # initial guess: assume 50% buys, 50% sells
    expected_imbalance = 0      # E[θ]: starts at 0 until warmup finishes

    for i, (label, price, date, volume) in enumerate(zip(df['Label'], df['Price'], df['Date'], df['Volume'])):

        # Tick Run Bar Length Definition — page 31
        # θ_T = max( Σ b_t where b_t=1, -Σ b_t where b_t=-1 )
        # Accumulate buy and sell ticks independently — NEVER reset mid-bar.
        # Note: the buy/sell counters are always non-negative integers.
        if label == 1:
            pos_run += 1    # this tick was a buy → increment the buy counter
        elif label == -1:
            neg_run += 1    # this tick was a sell → increment the sell counter
        # (if label is 0 for some reason, neither counter changes)

        # θ_T = whichever side is larger at this moment
        # This will keep growing as long as one side is consistently winning.
        theta = max(pos_run, neg_run)

        cumm_vol  += volume
        vol_price += price * volume
        collector.append(price)
        num_ticks += 1

        # Warmup: once we have enough ticks, compute the initial expected imbalance.
        # We use the current estimates of E[T] and P[b_t=1] to set a baseline.
        #
        # E0[θ_T] = E0[T] * max{P[b_t=1], 1 - P[b_t=1]}
        # = expected number of ticks × probability of the dominant direction
        if len(bars) == 0 and num_ticks >= expected_num_ticks_init:
            expected_imbalance = max(
                expected_num_ticks * max(expected_p_buy, 1 - expected_p_buy),
                1e-6
            )

        # Tick Run Bar Stopping Definition — page 31
        # T* = arg min_T { θ_T >= E0[T] * max{P[b_t=1], 1 - P[b_t=1]} }
        # Close the bar when the dominant one-sided run exceeds expectations.
        #
        # In plain English: "close the bar as soon as either buys or sells
        # have accumulated to a level larger than we would expect by chance
        # given a random mix with the historical buy proportion."
        if expected_imbalance != 0 and theta >= expected_imbalance:
            open_p  = collector[0]
            high_p  = np.max(collector)
            low_p   = np.min(collector)
            close_p = collector[-1]
            vwap    = vol_price / cumm_vol  # VWAP = Σ(p*v) / Σ(v)

            bars.append((date, i, open_p, low_p, high_p, close_p, vwap))
            bar_lengths.append(num_ticks)

            # Track buy proportion for future EWMA update of P[b_t=1].
            # buy_proportion = (buy ticks) / (total ticks in this bar).
            # Over time, EWMA of these proportions estimates P[b_t=1].
            buy_proportion = pos_run / num_ticks if num_ticks > 0 else 0.5
            buy_tick_proportions.append(buy_proportion)

            # Reset all per-bar accumulators
            pos_run   = 0
            neg_run   = 0
            cumm_vol  = 0
            vol_price = 0
            collector = []
            num_ticks = 0

            # Tick Run Bar Expected Imbalance update — page 31
            # E0[θ_T] = E0[T] * max{P[b_t=1], 1 - P[b_t=1]}
            # Update E0[T] via EWMA of bar lengths (last num_prev_bars values).
            # Update P[b_t=1] via EWMA of buy tick proportions from prior bars.
            # Then recompute the expected maximum run length.
            expected_num_ticks = ewma(bar_lengths, num_prev_bars)
            expected_p_buy     = ewma(buy_tick_proportions, num_prev_bars)
            expected_imbalance = max(
                expected_num_ticks * max(expected_p_buy, 1 - expected_p_buy),
                1e-6
            )

    cols = ['Date', 'Index', 'Open', 'Low', 'High', 'Close', 'Vwap']
    result = pd.DataFrame(bars, columns=cols)
    result['Date'] = pd.to_datetime(result['Date'])
    return result


def volume_run_bars(df, expected_num_ticks_init=10, num_prev_bars=3):
    # Volume/Dollar Run Bars (VRBs/DRBs) — AFML Chapter 2, page 32
    # Extends tick run bars by accumulating volume on each side instead of tick counts.
    #
    # --- Difference from tick run bars ---
    # Tick run bars count +1 per buy tick, +1 per sell tick.
    # Volume run bars add the VOLUME of each trade to the appropriate side counter.
    # A single institutional buy of 10,000 shares counts as 10,000 on the buy side,
    # not as 1. This makes the bar much more sensitive to large-block order flow.
    #
    # --- Stopping rule ---
    # θ_T = max(cumulative buy volume, cumulative sell volume)
    # Close when θ_T >= E[T] * max(P[b_t=1]*E[v|buy], (1-P[b_t=1])*E[v|sell])
    #
    # The threshold now has four adaptive components updated via EWMA:
    #   E[T]          = expected ticks per bar
    #   P[b_t=1]      = expected buy probability
    #   E[v | b_t=1]  = expected volume of a buy trade
    #   E[v | b_t=-1] = expected volume of a sell trade

    pos_run = 0     # accumulated BUY volume in current bar (Σ v_t where b_t=+1)
    neg_run = 0     # accumulated SELL volume in current bar (Σ v_t where b_t=-1)
    cumm_vol  = 0   # total volume in current bar (both sides, for VWAP)
    vol_price = 0   # Σ(price × volume) in current bar (for VWAP)
    collector = []
    bars = []

    bar_lengths = []            # ticks per completed bar
    buy_vol_proportions  = []   # average buy volume per tick in each completed bar
    sell_vol_proportions = []   # average sell volume per tick in each completed bar
    num_ticks = 0

    expected_num_ticks  = expected_num_ticks_init
    expected_p_buy      = 0.5       # initial guess: 50% of trades are buys
    expected_buy_vol    = 0.01      # initial guess for average buy volume per tick
    expected_sell_vol   = 0.01      # initial guess for average sell volume per tick
    expected_imbalance  = 0

    for i, (label, price, date, volume) in enumerate(zip(df['Label'], df['Price'], df['Date'], df['Volume'])):

        # Volume/Dollar Run Bar Length Definition — page 32
        # θ_T = max( Σ b_t*v_t where b_t=1, -Σ b_t*v_t where b_t=-1 )
        # Add this trade's volume to the appropriate side counter.
        # Note: counters are NEVER reset within a bar — they only grow.
        if label == 1:
            pos_run += volume   # add buy volume to the buy-side accumulator
        elif label == -1:
            neg_run += volume   # add sell volume to the sell-side accumulator

        # θ_T = the larger of the two one-sided volume totals
        theta = max(pos_run, neg_run)

        cumm_vol  += volume
        vol_price += price * volume
        collector.append(price)
        num_ticks += 1

        # Warmup: initialize the expected threshold before any bars have closed.
        # E0[θ_T] = E0[T] * max{ P[b_t=1]*E0[v_t|b_t=1],
        #                        (1-P[b_t=1])*E0[v_t|b_t=-1] }
        # = expected ticks × expected volume of the dominant side per tick
        if len(bars) == 0 and num_ticks >= expected_num_ticks_init:
            expected_imbalance = max(
                expected_num_ticks * max(
                    expected_p_buy * expected_buy_vol,          # buy side contribution
                    (1 - expected_p_buy) * expected_sell_vol    # sell side contribution
                ),
                1e-6
            )

        # Volume/Dollar Run Bar Stopping Rule — page 32
        # T* = arg min_T { θ_T >= E0[T] * max{P[b_t=1]*E0[v_t|b_t=1],
        #                                      (1-P[b_t=1])*E0[v_t|b_t=-1]} }
        # Close the bar when the dominant volume run exceeds its expected level.
        #
        # During quiet markets: average volumes are small → threshold is small
        #   → bars close quickly (detecting even small order flow imbalances).
        # During volatile markets: large trades inflate the threshold
        #   → each bar absorbs more volume before closing.
        if expected_imbalance != 0 and theta >= expected_imbalance:
            open_p  = collector[0]
            high_p  = np.max(collector)
            low_p   = np.min(collector)
            close_p = collector[-1]
            vwap    = vol_price / cumm_vol  # VWAP = Σ(p*v) / Σ(v)

            bars.append((date, i, open_p, low_p, high_p, close_p, vwap))
            bar_lengths.append(num_ticks)

            # Track average buy/sell volume per tick in this bar.
            # These ratios feed the EWMA updates of E[v|buy] and E[v|sell].
            # If a bar had 5 ticks and 200 units of buy volume → avg buy vol = 40/tick.
            buy_vol_proportions.append(pos_run / num_ticks  if num_ticks > 0 else expected_buy_vol)
            sell_vol_proportions.append(neg_run / num_ticks if num_ticks > 0 else expected_sell_vol)

            # Reset all per-bar accumulators
            pos_run   = 0
            neg_run   = 0
            cumm_vol  = 0
            vol_price = 0
            collector = []
            num_ticks = 0

            # Volume/Dollar Run Bar Expected Imbalance update — page 32
            # E0[θ_T] = E0[T] * max{P[b_t=1]*E0[v_t|b_t=1], (1-P[b_t=1])*E0[v_t|b_t=-1]}
            # Update all four adaptive components via EWMA of prior bar values:
            #   expected_num_ticks: EWMA of bar lengths
            #   expected_p_buy:     EWMA of buy/(buy+sell) volume fractions
            #   expected_buy_vol:   EWMA of average buy volume per tick
            #   expected_sell_vol:  EWMA of average sell volume per tick
            expected_num_ticks = ewma(bar_lengths, num_prev_bars)

            # Recompute P[b_t=1] as the volume-weighted buy fraction across recent bars.
            # b / (b+s) = fraction of volume that was buying in each past bar.
            expected_p_buy = ewma(
                [b / (b + s) if (b + s) > 0 else 0.5
                 for b, s in zip(buy_vol_proportions, sell_vol_proportions)],
                num_prev_bars
            )

            expected_buy_vol  = ewma(buy_vol_proportions,  num_prev_bars)
            expected_sell_vol = ewma(sell_vol_proportions, num_prev_bars)

            expected_imbalance = max(
                expected_num_ticks * max(
                    expected_p_buy * expected_buy_vol,
                    (1 - expected_p_buy) * expected_sell_vol
                ),
                1e-6
            )

    cols = ['Date', 'Index', 'Open', 'Low', 'High', 'Close', 'Vwap']
    result = pd.DataFrame(bars, columns=cols)
    result['Date'] = pd.to_datetime(result['Date'])
    return result
