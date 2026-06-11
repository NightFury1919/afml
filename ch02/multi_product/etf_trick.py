import numpy as np
import pandas as pd

def etf_trick(open_prices, close_prices, alloc_weights, point_values, dividends, rebalance_dates, trans_costs=None, volumes=None):
    # ETF Trick — AFML Chapter 2, Section 2.4.1, pages 33-34
    #
    # Converts a basket of multiple instruments into a single continuous
    # $1 investment series (K_t), eliminating issues with:
    #   - Changing weights causing artificial convergence
    #   - Negative spread values
    #   - Roll gaps in futures contracts
    #
    # --- Why do we need this? ---
    # Suppose you want to backtest a strategy across multiple futures contracts
    # simultaneously (e.g. SP98H, SP98M, SP98U). You face several problems:
    #
    # Problem 1 — Units: Each contract has a different notional value and
    #   point value (e.g. S&P futures = $250 per index point). You can't
    #   directly add prices across instruments with different units.
    #
    # Problem 2 — Roll gaps: When a futures contract expires and you switch
    #   to the next one, there is typically a price GAP (the new contract
    #   trades at a different price). Naively concatenating prices creates
    #   a false "jump" in the series.
    #
    # Problem 3 — Negative prices: Some futures (e.g. natural gas in contango)
    #   can produce negative values when you naively subtract roll gaps.
    #
    # The ETF Trick solves all three by tracking the VALUE (in dollars) of
    # a fixed $1 initial investment in the basket, evolving day by day through
    # actual P&L rather than trying to stitch raw prices together.
    #
    # --- Inputs ---
    # open_prices     : DataFrame (T × I) — open prices for each instrument each day
    # close_prices    : DataFrame (T × I) — close prices for each instrument each day
    # alloc_weights   : DataFrame (T × I) — target portfolio weight ω_{i,t} for each instrument
    #                   Weights can be positive (long) or negative (short).
    # point_values    : DataFrame (T × I) — φ_{i,t}: dollar value of one price point
    #                   (e.g. $250 per point for S&P 500 futures)
    # dividends       : DataFrame (T × I) — d_{i,t}: dividends/coupons paid on date t
    # rebalance_dates : list of dates when the portfolio is rebalanced to target weights
    # trans_costs     : Series (I,) — τ_i: transaction cost per dollar of notional
    #                   (e.g. 1e-4 = 1 basis point)
    # volumes         : DataFrame (T × I) — actual traded volume for each instrument each day
    #                   Used to compute tradeable volume v_t. If None, close_prices is used
    #                   as a proxy (not recommended).
    #
    # --- Output ---
    # DataFrame with columns:
    #   K              — portfolio value (starts at $1, grows/shrinks with P&L)
    #   rebalance_cost — transaction cost of rebalancing on that date
    #   bid_ask_cost   — cost of trading one unit of the virtual ETF (computed every bar)
    #   volume         — tradeable size limited by least liquid instrument

    T, I = close_prices.shape   # T = number of time bars, I = number of instruments
    bars = close_prices.index   # the datetime index (one entry per trading day)

    # --- Initialize output series ---
    K = pd.Series(index=bars, dtype=float)                              # K_t: portfolio value
    h = pd.DataFrame(0.0, index=bars, columns=close_prices.columns)    # h_i,t: holdings (units)
    rebalance_cost = pd.Series(0.0, index=bars)                         # c_t
    bid_ask_cost   = pd.Series(0.0, index=bars)                         # c~_t
    volume         = pd.Series(0.0, index=bars)                         # v_t

    # Default to zero transaction costs if not provided
    if trans_costs is None:
        trans_costs = pd.Series(0.0, index=close_prices.columns)

    # Use real volume data if provided, otherwise fall back to close prices as proxy
    volume_data = volumes if volumes is not None else close_prices

    K.iloc[0] = 1.0     # K_0 = $1 — we start with a $1 investment on day 0

    for t_idx in range(1, T):
        t      = bars[t_idx]        # current date (today)
        t_prev = bars[t_idx - 1]    # previous date (yesterday)

        # -----------------------------------------------------------------------
        # Step 1: Compute holdings h_i,t (page 34)
        # -----------------------------------------------------------------------
        # On a REBALANCE DATE (t-1 ∈ B): recalculate how many units of each
        # instrument to hold so that the allocation matches target weights ω.
        # We enter positions at tomorrow's OPEN price (o_i,t+1 = open_prices.loc[t]).
        #
        # Formula: h_i,t = (ω_i,t × K_t) / (o_i,t+1 × φ_i,t × Σ|ω_i,t|)
        #
        # Breaking this down:
        #   ω_i,t × K_t     = the dollar amount we want to allocate to instrument i
        #   o_i,t+1 × φ_i,t = the dollar value of ONE unit of instrument i at open
        #   Σ|ω_i,t|        = sum of absolute weights (normalises for leverage)
        #
        # Example: K = $1000, ω = 0.33, open = 2000, φ = $250, Σ|ω| = 1.0
        #   h = (0.33 × 1000) / (2000 × 250 × 1) ≈ 0.00066 contracts
        #
        # On a non-rebalance date: carry forward yesterday's holdings unchanged.
        if t_prev in rebalance_dates:
            w     = alloc_weights.loc[t_prev]   # target weights ω_i as of yesterday
            denom = w.abs().sum()               # Σ|ω_i|: de-levers to keep exposure bounded
            for i in close_prices.columns:
                h.loc[t, i] = (w[i] * K.iloc[t_idx - 1]) / (
                    open_prices.loc[t, i] * point_values.loc[t_prev, i] * denom
                )
        else:
            h.loc[t] = h.loc[t_prev]   # no rebalance → holdings unchanged

        # -----------------------------------------------------------------------
        # Step 2: Compute price change δ_i,t (page 34)
        # -----------------------------------------------------------------------
        # We need to know how much the price MOVED from where we entered.
        #
        # On the first bar after a rebalance: we entered at the OPEN price,
        #   so the P&L from open to close of that day is (close - open).
        #   δ = p_i,t - o_i,t
        #
        # On all other bars: we already held positions from the previous close,
        #   so the P&L is simply yesterday's close to today's close.
        #   δ = p_i,t - p_i,t-1  (day-to-day change)
        if t_prev in rebalance_dates:
            delta = close_prices.loc[t] - open_prices.loc[t]           # intraday change
        else:
            delta = close_prices.loc[t] - close_prices.loc[t_prev]     # overnight change

        # -----------------------------------------------------------------------
        # Step 3: Update portfolio value K_t (page 34)
        # -----------------------------------------------------------------------
        # K_t = K_{t-1} + Σ_i h_{i,t-1} × φ_{i,t} × (δ_{i,t} + d_{i,t})
        #
        # For each instrument i:
        #   h_{i,t-1}           = how many units we held (from yesterday)
        #   φ_{i,t}             = dollar value of 1 unit (e.g. $250 per S&P point)
        #   δ_{i,t} + d_{i,t}  = price move + any dividend received today
        #   product             = P&L in dollars from instrument i today
        #
        # Summing across all instruments gives total P&L, which is added to yesterday's K.
        pnl = (h.loc[t_prev] * point_values.loc[t] * (delta + dividends.loc[t])).sum()
        K.iloc[t_idx] = K.iloc[t_idx - 1] + pnl

        # -----------------------------------------------------------------------
        # Step 4: Transaction costs (page 34)
        # -----------------------------------------------------------------------
        if t_prev in rebalance_dates:
            # Rebalance cost c_t: total cost of CLOSING old positions AND OPENING new ones.
            # Only charged on rebalance days since that's when trades actually happen.
            # c_t = Σ_i (|h_{i,t-1}|*p_{i,t} + |h_{i,t}|*o_{i,t+1}) * φ_{i,t} * τ_i
            rebalance_cost.loc[t] = (
                (h.loc[t_prev].abs() * close_prices.loc[t] +
                 h.loc[t].abs()      * open_prices.loc[t]) *
                point_values.loc[t] * trans_costs
            ).sum()

        # Bid-ask cost c~_t: computed EVERY bar, not just on rebalance days.
        # This represents the implicit cost of holding positions — if you had to
        # exit right now, what would you lose to the bid-ask spread?
        # c~_t = Σ_i |h_{i,t-1}| * p_{i,t} * φ_{i,t} * τ_i
        bid_ask_cost.loc[t] = (
            h.loc[t_prev].abs() * close_prices.loc[t] *
            point_values.loc[t] * trans_costs
        ).sum()

        # -----------------------------------------------------------------------
        # Step 5: Tradeable volume v_t (page 34)
        # -----------------------------------------------------------------------
        # v_t = min_i { v_{i,t} / |h_{i,t-1}| }
        #
        # How many units of the ETF can we actually trade, given the liquidity
        # of each underlying instrument? We use real volume data if provided,
        # otherwise fall back to close prices as a rough proxy.
        h_prev_abs = h.loc[t_prev].abs()
        if (h_prev_abs > 0).any():      # avoid division by zero if holding nothing
            volume.loc[t] = (volume_data.loc[t] / h_prev_abs.replace(0, np.nan)).min()

    result = pd.DataFrame({
        'K':               K,               # portfolio value ($1 start)
        'rebalance_cost':  rebalance_cost,  # transaction cost on rebalance days only
        'bid_ask_cost':    bid_ask_cost,    # spread cost every bar
        'volume':          volume           # tradeable units limited by least liquid leg
    })

    return result
