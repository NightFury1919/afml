import numpy as np
import pandas as pd

def roll_gaps(series, dictio={'Instrument': 'Instrument', 'Open': 'Open', 'Close': 'Close'}, match_end=True):
    # Roll Gaps — AFML Chapter 2, Section 2.4.3, page 37
    # (Part of Snippet 2.2 in the book)
    #
    # Computes the cumulative price gaps introduced at each futures contract roll.
    # When a contract expires and a new one begins, the price may jump artificially.
    # This function identifies those gaps and returns their cumulative sum,
    # which can then be subtracted from the raw price series.
    #
    # --- What is a "roll gap" and why does it matter? ---
    # Futures contracts have expiry dates. When the March S&P 500 future expires,
    # traders switch to the June contract. The June contract may trade at, say,
    # 2300 while the March contract last traded at 2290. If you naively concatenate
    # these prices, you get a sudden +10 point "jump" that never actually happened
    # in the market. This is a ROLL GAP — an artefact of contract switching.
    #
    # Roll gaps poison any downstream analysis:
    #   - Returns computed across a roll date will appear artificially large.
    #   - Moving averages will show a phantom trend.
    #   - Volatility estimates will be inflated.
    #
    # This function computes the size of each roll gap and returns their cumulative
    # sum so you can SUBTRACT them from the raw price series to get a smooth series.
    #
    # --- What does match_end do? ---
    # There are two conventions for adjusting historical prices:
    #
    # match_end=True  (BACKWARD ROLL, most common):
    #   The MOST RECENT prices are left unchanged (they match today's market).
    #   All HISTORICAL prices are adjusted downward by subtracting the cumulative gaps.
    #   This ensures your latest prices are "real" — important for live trading.
    #
    # match_end=False (FORWARD ROLL):
    #   The OLDEST prices are left unchanged.
    #   Later prices are adjusted upward as gaps accumulate forward.
    #   Used less commonly in practice.
    #
    # Formula (page 37):
    #   gap_t = o_{new contract, t} - p_{old contract, t-1}   at each roll date
    #   gaps  = cumsum(gap_t)

    # -----------------------------------------------------------------------
    # Step 1: Find the roll dates
    # -----------------------------------------------------------------------
    # The 'Instrument' column contains the contract identifier (e.g. 'SP98H', 'SP98M').
    # A roll date is any date where the identifier CHANGES compared to the previous row.
    # drop_duplicates(keep='first') returns only the FIRST occurrence of each unique
    # instrument name, which corresponds exactly to the date it became the front contract.
    roll_dates = series[dictio['Instrument']].drop_duplicates(keep='first').index
    # roll_dates[0] = start of series (first contract, no gap)
    # roll_dates[1] = first roll date (switch from contract 1 to contract 2)
    # roll_dates[2] = second roll date, etc.

    # -----------------------------------------------------------------------
    # Step 2: Initialise a gap series of zeros
    # -----------------------------------------------------------------------
    # We want a series with the same index as the input, filled with 0.
    # Only roll dates will have non-zero values.
    gaps = series[dictio['Close']] * 0      # multiply by 0 to get zeros with same index/dtype

    # -----------------------------------------------------------------------
    # Step 3: Find the integer positions of the days just BEFORE each roll
    # -----------------------------------------------------------------------
    # At the roll date, the new contract opens. The gap is:
    #   gap = open of new contract on roll date - close of OLD contract the day BEFORE.
    #
    # We need the integer positions (not dates) so we can index with .iloc[]
    # and access "the row just before this roll date".
    iloc = list(series.index)                       # convert the datetime index to a list
    iloc = [iloc.index(i) - 1 for i in roll_dates] # for each roll date, get its position - 1
    # iloc[k] = the integer row index of the day BEFORE roll_dates[k]

    # -----------------------------------------------------------------------
    # Step 4: Compute the gap at each roll date
    # -----------------------------------------------------------------------
    # gap at roll_dates[k] = open price of new contract on roll_dates[k]
    #                        minus close price of old contract on iloc[k]
    #
    # We skip roll_dates[0] (the very first contract) because there is no
    # "previous contract" to compute a gap from — hence [1:] on both sides.
    #
    # .values strips the index from both series so numpy does element-wise subtraction.
    gaps.loc[roll_dates[1:]] = (
        series[dictio['Open']].loc[roll_dates[1:]].values -          # new contract open
        series[dictio['Close']].iloc[iloc[1:]].values                # old contract close
    )

    # -----------------------------------------------------------------------
    # Step 5: Cumulate the gaps over time
    # -----------------------------------------------------------------------
    # Instead of a single gap at each roll date, we want a RUNNING TOTAL of all
    # gaps up to each date. This is the total artificial price adjustment needed
    # to make the series look as if there had been no contract switches.
    gaps = gaps.cumsum()    # cumsum() replaces each value with the sum of itself and all prior values

    # -----------------------------------------------------------------------
    # Step 6: Apply backward or forward adjustment
    # -----------------------------------------------------------------------
    # match_end=True (backward roll): we want the LATEST prices to be unchanged.
    # The cumulative gap at the last date (gaps.iloc[-1]) represents the total
    # amount by which the series has been shifted. Subtracting this from all
    # values means the last value becomes 0 (no adjustment) and all earlier
    # values are shifted by the appropriate historical amount.
    if match_end:
        gaps -= gaps.iloc[-1]   # shift the whole series so that gaps[-1] = 0

    return gaps


def get_rolled_series(series, dictio={'Instrument': 'Instrument', 'Open': 'Open', 'Close': 'Close'}, match_end=True):
    # Rolled Price Series — AFML Chapter 2, Section 2.4.3, page 37
    # (Snippet 2.2 in the book)
    #
    # Applies the roll gap correction to produce a smooth continuous price series
    # by subtracting the cumulative gaps from open and close prices.
    # The resulting series has no artificial jumps at roll dates.
    #
    # --- How it works ---
    # Once roll_gaps() has computed how much each historical price was
    # artificially inflated/deflated by contract rolls, we simply subtract
    # that amount from the raw prices. The result is a single continuous series
    # that behaves as if the same contract had been trading since the beginning
    # of the sample — no jumps, no gaps.
    #
    # Example:
    #   March contract closes at 2290 on roll date.
    #   June contract opens at 2300 on roll date → gap = +10.
    #   To remove this gap, we subtract 10 from ALL prices BEFORE the roll date.
    #   Historical prices now line up smoothly with the June contract's level.

    # Compute the cumulative roll gaps (see roll_gaps() above for full explanation)
    gaps = roll_gaps(series, dictio=dictio, match_end=match_end)

    # Make a deep copy so we don't modify the original data
    rolled = series.copy(deep=True)

    # Subtract the cumulative gaps from both open and close prices.
    # We adjust both so that any spread between open and close is preserved.
    for fld in [dictio['Open'], dictio['Close']]:
        rolled[fld] -= gaps     # element-wise subtraction aligned by index

    return rolled


def non_negative_rolled_prices(series, dictio={'Instrument': 'Instrument', 'Open': 'Open', 'Close': 'Close'}, match_end=True):
    # Non-Negative Rolled Price Series — AFML Chapter 2, Section 2.4.3, page 37
    # (Snippet 2.3 in the book)
    #
    # Converts the rolled price series into a $1 investment series using
    # cumulative returns, guaranteeing the series is always positive.
    # Raw rolled prices can go negative in contango markets (e.g. natural gas).
    #
    # --- Why can prices go negative? ---
    # After backward roll adjustment, historical prices are shifted down by
    # the total of all future roll gaps. If those gaps are large enough,
    # some historical prices can become NEGATIVE — a mathematical artefact
    # of the adjustment, not a real-world event.
    #
    # Example: A commodity in contango where each contract is 20 points more
    # expensive than the previous one. After 5 rolls, the backward adjustment
    # is -100 points. If the original historical prices were only $80, some
    # will become $80 - $100 = -$20 after adjustment.
    #
    # The solution: instead of working with PRICE LEVELS, convert to RETURNS.
    # Returns are always defined (as long as the denominator is non-zero),
    # and compounding returns always produces a positive series.
    #
    # --- Three-step process from page 37 ---
    # Step 1: Compute rolled prices (subtract cumulative gaps)
    # Step 2: r_t = rolled_close_t / raw_close_{t-1} - 1  (percentage return)
    # Step 3: rPrices = (1 + r).cumprod()  ($1 compounded by daily returns)

    # -----------------------------------------------------------------------
    # Step 1: Get the roll-gap-adjusted price series
    # -----------------------------------------------------------------------
    gaps = roll_gaps(series, dictio=dictio, match_end=match_end)
    rolled = series.copy(deep=True)
    for fld in [dictio['Open'], dictio['Close']]:
        rolled[fld] -= gaps

    # -----------------------------------------------------------------------
    # Step 2: Compute daily percentage returns
    # -----------------------------------------------------------------------
    # r_t = (rolled_close_t - raw_close_{t-1}) / raw_close_{t-1}
    #     = rolled_close_t / raw_close_{t-1} - 1
    #
    # Key insight: we use rolled_close in the NUMERATOR (the price move, after
    # adjustment) but raw_close in the DENOMINATOR (the price LEVEL, unadjusted).
    #
    # Why? The adjustment shifts the price LEVEL but not the price MOVE.
    # Using the raw close in the denominator ensures the percentage return is
    # computed relative to the true market price, not the adjusted (potentially
    # negative) price. This prevents the denominator from ever being near zero
    # or negative.
    #
    # rolled[dictio['Close']].diff()     = change in rolled close price (= true price change)
    # series[dictio['Close']].shift(1)   = yesterday's RAW close (the true denominator)
    rolled['Returns'] = rolled[dictio['Close']].diff() / series[dictio['Close']].shift(1)

    # -----------------------------------------------------------------------
    # Step 3: Compound returns into a $1 investment series
    # -----------------------------------------------------------------------
    # rPrices_t = Π_{s=1}^{t} (1 + r_s)
    #
    # (1 + r_t).cumprod() computes the product of (1 + return) at each step,
    # which is exactly how a $1 investment grows under daily compounding.
    #
    # This is always positive as long as r_t > -1 (i.e. no day loses more than 100%).
    # The series starts at 1.0 (the $1 initial investment) and grows or shrinks
    # as the underlying instruments move, cleanly and without any negative prices.
    rolled['rPrices'] = (1 + rolled['Returns']).cumprod()

    return rolled
