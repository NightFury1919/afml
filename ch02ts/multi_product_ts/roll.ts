// roll.ts — mirror of ch02/multi_product/roll.py
// AFML Chapter 2, Section 2.4.3, page 37 (Snippets 2.2 and 2.3)
//
// 📁 C:\ws\AFML\
// └── ch02_ts\
//     └── multi_product_ts\
//         └── roll.ts   ← goes here

// ---------------------------------------------------------------------------
// Types
// ---------------------------------------------------------------------------

/**
 * One row of a futures price series.
 * Mirrors a single row of the pandas DataFrame used in the Python code.
 */
export interface FuturesRow {
  date: string;         // ISO date string, e.g. "1998-03-01"
  Instrument: string;   // contract identifier, e.g. "SP98H", "SP98M"
  Open: number;         // open price for this bar
  Close: number;        // close price for this bar
}

/**
 * One row of the rolled price series output.
 * Extends FuturesRow with adjusted prices and optional return / rPrice.
 */
export interface RolledRow extends FuturesRow {
  Returns?: number;   // daily percentage return (only in nonNegativeRolledPrices)
  rPrices?: number;   // cumulative $1 investment value (only in nonNegativeRolledPrices)
}

// ---------------------------------------------------------------------------
// rollGaps
// ---------------------------------------------------------------------------

/**
 * Roll Gaps — AFML Chapter 2, Section 2.4.3, page 37.
 *
 * Computes the cumulative price gaps introduced at each futures contract roll.
 * When a contract expires and the next one begins, the price may jump
 * artificially. This function returns the cumulative sum of those gaps so
 * they can be subtracted from the raw price series.
 *
 * Why does this matter?
 *   If March S&P closes at 2290 and June opens at 2300, naively concatenating
 *   the prices creates a +10 jump that never happened in the market. Roll gaps
 *   poison returns, moving averages, and volatility estimates.
 *
 * matchEnd (default true) — backward roll:
 *   The MOST RECENT prices are left unchanged (gaps[-1] = 0).
 *   All historical prices are adjusted downward by subtracting cumulative gaps.
 *   Use this for live trading so your latest prices are always "real."
 *
 * matchEnd = false — forward roll:
 *   The OLDEST prices are left unchanged; later prices are adjusted upward.
 *
 * Formula (page 37):
 *   gap_t = open_{new, t} − close_{old, t-1}   at each roll date
 *   gaps  = cumsum(gap_t)
 *
 * @param series   - array of FuturesRow in chronological order
 * @param matchEnd - true = backward (most common), false = forward
 * @returns array of cumulative gap values aligned to series by index
 */
export function rollGaps(series: FuturesRow[], matchEnd = true): number[] {
  const n = series.length;

  // -----------------------------------------------------------------------
  // Step 1: Find roll dates
  // -----------------------------------------------------------------------
  // A roll date is any row where the Instrument name CHANGES from the previous row.
  // roll_dates[0] = index 0 (start of series, no gap — first contract)
  // roll_dates[k] = index where contract k began
  const rollIndices: number[] = [0]; // always include row 0 (first contract)
  for (let i = 1; i < n; i++) {
    if (series[i].Instrument !== series[i - 1].Instrument) {
      rollIndices.push(i); // new contract starts here → roll date
    }
  }

  // -----------------------------------------------------------------------
  // Step 2: Initialise a gap array of zeros (same length as series)
  // -----------------------------------------------------------------------
  const gaps = new Array<number>(n).fill(0);

  // -----------------------------------------------------------------------
  // Steps 3 & 4: Compute the gap at each roll date (skip rollIndices[0])
  // -----------------------------------------------------------------------
  // gap at rollIndices[k] = open of new contract on that date
  //                        − close of old contract the day BEFORE (index - 1)
  for (let k = 1; k < rollIndices.length; k++) {
    const rollIdx  = rollIndices[k];      // row index of the new contract's first day
    const prevIdx  = rollIdx - 1;         // row index of the old contract's last day
    gaps[rollIdx] =
      series[rollIdx].Open -              // new contract open price
      series[prevIdx].Close;              // old contract close price
  }

  // -----------------------------------------------------------------------
  // Step 5: Cumulate the gaps over time (mirrors pandas .cumsum())
  // -----------------------------------------------------------------------
  // Replace each value with the running total of itself and all prior gaps.
  for (let i = 1; i < n; i++) {
    gaps[i] += gaps[i - 1];
  }

  // -----------------------------------------------------------------------
  // Step 6: Apply backward or forward adjustment
  // -----------------------------------------------------------------------
  // matchEnd=true: shift the whole array so gaps[n-1] = 0.
  // The latest price is "real"; all historical prices are shifted accordingly.
  if (matchEnd) {
    const lastGap = gaps[n - 1];
    for (let i = 0; i < n; i++) {
      gaps[i] -= lastGap;
    }
  }

  return gaps;
}

// ---------------------------------------------------------------------------
// getRolledSeries
// ---------------------------------------------------------------------------

/**
 * Rolled Price Series — AFML Chapter 2, Section 2.4.3, page 37 (Snippet 2.2).
 *
 * Applies the roll gap correction to produce a smooth continuous price series.
 * Subtracts cumulative gaps from open and close prices so there are no
 * artificial jumps at contract roll dates.
 *
 * Example:
 *   March closes at 2290. June opens at 2300 → gap = +10.
 *   We subtract 10 from all prices BEFORE the roll date.
 *   Historical prices now line up smoothly with the June contract's level.
 *
 * Does NOT mutate the input array — returns a new array of adjusted rows.
 *
 * @param series   - array of FuturesRow in chronological order
 * @param matchEnd - true = backward adjustment (default), false = forward
 */
export function getRolledSeries(series: FuturesRow[], matchEnd = true): RolledRow[] {
  const gaps = rollGaps(series, matchEnd);

  // Subtract the cumulative gap from both Open and Close on every row.
  // Adjusting both fields preserves the intraday open-to-close spread.
  return series.map((row, i) => ({
    ...row,
    Open:  row.Open  - gaps[i],
    Close: row.Close - gaps[i],
  }));
}

// ---------------------------------------------------------------------------
// nonNegativeRolledPrices
// ---------------------------------------------------------------------------

/**
 * Non-Negative Rolled Prices (rPrices) — AFML Chapter 2, Section 2.4.3, page 37
 * (Snippet 2.3).
 *
 * Converts the rolled price series into a $1 investment series using cumulative
 * returns, guaranteeing the output is always positive.
 *
 * Why can rolled prices go negative?
 *   After backward adjustment, historical prices are shifted down by the total
 *   of all future roll gaps. If those gaps are large enough, some historical
 *   prices become mathematically negative — an artefact, not a real-world event.
 *   (e.g. a commodity in deep contango where each contract is 20 pts more
 *   expensive; after 5 rolls the backward adjustment is −100 pts.)
 *
 * Solution: work with RETURNS, not price levels. Returns are always defined,
 * and compounding them always produces a positive series.
 *
 * Three-step process (page 37):
 *   1. Compute rolled prices (subtract cumulative gaps from raw prices)
 *   2. r_t = rolledClose_t / rawClose_{t-1} − 1
 *      (numerator: adjusted price move; denominator: raw price level — never negative)
 *   3. rPrices_t = Π (1 + r_s) from s=1 to t  ($1 compounded by daily returns)
 *
 * @param series   - array of FuturesRow in chronological order
 * @param matchEnd - true = backward adjustment (default), false = forward
 * @returns array of RolledRow with Returns and rPrices fields added
 */
export function nonNegativeRolledPrices(
  series: FuturesRow[],
  matchEnd = true
): RolledRow[] {
  const n = series.length;

  // -----------------------------------------------------------------------
  // Step 1: Compute rolled prices (adjusted for roll gaps)
  // -----------------------------------------------------------------------
  const gaps   = rollGaps(series, matchEnd);
  const rolled: RolledRow[] = series.map((row, i) => ({
    ...row,
    Open:  row.Open  - gaps[i],
    Close: row.Close - gaps[i],
  }));

  // -----------------------------------------------------------------------
  // Step 2: Compute daily percentage returns
  // -----------------------------------------------------------------------
  // r_t = (rolledClose_t − rolledClose_{t-1}) / rawClose_{t-1}
  //     = rolledClose_t.diff() / rawClose.shift(1)
  //
  // Numerator  : rolled close diff   = the TRUE price change (gap-adjusted)
  // Denominator: raw close (lagged)  = the TRUE price LEVEL (never near zero)
  //
  // Using raw close in the denominator prevents the division from blowing up
  // if the adjusted close ever approaches zero or goes negative.
  for (let i = 0; i < n; i++) {
    if (i === 0) {
      // No previous row — return is undefined (mirrors pandas .diff() → NaN on row 0)
      rolled[i].Returns = NaN;
    } else {
      const rolledCloseDiff = rolled[i].Close - rolled[i - 1].Close; // Δ adjusted close
      const rawClosePrev    = series[i - 1].Close;                   // raw close_{t-1}
      rolled[i].Returns     = rolledCloseDiff / rawClosePrev;
    }
  }

  // -----------------------------------------------------------------------
  // Step 3: Compound returns into a $1 investment series
  // -----------------------------------------------------------------------
  // rPrices_t = Π_{s=1}^{t} (1 + r_s)
  //
  // (1 + r_t).cumprod() from pandas → running product in TypeScript.
  // Starts at 1.0 (the $1 investment) on the first non-NaN return.
  // Always positive as long as no single day loses more than 100%.
  let cumProd = 1.0;
  for (let i = 0; i < n; i++) {
    if (i === 0 || isNaN(rolled[i].Returns!)) {
      rolled[i].rPrices = NaN; // no return yet → no rPrice
    } else {
      cumProd *= (1 + rolled[i].Returns!);
      rolled[i].rPrices = cumProd;
    }
  }

  return rolled;
}
