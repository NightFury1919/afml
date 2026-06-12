// filters.ts — mirror of ch02/bars/filters.py
// AFML Chapter 2, Section 2.5.2, page 39 (Snippet 2.4)

// ---------------------------------------------------------------------------
// Types
// ---------------------------------------------------------------------------

export interface PriceBar {
  date: string | Date; // date/timestamp of this bar
  Price: number;
}

// ---------------------------------------------------------------------------
// cusumFilter
// ---------------------------------------------------------------------------

/**
 * Symmetric CUSUM Filter — AFML Chapter 2, Section 2.5.2, page 39.
 *
 * A quality-control method that detects when a price series has drifted
 * significantly in either direction from a reset level of zero.
 * Fires an event when cumulative drift exceeds threshold h, then resets.
 *
 * Why use it?
 *   Many strategies generate signals at every bar, but if price hasn't moved
 *   meaningfully, acting on those signals wastes transaction costs and
 *   statistical power. The CUSUM filter is a "pre-filter" — it outputs only
 *   dates where price has drifted by at least h since the LAST time it fired.
 *
 *   Unlike a Bollinger Band (which can fire on many consecutive bars during a
 *   slow trend), the CUSUM filter RESETS after firing and must accumulate a
 *   full h of drift before firing again.
 *
 * Visual intuition:
 *   Imagine a ball on a rubber band anchored at 0. Every uptick stretches it
 *   upward (S+ grows); every downtick compresses it (S- grows negative).
 *   When the stretch exceeds h, the band SNAPS (event fires, reset to 0),
 *   and the process starts fresh.
 *
 * Formulas (page 39):
 *   S+_t = max{0, S+_{t-1} + Δp_t},   S+_0 = 0
 *   S-_t = min{0, S-_{t-1} + Δp_t},   S-_0 = 0
 *
 *   Event fires at t if S+_t >= h  OR  |S-_t| >= h
 *   After firing: reset the accumulator that triggered to 0.
 *
 * @param bars - array of { date, Price } objects in chronological order
 * @param h    - threshold; event fires when cumulative drift exceeds this
 * @returns    array of dates where a CUSUM event fired
 */
export function cusumFilter(bars: PriceBar[], h: number): (string | Date)[] {
  const events: (string | Date)[] = [];

  let sPos = 0; // S+_t — tracks upward drift since last reset
  let sNeg = 0; // S-_t — tracks downward drift since last reset

  for (let i = 1; i < bars.length; i++) {
    // Δp_t = price change since previous bar (mirrors pandas .diff())
    // i === 0 is skipped (no previous price), same as pd.isna(delta) in Python
    const delta = bars[i].Price - bars[i - 1].Price;

    // S+_t = max{0, S+_{t-1} + Δp_t}
    //   Grows when price rises; floored at 0 so it never goes negative.
    sPos = Math.max(0, sPos + delta);

    // S-_t = min{0, S-_{t-1} + Δp_t}
    //   Grows more negative when price falls; ceilinged at 0 so it never goes positive.
    sNeg = Math.min(0, sNeg + delta);

    if (sPos >= h) {
      // Upward drift exceeded threshold — fire event, reset S+
      sPos = 0;
      events.push(bars[i].date);
    } else if (Math.abs(sNeg) >= h) {
      // Downward drift exceeded threshold — fire event, reset S-
      // elif (not if) ensures only ONE event fires per bar, never both.
      sNeg = 0;
      events.push(bars[i].date);
    }
  }

  // Return the event dates — equivalent to pd.DatetimeIndex(events) in Python.
  // Callers can use these to slice other time-indexed data structures.
  return events;
}
