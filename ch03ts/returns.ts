// returns.ts — mirror of ch03/labeling/returns.py
// AFML Chapter 3, Section 3.2, page 43
//
// 📁 C:\ws\AFML\
// └── ch03_ts\
//     └── labeling_ts\
//         └── returns.ts   ← goes here

// ---------------------------------------------------------------------------
// Types
// ---------------------------------------------------------------------------

/**
 * One entry in a close price series.
 * Mirrors a pandas Series row: date → price.
 */
export interface ClosePrice {
  date: string; // ISO date string, e.g. "2026-03-01"
  price: number;
}

/**
 * One labeled observation.
 * Mirrors one row of the output pd.Series from fixed_time_horizon().
 */
export interface Label {
  date: string;
  label: 1 | 0 | -1;
}

// ---------------------------------------------------------------------------
// fixedTimeHorizon
// ---------------------------------------------------------------------------

/**
 * Fixed-Time Horizon Labeling — AFML Chapter 3, Section 3.2, page 43.
 *
 * Labels each event date as +1, 0, or -1 based on the price return
 * over the next h bars compared to a threshold τ.
 *
 * Why?
 *   ML models need labeled training data. The simplest approach is to look
 *   h bars into the future and check whether price moved enough to be
 *   considered a good or bad entry point.
 *
 * Limitation:
 *   Ignores what happens BETWEEN entry and bar h. A trade that dropped 10%
 *   before recovering to +1% still gets labeled +1. The Triple Barrier Method
 *   (Section 3.3) addresses this.
 *
 * Formula (page 43):
 *   r = P_{t0+h} / P_{t0} − 1
 *   y = −1  if r < −τ    (fell more than threshold)
 *       0   if |r| ≤ τ   (moved less than threshold)
 *       +1  if r > τ     (rose more than threshold)
 *
 * @param close     - array of ClosePrice in chronological order
 * @param events    - array of date strings to label (e.g. CUSUM filter output)
 * @param h         - number of bars to look forward
 * @param threshold - τ: return threshold as a decimal (e.g. 0.01 = 1%)
 * @returns array of Label objects for each successfully labeled event date
 */
export function fixedTimeHorizon(
  close: ClosePrice[],
  events: string[],
  h: number,
  threshold: number
): Label[] {
  // Build a map of date → integer index for O(1) lookups.
  // Mirrors: close_index = list(close.index)
  const dateToIndex = new Map<string, number>();
  for (let i = 0; i < close.length; i++) {
    dateToIndex.set(close[i].date, i);
  }

  const labels: Label[] = [];

  for (const eventDate of events) {

    // --- Find the integer position of this event in the price series ---
    // t_{i,0}: the bar at the event date (entry point).
    // Skip events that don't have a matching bar in the price series.
    const t0 = dateToIndex.get(eventDate);
    if (t0 === undefined) continue; // no matching bar → skip

    // --- Check that h bars ahead exists ---
    // If the event is too close to the end of the data, skip it.
    const t1 = t0 + h;
    if (t1 >= close.length) continue; // not enough future data → skip

    // --- Compute the price return over h bars ---
    // r = P_{t0+h} / P_{t0} − 1
    const p0 = close[t0].price;  // entry price
    const p1 = close[t1].price;  // exit price h bars later
    const r  = p1 / p0 - 1;     // percentage return

    // --- Assign label based on threshold τ (page 43) ---
    let label: 1 | 0 | -1;
    if (r > threshold)       label =  1;
    else if (r < -threshold) label = -1;
    else                     label =  0;

    labels.push({ date: eventDate, label });
  }

  return labels;
}
