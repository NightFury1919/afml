// triple_barrier.ts — mirror of ch03/labeling/triple_barrier.py
// AFML Chapter 3, Sections 3.3-3.5, pages 44-50
//
// 📁 C:\ws\AFML\
// └── ch03_ts\
//     └── labeling_ts\
//         └── triple_barrier.ts   ← goes here

// ---------------------------------------------------------------------------
// Types
// ---------------------------------------------------------------------------

export interface ClosePrice {
  date: string;   // ISO date string, e.g. "2026-03-01"
  price: number;
}

/**
 * One event row — mirrors a row of the events DataFrame in Python.
 * t1   = vertical barrier timestamp (null if none)
 * trgt = barrier width (daily volatility at entry)
 * side = trade direction: +1 (long) or -1 (short)
 */
export interface EventRow {
  date: string;         // entry date
  t1: string | null;   // vertical barrier date (null = no time limit)
  trgt: number;        // barrier width
  side: number;        // +1 long, -1 short
}

/**
 * Output of applyPtSlOnT1 — first touch timestamps for each barrier.
 * null means that barrier was never touched before the vertical barrier.
 */
export interface BarrierTouches {
  date: string;
  sl: string | null;  // first stop-loss touch date
  pt: string | null;  // first profit-target touch date
  t1: string | null;  // vertical barrier date (passed through)
}

/**
 * Output of getBins — final label and actual return for each event.
 */
export interface BinLabel {
  date: string;
  ret: number;          // actual return achieved
  bin: 1 | 0 | -1;     // label: which barrier was hit
}

// ---------------------------------------------------------------------------
// Helpers
// ---------------------------------------------------------------------------

/**
 * Binary search: find the index of the first date in sortedDates >= target.
 * Mirrors pandas index.searchsorted().
 */
function searchSorted(sortedDates: string[], target: string): number {
  let lo = 0, hi = sortedDates.length;
  while (lo < hi) {
    const mid = (lo + hi) >> 1;
    if (sortedDates[mid] < target) lo = mid + 1;
    else hi = mid;
  }
  return lo;
}

/**
 * Subtract one calendar day from an ISO date string.
 * Mirrors pd.Timedelta(days=1).
 */
function subtractOneDay(dateStr: string): string {
  const d = new Date(dateStr);
  d.setDate(d.getDate() - 1);
  return d.toISOString().slice(0, 10);
}

/**
 * Add numDays calendar days to an ISO date string.
 * Mirrors pd.Timedelta(days=numDays).
 */
function addDays(dateStr: string, numDays: number): string {
  const d = new Date(dateStr);
  d.setDate(d.getDate() + numDays);
  return d.toISOString().slice(0, 10);
}

// ---------------------------------------------------------------------------
// getDailyVol
// ---------------------------------------------------------------------------

/**
 * Daily Volatility Estimates — AFML Chapter 3, Snippet 3.1, page 44.
 *
 * Computes the exponentially weighted moving standard deviation of daily returns.
 * Used to set the width of horizontal barriers dynamically — so profit targets
 * and stop losses scale with current market volatility.
 *
 * Formula:
 *   r_t = close_t / close_{t-1 calendar day} − 1
 *   vol_t = EWMA std of r_t with span = span0
 *
 * @param close - array of ClosePrice in chronological order
 * @param span0 - EWMA span for volatility (default 100 bars)
 * @returns array of { date, vol } aligned to close
 */
export function getDailyVol(
  close: ClosePrice[],
  span0 = 100
): { date: string; vol: number }[] {
  const dates  = close.map(c => c.date);
  const prices = close.map(c => c.price);
  const n      = close.length;

  // For each bar, find the index of the bar whose date is closest to
  // (this bar's date − 1 calendar day). Mirrors index.searchsorted(index − 1 day).
  const returns: { idx: number; r: number }[] = [];

  for (let i = 0; i < n; i++) {
    const prevDateTarget = subtractOneDay(dates[i]);
    const prevIdx        = searchSorted(dates, prevDateTarget) - 1;
    if (prevIdx < 0) continue; // no previous bar available
    returns.push({ idx: i, r: prices[i] / prices[prevIdx] - 1 });
  }

  if (returns.length === 0) return [];

  // EWMA standard deviation with span = span0
  // alpha = 2 / (span0 + 1)
  // Mirrors pandas .ewm(span=span0).std()
  const alpha   = 2 / (span0 + 1);
  const results: { date: string; vol: number }[] = [];

  let ewmaMean = 0;
  let ewmaVar  = 0;
  let first    = true;

  for (const { idx, r } of returns) {
    if (first) {
      ewmaMean = r;
      ewmaVar  = 0;
      first    = false;
    } else {
      const prevMean = ewmaMean;
      ewmaMean = alpha * r + (1 - alpha) * ewmaMean;
      ewmaVar  = (1 - alpha) * (ewmaVar + alpha * (r - prevMean) ** 2);
    }
    results.push({ date: dates[idx], vol: Math.sqrt(ewmaVar) });
  }

  return results;
}

// ---------------------------------------------------------------------------
// applyPtSlOnT1
// ---------------------------------------------------------------------------

/**
 * Apply Profit-Taking and Stop-Loss Barriers — AFML Chapter 3, Snippet 3.2.
 *
 * For each event, walks forward through price bars between entry and the
 * vertical barrier, checking whether the upper (pt) or lower (sl) barrier
 * was crossed first. Returns the timestamp of the first touch for each.
 *
 * @param close  - array of ClosePrice in chronological order
 * @param events - array of EventRow (entry date, t1, trgt, side)
 * @param ptSl   - [pt_multiplier, sl_multiplier]
 * @returns array of BarrierTouches, one per event
 */
export function applyPtSlOnT1(
  close: ClosePrice[],
  events: EventRow[],
  ptSl: [number, number]
): BarrierTouches[] {
  const dates  = close.map(c => c.date);
  const prices = close.map(c => c.price);
  const lastDate = dates[dates.length - 1];

  const results: BarrierTouches[] = [];

  for (const event of events) {
    const { date: loc, t1, trgt, side } = event;

    // Barrier widths
    const ptLevel = ptSl[0] > 0 ? ptSl[0] * trgt : null;  // upper barrier
    const slLevel = ptSl[1] > 0 ? -ptSl[1] * trgt : null; // lower barrier

    // Walk from entry to vertical barrier (or end of series if no t1)
    const endDate  = t1 ?? lastDate;
    const startIdx = searchSorted(dates, loc);
    const endIdx   = Math.min(searchSorted(dates, endDate), dates.length - 1);

    const entryPrice = prices[startIdx];
    if (entryPrice === 0) {
      results.push({ date: loc, sl: null, pt: null, t1 });
      continue;
    }

    let slTouch: string | null = null;
    let ptTouch: string | null = null;

    for (let i = startIdx; i <= endIdx; i++) {
      // Return relative to entry price, scaled by trade direction
      // Mirrors: df0 = (close[loc:t1] / close[loc] - 1) * side
      const r = (prices[i] / entryPrice - 1) * side;

      if (slLevel !== null && slTouch === null && r < slLevel) {
        slTouch = dates[i]; // first time return drops below stop-loss
      }
      if (ptLevel !== null && ptTouch === null && r > ptLevel) {
        ptTouch = dates[i]; // first time return rises above profit target
      }

      // Stop early if both barriers hit
      if (slTouch !== null && ptTouch !== null) break;
    }

    results.push({ date: loc, sl: slTouch, pt: ptTouch, t1 });
  }

  return results;
}

// ---------------------------------------------------------------------------
// addVerticalBarrier
// ---------------------------------------------------------------------------

/**
 * Add Vertical Barrier — AFML Chapter 3, Snippet 3.4.
 *
 * For each event date, finds the timestamp of the bar that falls
 * approximately numDays later. This is the maximum holding period —
 * if neither horizontal barrier is hit by this date, the trade closes.
 *
 * @param close    - array of ClosePrice in chronological order
 * @param tEvents  - array of event date strings (from CUSUM filter)
 * @param numDays  - maximum holding period in calendar days
 * @returns array of { date: eventDate, t1: verticalBarrierDate }
 */
export function addVerticalBarrier(
  close: ClosePrice[],
  tEvents: string[],
  numDays: number
): { date: string; t1: string }[] {
  const dates = close.map(c => c.date);
  const n     = close.length;

  const result: { date: string; t1: string }[] = [];

  for (const eventDate of tEvents) {
    const targetDate = addDays(eventDate, numDays);
    const idx        = searchSorted(dates, targetDate);
    if (idx >= n) continue; // target date is beyond the end of data
    result.push({ date: eventDate, t1: dates[idx] });
  }

  return result;
}

// ---------------------------------------------------------------------------
// getEvents
// ---------------------------------------------------------------------------

/**
 * Get Time of First Touch — AFML Chapter 3, Snippet 3.3.
 *
 * Main orchestrator. Sets up three barriers for each event and finds
 * which one is touched first.
 *
 * @param close    - array of ClosePrice in chronological order
 * @param tEvents  - entry dates (from CUSUM filter)
 * @param ptSl     - [pt_multiplier, sl_multiplier]
 * @param trgt     - array of { date, vol } (daily volatility per event)
 * @param minRet   - minimum target return to include an event
 * @param t1       - vertical barrier dates (null = no time limit)
 * @returns array of EventRow with t1 = first touch timestamp
 */
export function getEvents(
  close: ClosePrice[],
  tEvents: string[],
  ptSl: [number, number],
  trgt: { date: string; vol: number }[],
  minRet: number,
  t1: { date: string; t1: string }[] | null = null
): EventRow[] {
  // Build a map of date → vol for fast lookup
  const trgtMap = new Map<string, number>(trgt.map(t => [t.date, t.vol]));

  // Build a map of date → t1 for vertical barriers
  const t1Map = new Map<string, string | null>();
  if (t1 !== null) {
    for (const row of t1) t1Map.set(row.date, row.t1);
  }

  // Build events: filter to dates that have a vol above minRet
  const events: EventRow[] = [];

  for (const date of tEvents) {
    // Get volatility at this event date (bfill: use next available if missing)
    // Mirrors: trgt.reindex(t_events, method='bfill')
    let vol: number | undefined;
    const trgtDates = trgt.map(t => t.date);
    const idx = searchSorted(trgtDates, date);
    if (idx < trgt.length) vol = trgt[idx].vol;
    if (vol === undefined || vol <= minRet) continue; // filter out low-vol events

    const verticalBarrier = t1Map.get(date) ?? null;

    events.push({
      date,
      t1:   verticalBarrier,
      trgt: vol,
      side: 1, // always +1 for primary model (meta-labeling handles direction)
    });
  }

  // Find first barrier touch for each event
  const touches = applyPtSlOnT1(close, events, ptSl);

  // For each event, t1 = earliest touch across sl, pt, and vertical barrier
  // Mirrors: df0.dropna(how='all').min(axis=1)
  const result: EventRow[] = [];

  for (let i = 0; i < events.length; i++) {
    const event = events[i];
    const touch = touches[i];

    // Collect all non-null touch dates and take the minimum (earliest)
    const candidates: string[] = [];
    if (touch.sl !== null) candidates.push(touch.sl);
    if (touch.pt !== null) candidates.push(touch.pt);
    if (touch.t1 !== null) candidates.push(touch.t1);

    if (candidates.length === 0) continue; // no barrier touched → skip

    const firstTouch = candidates.reduce((a, b) => (a < b ? a : b));

    result.push({ ...event, t1: firstTouch });
  }

  return result;
}

// ---------------------------------------------------------------------------
// getBins
// ---------------------------------------------------------------------------

/**
 * Labeling for Side and Size — AFML Chapter 3, Snippet 3.5.
 *
 * Assigns final labels (+1, -1, 0) based on actual return achieved
 * between entry and first barrier touch.
 *
 * ret = exit_price / entry_price − 1
 * bin = sign(ret): +1 if positive, -1 if negative, 0 if exactly zero
 *
 * @param events - output of getEvents()
 * @param close  - array of ClosePrice in chronological order
 * @returns array of BinLabel with ret and bin fields
 */
export function getBins(events: EventRow[], close: ClosePrice[]): BinLabel[] {
  const dates  = close.map(c => c.date);
  const prices = close.map(c => c.price);

  const results: BinLabel[] = [];

  for (const event of events) {
    if (event.t1 === null) continue; // no exit date → skip

    // Entry price: first bar at or after entry date
    const entryIdx = Math.min(searchSorted(dates, event.date), dates.length - 1);
    const entryPrice = prices[entryIdx];

    // Exit price: first bar at or after t1
    const exitIdx  = Math.min(searchSorted(dates, event.t1), dates.length - 1);
    const exitPrice = prices[exitIdx];

    if (entryPrice === 0) continue; // guard against division by zero

    const ret = exitPrice / entryPrice - 1;

    // bin = sign(ret): mirrors numpy.sign()
    const bin: 1 | 0 | -1 = ret > 0 ? 1 : ret < 0 ? -1 : 0;

    results.push({ date: event.date, ret, bin });
  }

  return results;
}
