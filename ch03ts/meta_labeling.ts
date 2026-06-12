// meta_labeling.ts — mirror of ch03/labeling/meta_labeling.py
// AFML Chapter 3, Sections 3.6-3.9, pages 50-54
//
// 📁 C:\ws\AFML\
// └── ch03_ts\
//     └── labeling_ts\
//         └── meta_labeling.ts   ← goes here

import { applyPtSlOnT1 } from './triple_barrier';
import type { ClosePrice, EventRow, BinLabel } from './triple_barrier';

// Re-export shared types so callers only need to import from one place
export type { ClosePrice, EventRow };

// ---------------------------------------------------------------------------
// Types
// ---------------------------------------------------------------------------

/**
 * Output of getBinsMeta — final label with meta-labeling applied.
 * bin ∈ {0, 1} when side is provided (was primary model right or wrong?)
 * bin ∈ {-1, 0, 1} when side is absent (standard triple barrier)
 */
export interface MetaBinLabel {
  date: string;
  ret: number;        // actual return (sign-adjusted if side provided)
  bin: 1 | 0 | -1;   // label
}

// ---------------------------------------------------------------------------
// Helpers (shared with triple_barrier.ts)
// ---------------------------------------------------------------------------

/** Binary search: index of first date >= target in sortedDates. */
function searchSorted(sortedDates: string[], target: string): number {
  let lo = 0, hi = sortedDates.length;
  while (lo < hi) {
    const mid = (lo + hi) >> 1;
    if (sortedDates[mid] < target) lo = mid + 1;
    else hi = mid;
  }
  return lo;
}

// ---------------------------------------------------------------------------
// getEventsMeta
// ---------------------------------------------------------------------------

/**
 * Expanded getEvents with Meta-Labeling — AFML Chapter 3, Snippet 3.6, page 51.
 *
 * Extension of getEvents() that adds support for a primary model's side
 * predictions. When side is provided, the barriers become asymmetric —
 * profit target and stop loss are applied relative to the predicted direction.
 *
 * Without side: symmetric barriers, assume long (+1) for all events.
 * With side:    barriers follow the primary model's direction.
 *               Primary long  (+1): upper = profit target, lower = stop loss
 *               Primary short (-1): upper = stop loss, lower = profit target
 *
 * @param close    - array of ClosePrice in chronological order
 * @param tEvents  - candidate entry dates (from CUSUM filter)
 * @param ptSl     - [pt_multiplier, sl_multiplier]
 * @param trgt     - array of { date, vol } from getDailyVol
 * @param minRet   - minimum target return to include an event
 * @param t1       - vertical barrier dates (null = no time limit)
 * @param side     - primary model's predicted side per event date (optional)
 *                   Map<date, +1 | -1>
 * @returns array of EventRow with t1 = first touch timestamp
 */
export function getEventsMeta(
  close: ClosePrice[],
  tEvents: string[],
  ptSl: [number, number],
  trgt: { date: string; vol: number }[],
  minRet: number,
  t1: { date: string; t1: string }[] | null = null,
  side: Map<string, number> | null = null
): EventRow[] {
  // Build vertical barrier map
  const t1Map = new Map<string, string | null>();
  if (t1 !== null) {
    for (const row of t1) t1Map.set(row.date, row.t1);
  }

  // Step 1: Filter trgt to event dates (bfill) and apply minimum return filter
  // Mirrors: trgt.reindex(t_events, method='bfill'); trgt[trgt > min_ret]
  const trgtDates = trgt.map(t => t.date);

  const events: EventRow[] = [];

  for (const date of tEvents) {
    // bfill: find next available vol at or after this date
    const idx = searchSorted(trgtDates, date);
    if (idx >= trgt.length) continue;
    const vol = trgt[idx].vol;
    if (vol <= minRet) continue; // minimum return filter

    const verticalBarrier = t1Map.get(date) ?? null;

    // Step 3: Determine side and barrier multipliers
    // No primary model → assume long (+1), symmetric barriers
    // Primary model provided → use its side, asymmetric barriers allowed
    const tradeSide = side !== null ? (side.get(date) ?? 1) : 1;

    events.push({
      date,
      t1:   verticalBarrier,
      trgt: vol,
      side: tradeSide,
    });
  }

  // Step 5: Find first barrier touch
  // No primary model → symmetric [pt, pt]; with model → [pt, sl] as given
  const ptSl_ = side === null
    ? [ptSl[0], ptSl[0]] as [number, number]  // symmetric
    : ptSl;                                    // asymmetric (primary model)

  const touches = applyPtSlOnT1(close, events, ptSl_);

  // Step 6: Take earliest touch across sl, pt, and vertical barrier
  const result: EventRow[] = [];

  for (let i = 0; i < events.length; i++) {
    const event = events[i];
    const touch = touches[i];

    const candidates: string[] = [];
    if (touch.sl !== null) candidates.push(touch.sl);
    if (touch.pt !== null) candidates.push(touch.pt);
    if (touch.t1 !== null) candidates.push(touch.t1);

    if (candidates.length === 0) continue;

    const firstTouch = candidates.reduce((a, b) => (a < b ? a : b));
    result.push({ ...event, t1: firstTouch });
  }

  return result;
}

// ---------------------------------------------------------------------------
// getBinsMeta
// ---------------------------------------------------------------------------

/**
 * Expanded getBins with Meta-Labeling — AFML Chapter 3, Snippet 3.7, pages 51-52.
 *
 * Assigns final labels with meta-labeling logic:
 *
 * Case 1 — No side (standard triple barrier):
 *   bin = sign(return) ∈ {-1, 0, +1} — direction of price move
 *
 * Case 2 — With side (meta-labeling):
 *   ret_adjusted = return × side  (flip sign for short trades)
 *   bin = 1 if primary model was correct (ret_adjusted > 0)
 *   bin = 0 if primary model was wrong   (ret_adjusted ≤ 0)
 *   bin ∈ {0, 1} — binary classification: right or wrong?
 *
 * Why binary?
 *   Separating direction (primary model) from correctness (meta-model) lets
 *   each model specialize. Binary classification is also easier for ML models.
 *
 * @param events    - output of getEventsMeta()
 * @param close     - array of ClosePrice in chronological order
 * @param hasSide   - true if events include primary model side predictions
 * @returns array of MetaBinLabel
 */
export function getBinsMeta(
  events: EventRow[],
  close: ClosePrice[],
  hasSide = false
): MetaBinLabel[] {
  const dates  = close.map(c => c.date);
  const prices = close.map(c => c.price);

  const results: MetaBinLabel[] = [];

  for (const event of events) {
    if (event.t1 === null) continue; // no exit date → skip

    // Entry price: first bar at or after entry date
    const entryIdx   = Math.min(searchSorted(dates, event.date), dates.length - 1);
    const entryPrice = prices[entryIdx];
    if (entryPrice === 0) continue;

    // Exit price: first bar at or after t1
    const exitIdx   = Math.min(searchSorted(dates, event.t1), dates.length - 1);
    const exitPrice = prices[exitIdx];

    let ret = exitPrice / entryPrice - 1;

    let bin: 1 | 0 | -1;

    if (hasSide) {
      // Meta-labeling case: adjust return by primary model's side
      // Long (+1) and price up → positive → correct
      // Short (-1) and price down → negative × -1 → positive → correct
      ret = ret * event.side;

      // bin = 1 if correct (ret > 0), 0 if wrong (ret ≤ 0)
      // ret === 0 (vertical barrier, neutral) counts as wrong
      bin = ret > 0 ? 1 : 0;
    } else {
      // Standard case: label by direction of price move
      bin = ret > 0 ? 1 : ret < 0 ? -1 : 0;
    }

    results.push({ date: event.date, ret, bin });
  }

  return results;
}

// ---------------------------------------------------------------------------
// dropLabels
// ---------------------------------------------------------------------------

/**
 * Dropping Under-Populated Labels — AFML Chapter 3, Snippet 3.8, page 53.
 *
 * Recursively removes rows with extremely rare labels until either:
 *   - All remaining labels appear in at least minPct of cases, OR
 *   - Only 2 labels remain (can't drop further without losing all signal)
 *
 * Why?
 *   If 95% of labels are +1 and only 5% are -1, a model can hit 95% accuracy
 *   by always predicting +1 — without learning anything useful. Removing
 *   rare labels forces the model to actually learn the harder cases.
 *
 * @param labels  - array of MetaBinLabel (must have bin field)
 * @param minPct  - minimum fraction a label must represent (default 0.05 = 5%)
 * @returns filtered array with rare-label rows removed
 */
export function dropLabels(
  labels: MetaBinLabel[],
  minPct = 0.05
): MetaBinLabel[] {
  let current = [...labels];

  while (true) {
    const total = current.length;
    if (total === 0) break;

    // Count how often each label appears
    // Mirrors: events['bin'].value_counts(normalize=True)
    const counts = new Map<number, number>();
    for (const row of current) {
      counts.set(row.bin, (counts.get(row.bin) ?? 0) + 1);
    }

    // Convert to fractions
    const fractions = new Map<number, number>();
    for (const [label, count] of counts) {
      fractions.set(label, count / total);
    }

    const minFrac  = Math.min(...fractions.values());
    const numLabels = fractions.size;

    // Stop if all labels meet the threshold or only 2 labels remain
    if (minFrac > minPct || numLabels < 3) break;

    // Find and drop the rarest label
    let rarestLabel = -999;
    for (const [label, frac] of fractions) {
      if (frac === minFrac) { rarestLabel = label; break; }
    }

    console.log(
      `Dropping label ${rarestLabel} (${(minFrac * 100).toFixed(1)}% of cases)`
    );

    current = current.filter(row => row.bin !== rarestLabel);
  }

  return current;
}
