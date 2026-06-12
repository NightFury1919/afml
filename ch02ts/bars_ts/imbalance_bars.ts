// imbalance_bars.ts — mirror of ch02/bars/imbalance_bars.py
// AFML Chapter 2, Section 2.3.2
//
// 📁 C:\ws\AFML\
// └── ch02_ts\
//     └── bars_ts\
//         └── imbalance_bars.ts   ← goes here

import { ewma } from './utils';

// ---------------------------------------------------------------------------
// Types
// ---------------------------------------------------------------------------

export interface LabeledTrade {
  Date: string | Date;
  Price: number;
  Volume: number;
  Label: number; // +1 (buy) or -1 (sell) from tick rule
}

export interface ImbalanceBar {
  Date: string | Date;
  Index: number;
  Open: number;
  Low: number;
  High: number;
  Close: number;
  Vwap?: number; // present in volume bars, absent in tick bars
}

// ---------------------------------------------------------------------------
// tickImbalanceBars
// ---------------------------------------------------------------------------

/**
 * Tick Imbalance Bars (TIBs) — AFML Chapter 2, page 29.
 *
 * Accumulates θ_T = Σ b_t (sum of +1/−1 trade directions).
 * Closes a bar when |θ_T| >= E[T] * |2*P[b=1] − 1|.
 *
 * Why?
 *   When an informed trader is active, they trade repeatedly in one direction,
 *   causing the running sum of buy/sell labels to drift strongly away from zero.
 *   That drift is a signal of informed trading. Bars close when the drift becomes
 *   unexpectedly large — so each bar represents the same AMOUNT OF INFORMATION,
 *   not the same amount of time or volume.
 *
 * Threshold components:
 *   E[T]              = expected number of ticks per bar (EWMA of past bar lengths)
 *   |2*P[b=1] − 1|   = expected imbalance per tick
 *                       (0 if perfectly balanced, 1 if all buys or all sells)
 *   product           = expected total imbalance for a typical bar
 *
 * @param trades               - labeled trades (must have Label field from tickRule)
 * @param expectedNumTicksInit - initial guess for bar length before any bars close
 * @param numPrevBars          - how many past bars to use in EWMA updates
 */
export function tickImbalanceBars(
  trades: LabeledTrade[],
  expectedNumTicksInit = 10,
  numPrevBars = 3
): ImbalanceBar[] {
  const bars: ImbalanceBar[] = [];

  let cumTheta = 0;           // θ_T: running sum of b_t within current bar
  let collector: number[] = []; // prices within the current bar
  let numTicks = 0;           // tick count for the current bar

  const imbalanceArray: number[] = []; // all b_t values ever seen (for EWMA lookback)
  const barLengths: number[]     = []; // number of ticks in each completed bar

  let expectedNumTicks = expectedNumTicksInit;
  let expectedImbalance = 0;

  for (let i = 0; i < trades.length; i++) {
    const { Label: label, Price: price, Date: date } = trades[i];

    // θ_T = Σ b_t — each tick contributes its direction (+1 or -1)
    // For tick bars, volume weight v_t = 1, so imbalance = b_t exactly
    const imbalance = label;
    imbalanceArray.push(imbalance);
    cumTheta += imbalance;

    collector.push(price);
    numTicks++;

    // Warmup: don't close any bars until we have enough ticks to estimate
    // the expected imbalance. Before that, expectedImbalance stays 0.
    if (bars.length === 0 && imbalanceArray.length >= expectedNumTicksInit) {
      expectedImbalance = Math.max(
        ewma(imbalanceArray, expectedNumTicksInit),
        1e-6  // floor prevents threshold = 0 (would close on every tick)
      );
    }

    // Stopping rule — page 29:
    // T* = arg min_T { |θ_T| >= E[T] * |2*P[b=1] − 1| }
    // Close when actual imbalance exceeds the expected imbalance for a bar
    // of this length.
    if (
      expectedImbalance !== 0 &&
      Math.abs(cumTheta) >= expectedNumTicks * Math.abs(expectedImbalance)
    ) {
      bars.push({
        Date:  date,
        Index: i,
        Open:  collector[0],
        High:  Math.max(...collector),
        Low:   Math.min(...collector),
        Close: collector[collector.length - 1],
      });

      barLengths.push(numTicks);

      // Reset accumulators for the next bar
      cumTheta  = 0;
      collector = [];
      numTicks  = 0;

      // Update EWMA estimates after each bar closes:
      //   expectedNumTicks: EWMA of last numPrevBars bar lengths
      //   expectedImbalance: EWMA over last (numPrevBars * E[T]) ticks
      // The wide window ensures we look back across ~numPrevBars worth of bars.
      expectedNumTicks = ewma(barLengths, numPrevBars);
      expectedImbalance = Math.max(
        ewma(
          imbalanceArray,
          Math.max(1, Math.floor(numPrevBars * expectedNumTicks))
        ),
        1e-6
      );
    }
  }

  return bars;
}

// ---------------------------------------------------------------------------
// volumeImbalanceBars
// ---------------------------------------------------------------------------

/**
 * Volume Imbalance Bars (VIBs) — AFML Chapter 2, page 30.
 *
 * Extends tick imbalance bars by weighting each tick by its volume (v_t).
 * θ_T = Σ b_t * v_t  (volume-weighted signed sum).
 * Closes when |θ_T| >= E[T] * |2v+ − E[v_t]|.
 *
 * Why weight by volume?
 *   In tick bars, a 1-share retail trade and a 10,000-share institutional
 *   block both count as ±1. That's unrealistic. Volume weighting means large
 *   trades have proportionally larger impact — closer to how information
 *   actually flows in real markets.
 *
 * Also computes VWAP = Σ(price × volume) / Σ(volume) per bar.
 *
 * @param trades               - labeled trades with Volume field
 * @param expectedNumTicksInit - initial guess for bar length
 * @param numPrevBars          - how many past bars to use in EWMA updates
 */
export function volumeImbalanceBars(
  trades: LabeledTrade[],
  expectedNumTicksInit = 10,
  numPrevBars = 3
): ImbalanceBar[] {
  const bars: ImbalanceBar[] = [];

  let cumTheta  = 0;            // θ_T: running volume-weighted imbalance
  let cummVol   = 0;            // Σ volume in current bar (VWAP denominator)
  let volPrice  = 0;            // Σ(price × volume) in current bar (VWAP numerator)
  let collector: number[] = []; // prices within the current bar
  let numTicks  = 0;

  const imbalanceArray: number[] = []; // all b_t * v_t values ever seen
  const barLengths: number[]     = [];

  let expectedNumTicks  = expectedNumTicksInit;
  let expectedImbalance = 0;

  for (let i = 0; i < trades.length; i++) {
    const { Label: label, Price: price, Volume: volume, Date: date } = trades[i];

    // θ_T = Σ b_t * v_t — signed volume contribution of this tick
    const imbalance = label * volume;
    imbalanceArray.push(imbalance);
    cumTheta += imbalance;

    cummVol  += volume;
    volPrice += price * volume;
    collector.push(price);
    numTicks++;

    // Warmup: same logic as tick imbalance bars
    if (bars.length === 0 && imbalanceArray.length >= expectedNumTicksInit) {
      expectedImbalance = Math.max(
        ewma(imbalanceArray, expectedNumTicksInit),
        1e-6
      );
    }

    // Stopping rule — page 30:
    // T* = arg min_T { |θ_T| >= E[T] * |2v+ − E[v_t]| }
    // Close when volume-weighted imbalance exceeds its expected value.
    if (
      expectedImbalance !== 0 &&
      Math.abs(cumTheta) >= expectedNumTicks * Math.abs(expectedImbalance)
    ) {
      const vwap = volPrice / cummVol; // VWAP = Σ(p*v) / Σ(v)

      bars.push({
        Date:  date,
        Index: i,
        Open:  collector[0],
        High:  Math.max(...collector),
        Low:   Math.min(...collector),
        Close: collector[collector.length - 1],
        Vwap:  vwap,
      });

      barLengths.push(numTicks);

      // Reset all accumulators
      cumTheta  = 0;
      cummVol   = 0;
      volPrice  = 0;
      collector = [];
      numTicks  = 0;

      // Update EWMA estimates — same logic as tick imbalance bars
      expectedNumTicks = ewma(barLengths, numPrevBars);
      expectedImbalance = Math.max(
        ewma(
          imbalanceArray,
          Math.max(1, Math.floor(numPrevBars * expectedNumTicks))
        ),
        1e-6
      );
    }
  }

  return bars;
}
