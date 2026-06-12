// run_bars.ts — mirror of ch02/bars/run_bars.py
// AFML Chapter 2, Section 2.3.2
//
// 📁 C:\ws\AFML\
// └── ch02_ts\
//     └── bars_ts\
//         └── run_bars.ts   ← goes here

import { ewma } from './utils';

// ---------------------------------------------------------------------------
// Types (re-use from imbalance_bars if preferred, duplicated here for clarity)
// ---------------------------------------------------------------------------

export interface LabeledTrade {
  Date: string | Date;
  Price: number;
  Volume: number;
  Label: number; // +1 (buy) or -1 (sell) from tick rule
}

export interface RunBar {
  Date: string | Date;
  Index: number;
  Open: number;
  Low: number;
  High: number;
  Close: number;
  Vwap: number;
}

// ---------------------------------------------------------------------------
// tickRunBars
// ---------------------------------------------------------------------------

/**
 * Tick Run Bars (TRBs) — AFML Chapter 2, page 31.
 *
 * Tracks the maximum of cumulative buy ticks vs cumulative sell ticks.
 * θ_T = max(Σ buy ticks, Σ sell ticks)
 * Closes when θ_T >= E[T] * max(P[b=1], 1 − P[b=1])
 *
 * How is this different from imbalance bars?
 *   Imbalance: θ_T = buys MINUS sells  (they cancel each other out)
 *   Run:       θ_T = max(buys, sells)  (they NEVER cancel — always growing)
 *
 *   Example: 60 buys, 40 sells
 *   Imbalance: θ_T = 60 − 40 = 20
 *   Run:       θ_T = max(60, 40) = 60
 *
 *   Run bars are more sensitive to one-sided "sweeps" of the order book
 *   by large institutional traders.
 *
 * @param trades               - labeled trades (Label field required)
 * @param expectedNumTicksInit - initial guess for bar length
 * @param numPrevBars          - how many past bars to use in EWMA updates
 */
export function tickRunBars(
  trades: LabeledTrade[],
  expectedNumTicksInit = 10,
  numPrevBars = 3
): RunBar[] {
  const bars: RunBar[] = [];

  let posRun = 0;             // cumulative buy tick count within current bar
  let negRun = 0;             // cumulative sell tick count within current bar
  let cummVol  = 0;
  let volPrice = 0;
  let collector: number[] = [];
  let numTicks = 0;

  const barLengths: number[]         = [];
  const buyTickProportions: number[] = []; // fraction of buy ticks per completed bar

  let expectedNumTicks  = expectedNumTicksInit;
  let expectedPBuy      = 0.5;  // initial guess: 50% buys
  let expectedImbalance = 0;

  for (let i = 0; i < trades.length; i++) {
    const { Label: label, Price: price, Volume: volume, Date: date } = trades[i];

    // θ_T = max(Σ buy ticks, Σ sell ticks) — page 31
    // Accumulate buy and sell ticks independently; never reset mid-bar.
    if (label === 1)       posRun++;  // buy tick
    else if (label === -1) negRun++;  // sell tick

    const theta = Math.max(posRun, negRun); // dominant one-sided run so far

    cummVol  += volume;
    volPrice += price * volume;
    collector.push(price);
    numTicks++;

    // Warmup: compute initial expected imbalance once we have enough ticks.
    // E0[θ_T] = E[T] * max(P[b=1], 1 − P[b=1])
    if (bars.length === 0 && numTicks >= expectedNumTicksInit) {
      expectedImbalance = Math.max(
        expectedNumTicks * Math.max(expectedPBuy, 1 - expectedPBuy),
        1e-6
      );
    }

    // Stopping rule — page 31:
    // T* = arg min_T { θ_T >= E[T] * max(P[b=1], 1 − P[b=1]) }
    // Close when the dominant run exceeds what chance would produce.
    if (expectedImbalance !== 0 && theta >= expectedImbalance) {
      bars.push({
        Date:  date,
        Index: i,
        Open:  collector[0],
        High:  Math.max(...collector),
        Low:   Math.min(...collector),
        Close: collector[collector.length - 1],
        Vwap:  volPrice / cummVol,
      });

      barLengths.push(numTicks);

      // Buy proportion = buy ticks / total ticks in this bar.
      // EWMA of these estimates P[b=1] going forward.
      const buyProportion = numTicks > 0 ? posRun / numTicks : 0.5;
      buyTickProportions.push(buyProportion);

      // Reset per-bar accumulators
      posRun    = 0;
      negRun    = 0;
      cummVol   = 0;
      volPrice  = 0;
      collector = [];
      numTicks  = 0;

      // Update EWMA estimates — page 31:
      // E0[θ_T] = E0[T] * max{P[b=1], 1 − P[b=1]}
      expectedNumTicks  = ewma(barLengths, numPrevBars);
      expectedPBuy      = ewma(buyTickProportions, numPrevBars);
      expectedImbalance = Math.max(
        expectedNumTicks * Math.max(expectedPBuy, 1 - expectedPBuy),
        1e-6
      );
    }
  }

  return bars;
}

// ---------------------------------------------------------------------------
// volumeRunBars
// ---------------------------------------------------------------------------

/**
 * Volume Run Bars (VRBs) — AFML Chapter 2, page 32.
 *
 * Extends tick run bars by accumulating VOLUME on each side instead of tick counts.
 * θ_T = max(Σ buy volume, Σ sell volume)
 * Closes when θ_T >= E[T] * max(P[b=1]*E[v|buy], (1−P[b=1])*E[v|sell])
 *
 * Why weight by volume?
 *   A 10,000-share institutional buy counts as 10,000 on the buy side,
 *   not as 1. Much more sensitive to large block order flow than tick run bars.
 *
 * Four adaptive components updated via EWMA after each bar:
 *   E[T]          = expected ticks per bar
 *   P[b=1]        = expected buy probability (volume-weighted)
 *   E[v | buy]    = expected volume of a buy trade
 *   E[v | sell]   = expected volume of a sell trade
 *
 * @param trades               - labeled trades with Volume field
 * @param expectedNumTicksInit - initial guess for bar length
 * @param numPrevBars          - how many past bars to use in EWMA updates
 */
export function volumeRunBars(
  trades: LabeledTrade[],
  expectedNumTicksInit = 10,
  numPrevBars = 3
): RunBar[] {
  const bars: RunBar[] = [];

  let posRun   = 0;   // accumulated BUY volume in current bar
  let negRun   = 0;   // accumulated SELL volume in current bar
  let cummVol  = 0;
  let volPrice = 0;
  let collector: number[] = [];
  let numTicks = 0;

  const barLengths: number[]          = [];
  const buyVolProportions: number[]   = []; // avg buy volume per tick per bar
  const sellVolProportions: number[]  = []; // avg sell volume per tick per bar

  let expectedNumTicks  = expectedNumTicksInit;
  let expectedPBuy      = 0.5;
  let expectedBuyVol    = 0.01; // initial guess for avg buy volume per tick
  let expectedSellVol   = 0.01; // initial guess for avg sell volume per tick
  let expectedImbalance = 0;

  for (let i = 0; i < trades.length; i++) {
    const { Label: label, Price: price, Volume: volume, Date: date } = trades[i];

    // θ_T = max(Σ buy volume, Σ sell volume) — page 32
    // Add this trade's volume to the appropriate side. Never resets mid-bar.
    if (label === 1)       posRun += volume;  // buy side grows
    else if (label === -1) negRun += volume;  // sell side grows

    const theta = Math.max(posRun, negRun); // dominant volume run so far

    cummVol  += volume;
    volPrice += price * volume;
    collector.push(price);
    numTicks++;

    // Warmup: initialize expected imbalance before any bar has closed.
    // E0[θ_T] = E[T] * max{ P[b=1]*E[v|buy], (1−P[b=1])*E[v|sell] }
    if (bars.length === 0 && numTicks >= expectedNumTicksInit) {
      expectedImbalance = Math.max(
        expectedNumTicks * Math.max(
          expectedPBuy * expectedBuyVol,
          (1 - expectedPBuy) * expectedSellVol
        ),
        1e-6
      );
    }

    // Stopping rule — page 32:
    // T* = arg min_T { θ_T >= E[T] * max{ P[b=1]*E[v|buy], (1−P[b=1])*E[v|sell] } }
    if (expectedImbalance !== 0 && theta >= expectedImbalance) {
      bars.push({
        Date:  date,
        Index: i,
        Open:  collector[0],
        High:  Math.max(...collector),
        Low:   Math.min(...collector),
        Close: collector[collector.length - 1],
        Vwap:  volPrice / cummVol,
      });

      barLengths.push(numTicks);

      // Track avg buy/sell volume per tick — feeds EWMA updates of E[v|buy] and E[v|sell].
      // e.g. bar had 5 ticks and 200 units of buy volume → avg buy vol = 40/tick
      buyVolProportions.push(numTicks > 0 ? posRun / numTicks : expectedBuyVol);
      sellVolProportions.push(numTicks > 0 ? negRun / numTicks : expectedSellVol);

      // Reset
      posRun    = 0;
      negRun    = 0;
      cummVol   = 0;
      volPrice  = 0;
      collector = [];
      numTicks  = 0;

      // Update all four adaptive components — page 32:
      expectedNumTicks = ewma(barLengths, numPrevBars);

      // P[b=1] = volume-weighted buy fraction across recent bars
      expectedPBuy = ewma(
        buyVolProportions.map((b, idx) => {
          const s = sellVolProportions[idx];
          return (b + s) > 0 ? b / (b + s) : 0.5;
        }),
        numPrevBars
      );

      expectedBuyVol  = ewma(buyVolProportions,  numPrevBars);
      expectedSellVol = ewma(sellVolProportions, numPrevBars);

      expectedImbalance = Math.max(
        expectedNumTicks * Math.max(
          expectedPBuy * expectedBuyVol,
          (1 - expectedPBuy) * expectedSellVol
        ),
        1e-6
      );
    }
  }

  return bars;
}
