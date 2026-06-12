// utils.ts — mirror of ch02/bars/utils.py
// Utility functions used throughout Chapter 2 bar construction.

// ---------------------------------------------------------------------------
// Types
// ---------------------------------------------------------------------------

export interface Tick {
  Price: number;
  [key: string]: number; // allow extra columns (Volume, etc.)
}

export interface TickWithDelta extends Tick {
  Delta: number;
}

export interface TickWithLabel extends TickWithDelta {
  Label: number; // +1 (buy) or -1 (sell)
}

// ---------------------------------------------------------------------------
// ewma
// ---------------------------------------------------------------------------

/**
 * Exponentially Weighted Moving Average (EWMA)
 * AFML pages 31–32.
 *
 * Formula: ewma_t = alpha * x_t + (1 - alpha) * ewma_{t-1}
 *          where alpha = 2 / (window + 1)
 *
 * Why use it?
 *   A regular average treats every data point equally.
 *   An EWMA gives MORE weight to recent observations and LESS weight to older ones,
 *   so the average "follows" recent data more closely — ideal for financial data
 *   where recent behavior is more predictive than distant history.
 *
 *   alpha close to 1  → nearly all weight on the newest value (very reactive)
 *   alpha close to 0  → nearly all weight on history    (very smooth / slow)
 *
 * Example with window=3:  alpha = 2/(3+1) = 0.5
 *   arr = [10, 12, 8]
 *   ewma_0 = 10
 *   ewma_1 = 0.5*12 + 0.5*10 = 11.0
 *   ewma_2 = 0.5*8  + 0.5*11 = 9.5
 */
export function ewma(arr: number[], window: number): number {
  if (arr.length === 0) {
    // Nothing to average — return 0 (mirrors Python early return)
    return 0;
  }

  // alpha: smoothing factor — controls how fast old values decay
  const alpha = 2 / (window + 1);

  // Seed the running average with the very first value.
  // There is no "previous" average yet, so we use the first data point.
  let ewmaVal = arr[0];

  for (let i = 1; i < arr.length; i++) {
    // Blend: new value gets weight alpha, running history gets weight (1 - alpha)
    ewmaVal = alpha * arr[i] + (1 - alpha) * ewmaVal;
  }

  return ewmaVal;
}

// ---------------------------------------------------------------------------
// delta
// ---------------------------------------------------------------------------

/**
 * Computes Δp_t — the price change between consecutive ticks.
 * Used as input to the Tick Rule (page 29).
 *
 * Why price differences?
 *   The Tick Rule needs to know whether the price went UP or DOWN at each trade.
 *   We subtract the previous price from the current price; the sign (+/-) tells
 *   us the direction.
 *
 *   The first tick has no previous price, so its delta is defined as 0.
 *
 * Returns a new array of ticks with a Delta field added (does not mutate input).
 */
export function delta(ticks: Tick[]): TickWithDelta[] {
  return ticks.map((tick, i) => {
    const d = i === 0
      ? 0                                    // no previous price → delta = 0
      : tick.Price - ticks[i - 1].Price;    // current minus previous
    return { ...tick, Delta: d };
  });
}

// ---------------------------------------------------------------------------
// tickRule
// ---------------------------------------------------------------------------

/**
 * Tick Rule — AFML Chapter 2, page 29.
 *
 * Assigns a direction b_t to each trade:
 *   b_t = b_{t-1}         if Δp_t = 0  (price unchanged → carry forward)
 *   b_t = |Δp_t| / Δp_t  if Δp_t ≠ 0  (gives +1 for uptick, -1 for downtick)
 *
 * b_t is a proxy for trade direction: buy = +1, sell = -1.
 *
 * Why do we need direction labels?
 *   Public trade data usually does NOT say whether a trade was buyer- or
 *   seller-initiated. The Tick Rule is a simple heuristic:
 *     price rose  → buyer was aggressive → b = +1
 *     price fell  → seller was aggressive → b = -1
 *     price flat  → we can't tell → carry forward the previous label
 *
 * |Δp_t| / Δp_t is just the mathematical sign function:
 *   positive Δp_t → +1
 *   negative Δp_t → -1
 *
 * Input ticks must already have a Delta field (run delta() first).
 * Returns a new array with a Label field added (does not mutate input).
 */
export function tickRule(ticks: TickWithDelta[]): TickWithLabel[] {
  const labels: number[] = new Array(ticks.length).fill(1); // default +1

  for (let i = 1; i < ticks.length; i++) {      // skip index 0 (no previous)
    const d = ticks[i].Delta;
    if (d === 0) {
      labels[i] = labels[i - 1];                // flat → carry forward
    } else {
      labels[i] = Math.abs(d) / d;              // uptick → +1, downtick → -1
    }
  }

  return ticks.map((tick, i) => ({ ...tick, Label: labels[i] }));
}

// ---------------------------------------------------------------------------
// estimateBuySellProbs
// ---------------------------------------------------------------------------

/**
 * Estimates p_b and p_s — the probability that a tick is a buy or sell.
 * Used to initialize expected imbalance before the first bar is formed.
 * AFML page 31 (initial conditions for imbalance bars).
 *
 *   p_b = count of buy ticks  / total ticks
 *   p_s = count of sell ticks / total ticks
 *
 * Why do we need these probabilities?
 *   Imbalance and run bars need an initial guess for "how imbalanced is a
 *   typical bar?" before any bars have been formed. Rather than hard-coding
 *   a guess, we estimate it empirically from the raw tick data.
 *
 * Example: 900 buy ticks, 100 sell ticks → p_b = 0.9, p_s = 0.1
 *
 * Input ticks must already have a Label field (run tickRule() first).
 */
export function estimateBuySellProbs(ticks: TickWithLabel[]): { p_b: number; p_s: number } {
  const buys  = ticks.filter(t => t.Label ===  1).length;
  const sells = ticks.filter(t => t.Label === -1).length;
  const total = buys + sells;

  if (total === 0) {
    // No labeled ticks — return 50/50 as a safe default
    return { p_b: 0.5, p_s: 0.5 };
  }

  return {
    p_b: buys  / total,   // fraction of ticks that were buys
    p_s: sells / total,   // fraction of ticks that were sells
  };
}
