// etf_trick.ts — mirror of ch02/multi_product/etf_trick.py
// AFML Chapter 2, Section 2.4.1, pages 33-34
//
// 📁 C:\ws\AFML\
// └── ch02_ts\
//     └── multi_product_ts\
//         └── etf_trick.ts   ← goes here

// ---------------------------------------------------------------------------
// Types
// ---------------------------------------------------------------------------

/**
 * A 2-D data table: rows indexed by date string, columns by instrument name.
 * Mirrors a pandas DataFrame with a DatetimeIndex and named columns.
 *
 * Example:
 *   matrix["2026-03-01"]["SP98H"] = 4500.25
 */
export type Matrix = Record<string, Record<string, number>>;

/**
 * A 1-D series indexed by date string.
 * Mirrors a pandas Series with a DatetimeIndex.
 */
export type DateSeries = Record<string, number>;

/**
 * A fixed vector indexed by instrument name (no date dimension).
 * Used for trans_costs: one value per instrument, constant over time.
 */
export type InstrumentVector = Record<string, number>;

/** Output row for one time bar. */
export interface EtfBar {
  date: string;
  K: number;               // portfolio value (starts at $1)
  rebalanceCost: number;   // transaction cost on rebalance days, else 0
  bidAskCost: number;      // spread cost every bar
  volume: number;          // tradeable units limited by least-liquid leg
}

// ---------------------------------------------------------------------------
// Helpers
// ---------------------------------------------------------------------------

/** Sum of absolute values across an instrument row. */
function sumAbs(row: Record<string, number>): number {
  return Object.values(row).reduce((acc, v) => acc + Math.abs(v), 0);
}

/** Dot product of two instrument rows: Σ a_i * b_i */
function dot(a: Record<string, number>, b: Record<string, number>): number {
  return Object.keys(a).reduce((acc, k) => acc + (a[k] ?? 0) * (b[k] ?? 0), 0);
}

/** Element-wise multiply two rows: { i: a_i * b_i } */
function elemMul(
  a: Record<string, number>,
  b: Record<string, number>
): Record<string, number> {
  const result: Record<string, number> = {};
  for (const k of Object.keys(a)) result[k] = (a[k] ?? 0) * (b[k] ?? 0);
  return result;
}

/** Element-wise absolute value of a row. */
function absRow(row: Record<string, number>): Record<string, number> {
  const result: Record<string, number> = {};
  for (const k of Object.keys(row)) result[k] = Math.abs(row[k] ?? 0);
  return result;
}

/** Minimum value in a row (ignoring NaN / Infinity). */
function rowMin(row: Record<string, number>): number {
  const vals = Object.values(row).filter(v => isFinite(v));
  return vals.length > 0 ? Math.min(...vals) : 0;
}

// ---------------------------------------------------------------------------
// etfTrick
// ---------------------------------------------------------------------------

/**
 * ETF Trick — AFML Chapter 2, Section 2.4.1, pages 33-34.
 *
 * Converts a basket of multiple instruments into a single continuous $1
 * investment series (K_t), solving three problems with naive price stitching:
 *
 *   Problem 1 — Units: instruments have different notional values / point values.
 *   Problem 2 — Roll gaps: futures contract switches create artificial price jumps.
 *   Problem 3 — Negative prices: naive gap adjustments can produce negative values.
 *
 * The ETF Trick tracks the VALUE (in dollars) of a fixed $1 initial investment,
 * evolving day by day through actual P&L rather than stitching raw prices.
 *
 * @param dates           - chronological array of date strings (one per bar)
 * @param openPrices      - Matrix[date][instrument] — open prices
 * @param closePrices     - Matrix[date][instrument] — close prices
 * @param allocWeights    - Matrix[date][instrument] — target weights ω_{i,t}
 * @param pointValues     - Matrix[date][instrument] — φ_{i,t}: $/point
 * @param dividends       - Matrix[date][instrument] — d_{i,t}: dividends/coupons
 * @param rebalanceDates  - Set of date strings when portfolio is rebalanced
 * @param transCosts      - InstrumentVector — τ_i: cost per dollar of notional
 * @param volumes         - Matrix[date][instrument] — actual traded volume (optional)
 * @returns array of EtfBar, one per time bar starting from bar index 1
 */
export function etfTrick(
  dates: string[],
  openPrices: Matrix,
  closePrices: Matrix,
  allocWeights: Matrix,
  pointValues: Matrix,
  dividends: Matrix,
  rebalanceDates: Set<string>,
  transCosts?: InstrumentVector,
  volumes?: Matrix
): EtfBar[] {
  const T = dates.length;
  const instruments = Object.keys(closePrices[dates[0]]);

  // Default to zero transaction costs if not provided
  const costs: InstrumentVector = transCosts ?? Object.fromEntries(
    instruments.map(i => [i, 0])
  );

  // Use real volume data if provided, otherwise fall back to close prices as proxy
  const volumeData = volumes ?? closePrices;

  // Holdings h_{i,t}: units of each instrument held at bar t
  // Initialise to zero for all instruments on all dates
  const h: Record<string, Record<string, number>> = {};
  for (const d of dates) {
    h[d] = Object.fromEntries(instruments.map(i => [i, 0]));
  }

  // K_t: portfolio value — starts at $1 on day 0
  const K: Record<string, number> = { [dates[0]]: 1.0 };

  const result: EtfBar[] = [];

  for (let tIdx = 1; tIdx < T; tIdx++) {
    const t     = dates[tIdx];       // current date
    const tPrev = dates[tIdx - 1];  // previous date

    // -------------------------------------------------------------------
    // Step 1: Compute holdings h_{i,t} (page 34)
    // -------------------------------------------------------------------
    // On a REBALANCE DATE (t-1 ∈ B): recalculate units to match target weights.
    // We enter at tomorrow's OPEN price (open_prices[t]).
    //
    // Formula: h_{i,t} = (ω_{i,t} × K_t) / (o_{i,t+1} × φ_{i,t} × Σ|ω_{i,t}|)
    //
    //   ω_{i,t} × K_t      = dollar allocation to instrument i
    //   o_{i,t+1} × φ_{i,t} = dollar value of one unit at open
    //   Σ|ω_{i,t}|         = normalises for leverage
    //
    // On a non-rebalance date: carry forward yesterday's holdings unchanged.
    if (rebalanceDates.has(tPrev)) {
      const w     = allocWeights[tPrev];
      const denom = sumAbs(w); // Σ|ω_i|
      for (const i of instruments) {
        h[t][i] = (w[i] * K[tPrev]) / (
          openPrices[t][i] * pointValues[tPrev][i] * denom
        );
      }
    } else {
      // Carry forward — copy previous holdings
      for (const i of instruments) h[t][i] = h[tPrev][i];
    }

    // -------------------------------------------------------------------
    // Step 2: Compute price change δ_{i,t} (page 34)
    // -------------------------------------------------------------------
    // After rebalance: entered at open, P&L is close − open (intraday).
    // Otherwise: held from previous close, P&L is close_t − close_{t-1}.
    const delta: Record<string, number> = {};
    for (const i of instruments) {
      delta[i] = rebalanceDates.has(tPrev)
        ? closePrices[t][i] - openPrices[t][i]    // intraday change
        : closePrices[t][i] - closePrices[tPrev][i]; // overnight change
    }

    // -------------------------------------------------------------------
    // Step 3: Update portfolio value K_t (page 34)
    // -------------------------------------------------------------------
    // K_t = K_{t-1} + Σ_i h_{i,t-1} × φ_{i,t} × (δ_{i,t} + d_{i,t})
    //
    //   h_{i,t-1}          = units held (from yesterday)
    //   φ_{i,t}            = $/point
    //   δ_{i,t} + d_{i,t} = price move + dividend
    //   sum across i       = total P&L today → added to yesterday's K
    let pnl = 0;
    for (const i of instruments) {
      pnl += h[tPrev][i] * pointValues[t][i] * (delta[i] + dividends[t][i]);
    }
    K[t] = K[tPrev] + pnl;

    // -------------------------------------------------------------------
    // Step 4: Transaction costs (page 34)
    // -------------------------------------------------------------------
    let rebalanceCost = 0;

    if (rebalanceDates.has(tPrev)) {
      // Rebalance cost c_t: cost of closing old + opening new positions.
      // c_t = Σ_i (|h_{i,t-1}|*p_{i,t} + |h_{i,t}|*o_{i,t+1}) * φ_{i,t} * τ_i
      for (const i of instruments) {
        rebalanceCost +=
          (Math.abs(h[tPrev][i]) * closePrices[t][i] +
           Math.abs(h[t][i])     * openPrices[t][i]) *
          pointValues[t][i] * costs[i];
      }
    }

    // Bid-ask cost c~_t: computed EVERY bar.
    // Represents the cost of exiting positions right now at the spread.
    // c~_t = Σ_i |h_{i,t-1}| × p_{i,t} × φ_{i,t} × τ_i
    let bidAskCost = 0;
    for (const i of instruments) {
      bidAskCost +=
        Math.abs(h[tPrev][i]) * closePrices[t][i] *
        pointValues[t][i] * costs[i];
    }

    // -------------------------------------------------------------------
    // Step 5: Tradeable volume v_t (page 34)
    // -------------------------------------------------------------------
    // v_t = min_i { v_{i,t} / |h_{i,t-1}| }
    //
    // The portfolio can only trade as many ETF units as the least-liquid
    // instrument allows. Each instrument's available ETF units =
    // its own traded volume / the number of units we hold of it.
    let vol = 0;
    const hPrevAbs = absRow(h[tPrev]);
    const anyHeld  = Object.values(hPrevAbs).some(v => v > 0);

    if (anyHeld) {
      const volRatios: Record<string, number> = {};
      for (const i of instruments) {
        volRatios[i] = hPrevAbs[i] > 0
          ? volumeData[t][i] / hPrevAbs[i]
          : Infinity; // not holding this instrument → no constraint from it
      }
      vol = rowMin(volRatios); // bottleneck = least-liquid leg
    }

    result.push({
      date:          t,
      K:             K[t],
      rebalanceCost,
      bidAskCost,
      volume:        vol,
    });
  }

  return result;
}
