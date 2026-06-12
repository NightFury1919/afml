// standard_bars.ts — mirror of ch02/bars/standard_bars.py
// AFML Chapter 2, Section 2.3.1
//
// 📁 C:\ws\AFML\
// └── ch02_ts\
//     └── bars_ts\
//         └── standard_bars.ts   ← goes here

// ---------------------------------------------------------------------------
// Types
// ---------------------------------------------------------------------------

export interface Trade {
  Date: string | Date;
  Price: number;
  Volume: number;
}

export interface Bar {
  Date: string | Date;
  Index: number;
  Open: number;
  Low: number;
  High: number;
  Close: number;
  Vwap: number;
}

// ---------------------------------------------------------------------------
// Helpers
// ---------------------------------------------------------------------------

/** Build a Bar object from a completed collector window. */
function buildBar(
  date: string | Date,
  index: number,
  collector: number[],
  volPrice: number,
  cummVol: number
): Bar {
  return {
    Date:  date,
    Index: index,
    Open:  collector[0],
    High:  Math.max(...collector),
    Low:   Math.min(...collector),
    Close: collector[collector.length - 1],
    Vwap:  volPrice / cummVol,   // VWAP = Σ(price × volume) / Σ(volume)
  };
}

// ---------------------------------------------------------------------------
// timeBars
// ---------------------------------------------------------------------------

/**
 * Time Bars — sample at fixed calendar intervals.
 * AFML Chapter 2, Section 2.3.1.
 *
 * Each bar represents one period (e.g. one day) regardless of trading activity.
 * The book notes time bars oversample quiet periods and undersample active ones.
 *
 * How it works:
 *   Groups all trades that share the same period key (e.g. "2026-03-01"),
 *   then aggregates: Open = first price, High = max, Low = min,
 *   Close = last price, VWAP = Σ(p*v)/Σ(v).
 *
 * @param trades - array of trades in chronological order
 * @param freq   - 'D' (daily) | 'W' (weekly) | 'M' (monthly) | 'H' (hourly)
 * @returns array of OHLC+VWAP bars, one per period
 */
export function timeBars(trades: Trade[], freq: 'D' | 'W' | 'M' | 'H' = 'W'): Bar[] {
  if (trades.length === 0) return [];

  // Group trades by period key (mirrors pandas resample())
  const groups = new Map<string, Trade[]>();

  for (const trade of trades) {
    const key = getPeriodKey(new Date(trade.Date), freq);
    if (!groups.has(key)) groups.set(key, []);
    groups.get(key)!.push(trade);
  }

  const bars: Bar[] = [];
  let index = 0;

  // Iterate in insertion order (chronological, since trades are sorted)
  for (const [, group] of groups) {
    const prices  = group.map(t => t.Price);
    const volumes = group.map(t => t.Volume);
    const volPrice = group.reduce((sum, t) => sum + t.Price * t.Volume, 0);
    const cummVol  = volumes.reduce((a, b) => a + b, 0);

    if (cummVol === 0) continue; // skip empty periods (mirrors .dropna())

    bars.push({
      Date:  group[0].Date,
      Index: index++,
      Open:  prices[0],
      High:  Math.max(...prices),
      Low:   Math.min(...prices),
      Close: prices[prices.length - 1],
      Vwap:  volPrice / cummVol,
    });
  }

  return bars;
}

/** Returns a string key that groups a date into the requested period. */
function getPeriodKey(date: Date, freq: 'D' | 'W' | 'M' | 'H'): string {
  switch (freq) {
    case 'H':
      // e.g. "2026-03-01T14" — one bucket per hour
      return `${date.getFullYear()}-${pad(date.getMonth() + 1)}-${pad(date.getDate())}T${pad(date.getHours())}`;
    case 'D':
      // e.g. "2026-03-01"
      return `${date.getFullYear()}-${pad(date.getMonth() + 1)}-${pad(date.getDate())}`;
    case 'W': {
      // ISO week: find the Monday of the current week
      const monday = new Date(date);
      const day = date.getDay();                         // 0=Sun … 6=Sat
      const diff = (day === 0 ? -6 : 1 - day);          // days back to Monday
      monday.setDate(date.getDate() + diff);
      return `${monday.getFullYear()}-W${pad(monday.getMonth() + 1)}-${pad(monday.getDate())}`;
    }
    case 'M':
      // e.g. "2026-03"
      return `${date.getFullYear()}-${pad(date.getMonth() + 1)}`;
  }
}

function pad(n: number): string {
  return n.toString().padStart(2, '0');
}

// ---------------------------------------------------------------------------
// tickBars
// ---------------------------------------------------------------------------

/**
 * Tick Bars — close a bar every N trades.
 * AFML Chapter 2, Section 2.3.1.
 *
 * Every bar contains exactly `thresh` trades. During active markets, bars
 * close faster; during quiet markets, slower. Each bar represents the same
 * NUMBER of trading events — not the same calendar time.
 *
 * Remaining limitation: ignores trade size. Volume bars fix this.
 *
 * @param trades - array of trades in chronological order
 * @param thresh - number of trades per bar
 */
export function tickBars(trades: Trade[], thresh: number): Bar[] {
  const bars: Bar[]     = [];
  let collector: number[] = [];
  let cummVol  = 0;
  let volPrice = 0;

  for (let i = 0; i < trades.length; i++) {
    const { Price: price, Volume: volume, Date: date } = trades[i];

    collector.push(price);
    cummVol  += volume;
    volPrice += price * volume;

    if (collector.length >= thresh) {
      bars.push(buildBar(date, i, collector, volPrice, cummVol));
      // Reset — start a fresh bar
      collector = [];
      cummVol   = 0;
      volPrice  = 0;
    }
  }

  return bars;
}

// ---------------------------------------------------------------------------
// volumeBars
// ---------------------------------------------------------------------------

/**
 * Volume Bars — close a bar every time `thresh` units have been traded.
 * AFML Chapter 2, Section 2.3.1.
 *
 * Improves on tick bars by accounting for trade SIZE. One massive institutional
 * trade and thousands of tiny retail trades contribute equally per unit traded.
 * Each bar represents the same QUANTITY of asset changing hands.
 *
 * Remaining limitation: ignores price level. Dollar bars fix this.
 *
 * @param trades - array of trades in chronological order
 * @param thresh - volume units per bar
 */
export function volumeBars(trades: Trade[], thresh: number): Bar[] {
  const bars: Bar[]     = [];
  let collector: number[] = [];
  let cummVol  = 0;
  let volPrice = 0;

  for (let i = 0; i < trades.length; i++) {
    const { Price: price, Volume: volume, Date: date } = trades[i];

    cummVol  += volume;
    volPrice += price * volume;
    collector.push(price);

    if (cummVol >= thresh) {
      bars.push(buildBar(date, i, collector, volPrice, cummVol));
      collector = [];
      cummVol   = 0;
      volPrice  = 0;
    }
  }

  return bars;
}

// ---------------------------------------------------------------------------
// dollarBars
// ---------------------------------------------------------------------------

/**
 * Dollar Bars — close a bar every time `thresh` dollars of value has traded.
 * AFML Chapter 2, Section 2.3.1.
 *
 * Most robust standard bar type. Unaffected by stock splits, price-level
 * changes, or corporate actions. Each bar represents the same DOLLAR VALUE
 * of economic activity.
 *
 * Example: thresh = $50,000.
 *   One trade: 10 BTC at $5,000 each = $50,000 → bar closes immediately.
 *   Ten trades: 1 BTC at $5,000 each → bar closes after 10 trades.
 *
 * The book recommends dollar bars as the best standard bar type and uses
 * them throughout subsequent chapters.
 *
 * @param trades - array of trades in chronological order
 * @param thresh - dollar value per bar
 */
export function dollarBars(trades: Trade[], thresh: number): Bar[] {
  const bars: Bar[]     = [];
  let collector: number[] = [];
  let cummDollar = 0;
  let cummVol    = 0;
  let volPrice   = 0;

  for (let i = 0; i < trades.length; i++) {
    const { Price: price, Volume: volume, Date: date } = trades[i];
    const dollar = price * volume;

    cummDollar += dollar;
    cummVol    += volume;
    volPrice   += dollar;   // volPrice === cummDollar for dollar bars
    collector.push(price);

    if (cummDollar >= thresh) {
      bars.push(buildBar(date, i, collector, volPrice, cummVol));
      collector  = [];
      cummDollar = 0;
      cummVol    = 0;
      volPrice   = 0;
    }
  }

  return bars;
}
