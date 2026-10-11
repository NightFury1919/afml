"""Pull long-history daily candles for Binance spot USDT pairs from Binance's public data archive.

Research use only. This reads the public archive at data.binance.vision (monthly kline zip files),
not the trading API, so it needs no key and has no order-book or account access.

WHY
    Kraken's /OHLC endpoint returns at most 720 daily candles, about two years. The 21-day edge test
    (arm C of the crypto family) needs many more independent periods than two years can give. The
    archive goes back to about 2017 and, unlike a list of today's coins, its directory listing also
    contains pairs that were later delisted, which reduces (does not remove) survivorship bias.

Run from C:\\ws\\AFML\\portfolio in the mlfinlab env:
    python pull_binance_daily.py --max-symbols 5 --tag test5     # quick check first
    python pull_binance_daily.py                                 # full pull (tens of minutes)

Outputs (tag defaults to today's UTC date; existing files are NEVER overwritten):
    binance_prices_daily_asof_<tag>.csv      date x symbol close prices (panel_data.load_prices format)
    binance_ohlcv_daily_asof_<tag>.csv       long: date, symbol, open, high, low, close, volume,
                                             quote_volume (dollar volume), trades
    binance_pairs_manifest_asof_<tag>.csv    one row per symbol: history, gaps, liquidity, status

LOAD-BEARING DECISIONS (do not change without a reason written next to the change)
    * Timestamps: from 2025 the archive's spot files use MICROSECONDS, older files use milliseconds.
      parse_kline_zip converts by magnitude. Reading microseconds as milliseconds puts every date
      thousands of years in the future, so the date range is also sanity-checked.
    * Leveraged tokens (UP, DOWN, BULL, BEAR) are dropped: they are derivatives of another coin and
      decay by construction. Stablecoin and fiat bases are FLAGGED in the manifest, not dropped, so
      the exclusion rule is a visible, registered choice (same policy as pull_crypto_daily.py).
    * The archive's monthly files stop at the last COMPLETE month, so the newest days are missing.
      That is intended; the Kraken pull covers the recent period and no partial day ever enters.
    * Missing months inside a coin's history are counted, not filled.
    * Overwrite is refused BEFORE any network call.
"""
import argparse
import io
import re
import time
import xml.etree.ElementTree as ET
import zipfile
from concurrent.futures import ThreadPoolExecutor
from datetime import datetime, timezone
from pathlib import Path

import pandas as pd

ARCHIVE = "https://data.binance.vision"
BUCKET_LISTING = "https://s3-ap-northeast-1.amazonaws.com/data.binance.vision"
KLINE_PREFIX = "data/spot/monthly/klines/"
QUOTE = "USDT"
KLINE_COLUMNS = ["open_time", "open", "high", "low", "close", "volume", "close_time",
                 "quote_volume", "trades", "taker_base", "taker_quote", "ignore"]
LEVERAGED_SUFFIXES = ("UP", "DOWN", "BULL", "BEAR")
STABLE_OR_FIAT_BASES = {
    "USDC", "TUSD", "BUSD", "FDUSD", "USDP", "DAI", "USDS", "USDE", "PYUSD", "AEUR", "UST", "USTC",
    "EUR", "EURI", "GBP", "AUD", "TRY", "BRL", "RUB", "UAH", "NGN", "ARS", "BIDR", "IDRT", "BVND",
    "PAX", "SUSD", "VAI", "XUSD", "USD1", "RLUSD", "USDSB",
}
MIN_PLAUSIBLE = pd.Timestamp("2015-01-01")
MAX_PLAUSIBLE = pd.Timestamp("2100-01-01")
RETRY_STATUS = {429, 500, 502, 503, 504}


# ------------------------------------------------------------------ listing
def parse_listing_xml(text):
    """One page of an S3 listing -> (prefixes, keys, next_marker or None)."""
    root = ET.fromstring(text)
    ns = root.tag.split("}")[0] + "}" if root.tag.startswith("{") else ""
    prefixes = [p.findtext(f"{ns}Prefix") for p in root.findall(f"{ns}CommonPrefixes")]
    keys = [c.findtext(f"{ns}Key") for c in root.findall(f"{ns}Contents")]
    truncated = (root.findtext(f"{ns}IsTruncated") or "").lower() == "true"
    nxt = root.findtext(f"{ns}NextMarker")
    if truncated and not nxt:
        # without a delimiter S3 omits NextMarker; the last key or prefix is the marker
        last = (keys or prefixes or [None])[-1]
        nxt = last
    return prefixes, keys, (nxt if truncated else None)


def _get(session, url, params, sleep, backoff, max_retries):
    """GET with retries on rate-limit / server errors and on connection errors. Returns the response."""
    last = None
    for attempt in range(max_retries + 1):
        try:
            resp = session.get(url, params=params, timeout=60)
        except Exception as e:                       # connection reset, timeout, ...
            last = e
            if attempt == max_retries:
                raise
            sleep(backoff * (attempt + 1))
            continue
        if resp.status_code in RETRY_STATUS:
            last = RuntimeError(f"HTTP {resp.status_code} for {url}")
            if attempt == max_retries:
                raise last
            sleep(backoff * (attempt + 1))
            continue
        return resp
    raise last                                        # pragma: no cover


def list_all(session, prefix, delimiter, sleep=time.sleep, backoff=2.0, max_retries=5):
    """Every prefix and key under `prefix`, following pagination."""
    prefixes, keys, marker = [], [], None
    while True:
        params = {"prefix": prefix, "max-keys": 1000}
        if delimiter:
            params["delimiter"] = delimiter
        if marker:
            params["marker"] = marker
        resp = _get(session, BUCKET_LISTING, params, sleep, backoff, max_retries)
        if resp.status_code != 200:
            raise RuntimeError(f"listing {prefix!r} failed: HTTP {resp.status_code}")
        p, k, marker = parse_listing_xml(resp.text)
        prefixes += p
        keys += k
        if marker is None:
            return prefixes, keys


def select_symbols(prefixes):
    """USDT pairs from the listing's prefixes. Returns dicts: symbol, base, stable_or_fiat_base.

    A leveraged token is a base made of ANOTHER listed base plus UP / DOWN / BULL / BEAR (BTCUP,
    ETHDOWN). Testing the suffix alone would wrongly drop real coins such as JUP or SYRUP, so the
    remainder must itself be a listed USDT base. Leveraged tokens are dropped; stablecoin and fiat
    bases are kept and flagged.
    """
    syms = []
    for p in prefixes:
        sym = p.rstrip("/").split("/")[-1]
        if sym.endswith(QUOTE) and len(sym) > len(QUOTE):
            syms.append(sym)
    bases = {s[: -len(QUOTE)] for s in syms}
    out = []
    for sym in syms:
        base = sym[: -len(QUOTE)]
        if any(base.endswith(x) and base[: -len(x)] in bases for x in LEVERAGED_SUFFIXES):
            continue
        out.append({"symbol": sym, "base": base, "stable_or_fiat_base": base in STABLE_OR_FIAT_BASES})
    return sorted(out, key=lambda d: d["symbol"])


# ------------------------------------------------------------------- parsing
def _to_datetime(open_time):
    """Open times -> UTC dates. Archive files use ms before 2025 and us after; decide by magnitude."""
    t = pd.to_numeric(open_time, errors="raise").astype("int64")
    unit_us = t > 10 ** 14                           # ms values are ~1.5e12, us values ~1.7e15
    ms = t.where(~unit_us, t // 1000)
    return pd.to_datetime(ms, unit="ms")


def parse_kline_zip(blob, symbol=""):
    """Bytes of one monthly zip -> DataFrame [date, open, high, low, close, volume, quote_volume, trades]."""
    with zipfile.ZipFile(io.BytesIO(blob)) as z:
        bad = z.testzip()
        if bad is not None:
            raise ValueError(f"{symbol}: corrupt zip member {bad}")
        name = z.namelist()[0]
        raw = z.read(name).decode("utf-8")
    df = pd.read_csv(io.StringIO(raw), header=None, names=KLINE_COLUMNS, dtype=str)
    if len(df) and not str(df.iloc[0]["open_time"]).strip().isdigit():
        df = df.iloc[1:]                              # a header row, in files that carry one
    if df.empty:
        return pd.DataFrame(columns=["date", "open", "high", "low", "close", "volume", "quote_volume", "trades"])
    ts = _to_datetime(df["open_time"])
    out = pd.DataFrame({"date": ts.dt.normalize()})
    for c in ["open", "high", "low", "close", "volume", "quote_volume", "trades"]:
        out[c] = pd.to_numeric(df[c].to_numpy(), errors="raise")
    if (out["date"] < MIN_PLAUSIBLE).any() or (out["date"] > MAX_PLAUSIBLE).any():
        raise ValueError(f"{symbol}: dates outside {MIN_PLAUSIBLE.year}-{MAX_PLAUSIBLE.year}; "
                         "timestamp unit probably misread")
    return out


def month_of(key):
    """'.../BTCUSDT-1d-2024-03.zip' -> '2024-03'."""
    m = re.search(r"-(\d{4}-\d{2})\.zip$", key)
    return m.group(1) if m else None


def missing_months(months):
    """Months absent between the first and last of a sorted list of 'YYYY-MM' strings."""
    if not months:
        return []
    allm = pd.period_range(months[0], months[-1], freq="M").strftime("%Y-%m")
    return [m for m in allm if m not in set(months)]


# ---------------------------------------------------------------- per-symbol
def fetch_symbol(session, symbol, start_year=2017, sleep=time.sleep, backoff=2.0, max_retries=5):
    """All monthly files for one symbol -> (daily DataFrame, list of missing months, n_files)."""
    prefix = f"{KLINE_PREFIX}{symbol}/1d/"
    _, keys = list_all(session, prefix, None, sleep, backoff, max_retries)
    zips = sorted(k for k in keys if k.endswith(".zip") and month_of(k))
    zips = [k for k in zips if int(month_of(k)[:4]) >= start_year]
    frames = []
    for k in zips:
        resp = _get(session, f"{ARCHIVE}/{k}", None, sleep, backoff, max_retries)
        if resp.status_code != 200:
            raise RuntimeError(f"{symbol}: {k} -> HTTP {resp.status_code}")
        frames.append(parse_kline_zip(resp.content, symbol))
    if not frames:
        return pd.DataFrame(), [], 0
    df = pd.concat(frames, ignore_index=True)
    df = df.drop_duplicates("date", keep="last").sort_values("date").reset_index(drop=True)
    return df, missing_months([month_of(k) for k in zips]), len(zips)


def summarize_symbol(df, missing, n_files):
    if df.empty:
        return dict(n_rows=0, first_date="", last_date="", max_gap_days=0, median_dollar_volume=0.0,
                    total_trades=0, n_months=0, n_missing_months=0, n_nonpositive_close=0)
    gaps = df["date"].diff().dt.days
    return dict(n_rows=int(len(df)), first_date=df["date"].iloc[0].date().isoformat(),
                last_date=df["date"].iloc[-1].date().isoformat(),
                max_gap_days=int(gaps.max()) if len(df) > 1 else 0,
                median_dollar_volume=float(df["quote_volume"].median()),
                total_trades=int(df["trades"].sum()), n_months=int(n_files),
                n_missing_months=int(len(missing)), n_nonpositive_close=int((df["close"] <= 0).sum()))


def build_wide_close(frames):
    """{symbol: daily df} -> date x symbol close prices, no fill."""
    series = {s: f.set_index("date")["close"] for s, f in frames.items() if len(f)}
    return pd.DataFrame(series).sort_index().rename_axis("date")


# --------------------------------------------------------------------- files
def output_paths(out_dir, tag):
    d = Path(out_dir)
    return {"prices": d / f"binance_prices_daily_asof_{tag}.csv",
            "ohlcv": d / f"binance_ohlcv_daily_asof_{tag}.csv",
            "manifest": d / f"binance_pairs_manifest_asof_{tag}.csv"}


def check_outputs_free(paths):
    for p in paths:
        if Path(p).exists():
            raise FileExistsError(f"{p} already exists -- refusing to overwrite. "
                                  "Use a different --tag or move the old file.")


def run(out_dir, tag, only_symbols=None, max_symbols=None, start_year=2017, workers=8,
        session_factory=None, sleep=time.sleep, log=print):
    paths = output_paths(out_dir, tag)
    check_outputs_free(paths.values())               # before any network call
    if session_factory is None:
        import requests
        session_factory = requests.Session

    prefixes, _ = list_all(session_factory(), KLINE_PREFIX, "/", sleep)
    chosen = select_symbols(prefixes)
    if only_symbols:
        want = set(only_symbols)
        unknown = want - {c["symbol"] for c in chosen}
        if unknown:
            raise ValueError(f"not USDT spot symbols in the archive: {sorted(unknown)}")
        chosen = [c for c in chosen if c["symbol"] in want]
    if max_symbols:
        chosen = chosen[:max_symbols]
    log(f"{len(chosen)} USDT symbols to pull")

    def work(c):
        try:
            df, missing, n = fetch_symbol(session_factory(), c["symbol"], start_year, sleep)
            return c, df, missing, n, None
        except Exception as e:                         # recorded in the manifest, run continues
            return c, pd.DataFrame(), [], 0, f"{type(e).__name__}: {e}"

    rows, frames = [], {}
    with ThreadPoolExecutor(max_workers=workers) as pool:
        for i, (c, df, missing, n, err) in enumerate(pool.map(work, chosen), 1):
            s = summarize_symbol(df, missing, n)
            status = "error" if err else ("empty" if df.empty else "ok")
            rows.append({**c, **s, "status": status, "error": err or ""})
            if len(df):
                frames[c["symbol"]] = df
            tail = err or f"{s['n_rows']} days {s['first_date']} to {s['last_date']}"
            log(f"[{i}/{len(chosen)}] {c['symbol']}: {tail}")

    manifest = pd.DataFrame(rows)
    manifest.to_csv(paths["manifest"], index=False)    # written first so a crash still leaves a record
    if not frames:
        raise RuntimeError("No symbol returned data; manifest written, nothing else saved. "
                           "Check the errors in the manifest.")
    wide = build_wide_close(frames)
    wide.to_csv(paths["prices"], index_label="date")
    long = pd.concat([f.assign(symbol=s) for s, f in frames.items()], ignore_index=True)
    long = long[["date", "symbol", "open", "high", "low", "close", "volume", "quote_volume", "trades"]]
    long.sort_values(["date", "symbol"]).to_csv(paths["ohlcv"], index=False)

    ok = manifest[manifest["status"] == "ok"]
    log(f"\n{len(ok)} symbols with data, {(manifest.status == 'error').sum()} errors, "
        f"{(manifest.status == 'empty').sum()} empty")
    log(f"Days of history per symbol: min {int(ok.n_rows.min())}, median {int(ok.n_rows.median())}, "
        f"max {int(ok.n_rows.max())}")
    log(f"Symbols with a gap longer than 3 days: {int((ok.max_gap_days > 3).sum())}")
    log(f"Symbols with missing months: {int((ok.n_missing_months > 0).sum())}")
    log(f"Symbols flagged stablecoin or fiat base: {int(manifest.stable_or_fiat_base.sum())}")
    for p in paths.values():
        log(f"Saved {p}")
    return paths


def main():
    here = Path(__file__).resolve().parent
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--out-dir", default=str(here))
    ap.add_argument("--tag", default=datetime.now(timezone.utc).strftime("%Y-%m-%d"))
    ap.add_argument("--symbols", default=None, help="comma-separated, e.g. BTCUSDT,ETHUSDT")
    ap.add_argument("--max-symbols", type=int, default=None)
    ap.add_argument("--start-year", type=int, default=2017)
    ap.add_argument("--workers", type=int, default=8)
    a = ap.parse_args()
    only = [s.strip().upper() for s in a.symbols.split(",")] if a.symbols else None
    run(a.out_dir, a.tag, only, a.max_symbols, a.start_year, a.workers)


if __name__ == "__main__":
    main()

# ---------------------------------------------------------------------------
# TDD RESULTS (pytest, 2026-10-10, SANDBOX run: Python 3.13.16, pytest 9.1.1)
# NOTE: sandbox, NOT the mlfinlab env. Fake HTTP only: NOT yet run against the real archive.
# The bucket-listing URL and pagination are from memory of Binance's documented layout and
# are unverified. Run with --max-symbols 5 first.
# $ cd portfolio ; pytest test_pull_binance_daily.py -v
#
# collecting ... collected 19 items
# test_pull_binance_daily.py::test_listing_page_parses_prefixes_keys_and_truncation PASSED [  5%]
# test_pull_binance_daily.py::test_list_all_follows_pagination_to_the_end PASSED [ 10%]
# test_pull_binance_daily.py::test_select_symbols_keeps_usdt_drops_leveraged_and_flags_stables PASSED [ 15%]
# test_pull_binance_daily.py::test_parse_reads_milliseconds_and_microseconds_to_the_same_dates PASSED [ 21%]
# test_pull_binance_daily.py::test_parse_skips_a_header_row PASSED         [ 26%]
# test_pull_binance_daily.py::test_parse_rejects_implausible_dates PASSED  [ 31%]
# test_pull_binance_daily.py::test_parse_rejects_a_corrupt_zip PASSED      [ 36%]
# test_pull_binance_daily.py::test_month_helpers PASSED                    [ 42%]
# test_pull_binance_daily.py::test_fetch_symbol_joins_months_and_reports_missing_ones PASSED [ 47%]
# test_pull_binance_daily.py::test_fetch_symbol_retries_a_503_and_then_succeeds PASSED [ 52%]
# test_pull_binance_daily.py::test_fetch_symbol_gives_up_after_max_retries PASSED [ 57%]
# test_pull_binance_daily.py::test_start_year_filters_old_files PASSED     [ 63%]
# test_pull_binance_daily.py::test_overlapping_months_are_deduplicated_by_date PASSED [ 68%]
# test_pull_binance_daily.py::test_summary_known_values PASSED             [ 73%]
# test_pull_binance_daily.py::test_run_writes_three_files_with_aligned_columns PASSED [ 78%]
# test_pull_binance_daily.py::test_run_refuses_to_overwrite_before_any_network_call PASSED [ 84%]
# test_pull_binance_daily.py::test_run_records_a_failing_symbol_and_continues PASSED [ 89%]
# test_pull_binance_daily.py::test_run_limits_symbols_and_rejects_unknown_ones PASSED [ 94%]
# test_pull_binance_daily.py::test_run_with_no_data_raises_but_still_writes_the_manifest PASSED [100%]
# ============================== 19 passed in 0.45s ==============================
# ---------------------------------------------------------------------------
