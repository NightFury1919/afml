"""Pull daily candles for every USD-quoted pair on Kraken.

Research use only. This reads Kraken's public REST API, so no keys are needed.

Why daily candles and not raw trades: the cross-sectional test needs one price
per coin per day, and one candle request returns up to about 720 days. Pulling
raw trades for a hundred coins would take days (see ingestion_kraken.py).

Run from C:\\ws\\AFML\\portfolio in the mlfinlab env:
    python pull_crypto_daily.py
    python pull_crypto_daily.py --max-pairs 5               (quick smoke test)
    python pull_crypto_daily.py --pairs XBTUSD,ETHUSD,TAOUSD

Outputs (CSV only, so any environment can read them):
    crypto_prices_daily_asof_<tag>.csv    wide: one row per UTC date, one column per
                                          pair, close price (same shape as
                                          prices_daily_asof_*.csv, so panel_data can load it)
    crypto_ohlcv_daily_asof_<tag>.csv     long: one row per (date, pair) with open, high,
                                          low, close, vwap, volume, trades, dollar_volume
    crypto_pairs_manifest_asof_<tag>.csv  one row per pair: status, depth, gaps, liquidity
"""
import argparse
import time
from datetime import datetime, timezone
from pathlib import Path

import pandas as pd
import requests

BASE_URL = "https://api.kraken.com/0/public"
INTERVAL_MINUTES = 1440          # one candle per UTC day
DAY_SECONDS = 86400
CANDLE_CAP = 720                 # most candles one OHLC request returns (checked, not assumed: see hit_cap)

OHLC_COLUMNS = ["time", "open", "high", "low", "close", "vwap", "volume", "trades"]
FRAME_COLUMNS = ["date", "open", "high", "low", "close", "vwap", "volume", "trades", "dollar_volume"]

# LOAD-BEARING (2026-10-10): stablecoin and fiat bases are FLAGGED in the manifest,
# never dropped here. Whether they belong in the test universe is a design choice for
# the pre-registration, so this script keeps every USD pair and records the flag.
STABLE_OR_FIAT_BASES = {
    "USDT", "USDC", "DAI", "PYUSD", "TUSD", "USDG", "USDQ", "EURT", "EURR", "EURQ",
    "ZEUR", "ZGBP", "ZAUD", "ZCAD", "ZJPY", "ZUSD", "EUR", "GBP", "AUD", "CAD", "CHF", "JPY",
}


def select_usd_pairs(asset_pairs):
    """USD-quoted, online, non-darkpool pairs from Kraken's AssetPairs result.

    Returns a list of dicts sorted by pair name (Kraken's altname, e.g. XBTUSD).
    A pair with no status field is kept, since older entries may not carry one.
    """
    rows = []
    for internal, info in asset_pairs.items():
        if info.get("quote") not in ("ZUSD", "USD"):
            continue
        altname = info.get("altname") or internal
        if internal.endswith(".d") or altname.endswith(".d"):
            continue
        status = info.get("status")
        if status is not None and status != "online":
            continue
        base = info.get("base", "")
        rows.append({
            "pair": altname,
            "internal_name": internal,
            "base": base,
            "wsname": info.get("wsname", ""),
            "stable_or_fiat_base": base in STABLE_OR_FIAT_BASES,
        })
    rows.sort(key=lambda r: r["pair"])
    return rows


def parse_ohlc_rows(rows, now_unix):
    """Kraken OHLC rows -> (clean DataFrame, number of raw rows returned).

    LOAD-BEARING (2026-10-10): the last candle Kraken returns is the day still in
    progress. It is dropped, because using it would put a partly finished day into
    the data as if it were final (look-ahead). A candle is kept only when its day
    has fully closed: time + 86400 <= now.
    """
    df = pd.DataFrame(rows, columns=OHLC_COLUMNS)
    n_raw = len(df)
    df["time"] = df["time"].astype("int64")
    for c in ("open", "high", "low", "close", "vwap", "volume"):
        df[c] = df[c].astype(float)
    df["trades"] = df["trades"].astype("int64")
    df = df[df["time"] + DAY_SECONDS <= now_unix]
    df = df.drop_duplicates("time").sort_values("time").reset_index(drop=True)
    df["date"] = pd.to_datetime(df["time"], unit="s").dt.normalize()
    df["dollar_volume"] = df["vwap"] * df["volume"]
    return df[FRAME_COLUMNS], n_raw


def summarize_pair(df, n_raw):
    """Depth, gaps and liquidity for one pair's cleaned candles.

    max_gap_days is the longest calendar gap between consecutive candles (1 means
    no missing days). hit_cap is True when the raw response had the maximum number
    of candles, which means older history exists that this request did not return.
    """
    if df.empty:
        return {"n_rows": 0, "first_date": "", "last_date": "", "max_gap_days": 0,
                "median_dollar_volume": float("nan"), "total_trades": 0,
                "hit_cap": n_raw >= CANDLE_CAP, "n_nonpositive_close": 0}
    gaps = df["date"].diff().dt.days
    return {
        "n_rows": int(len(df)),
        "first_date": df["date"].iloc[0].strftime("%Y-%m-%d"),
        "last_date": df["date"].iloc[-1].strftime("%Y-%m-%d"),
        "max_gap_days": int(gaps.max()) if len(df) > 1 else 0,
        "median_dollar_volume": float(df["dollar_volume"].median()),
        "total_trades": int(df["trades"].sum()),
        "hit_cap": bool(n_raw >= CANDLE_CAP),
        "n_nonpositive_close": int((df["close"] <= 0).sum()),
    }


def build_wide_close(frames):
    """{pair: cleaned frame} -> wide close prices (date index, one column per pair).

    Nothing is filled: a pair only has values on dates it actually has a candle.
    """
    series = {p: f.set_index("date")["close"] for p, f in sorted(frames.items())}
    wide = pd.DataFrame(series).sort_index()
    wide.index.name = "date"
    return wide


def _get_json(session, url, params, sleep, backoff, max_retries):
    """GET with retries on HTTP 429 and on Kraken's rate-limit error in a 200 reply.

    Any other API error raises ValueError straight away, with no retry.
    """
    for attempt in range(max_retries + 1):
        resp = session.get(url, params=params, timeout=15)
        if resp.status_code == 429:
            limited = True
        else:
            resp.raise_for_status()
            data = resp.json()
            errors = data.get("error") or []
            limited = any("rate limit" in str(e).lower() or "too many requests" in str(e).lower()
                          for e in errors)
            if not limited:
                if errors:
                    raise ValueError(f"Kraken API returned errors: {errors}")
                return data
        if attempt == max_retries:
            raise ValueError(f"Kraken rate-limited {max_retries + 1} times in a row for {url}")
        sleep(backoff * (attempt + 1))


def fetch_daily_candles(session, pair, now_unix, sleep=time.sleep,
                        retry_backoff_seconds=5.0, max_retries=5):
    """One OHLC request for `pair`. Returns (cleaned frame, raw row count)."""
    data = _get_json(session, f"{BASE_URL}/OHLC",
                     {"pair": pair, "interval": INTERVAL_MINUTES},
                     sleep, retry_backoff_seconds, max_retries)
    result = data.get("result", {})
    keys = [k for k in result if k != "last"]  # Kraken keys the rows by its own pair name
    if not keys:
        raise ValueError(f"No candle data in Kraken response for {pair}")
    return parse_ohlc_rows(result[keys[0]], now_unix)


def output_paths(out_dir, tag):
    out_dir = Path(out_dir)
    return {
        "prices": out_dir / f"crypto_prices_daily_asof_{tag}.csv",
        "ohlcv": out_dir / f"crypto_ohlcv_daily_asof_{tag}.csv",
        "manifest": out_dir / f"crypto_pairs_manifest_asof_{tag}.csv",
    }


def check_outputs_free(paths):
    """Refuse to overwrite. LOAD-BEARING: this runs BEFORE any network call, so a
    rerun that would collide fails in a second instead of after a long pull."""
    for p in paths:
        if Path(p).exists():
            raise FileExistsError(f"{p} already exists -- refusing to overwrite. "
                                  "Use a different --tag or move the old file.")


def run(out_dir, tag, only_pairs=None, max_pairs=None, sleep_seconds=1.0,
        session=None, now_unix=None, sleep=time.sleep, log=print):
    paths = output_paths(out_dir, tag)
    check_outputs_free(paths.values())

    session = session or requests.Session()
    now_unix = time.time() if now_unix is None else now_unix

    data = _get_json(session, f"{BASE_URL}/AssetPairs", {}, sleep, 5.0, 5)
    pairs = select_usd_pairs(data.get("result", {}))
    if only_pairs:
        wanted = [p.strip().upper() for p in only_pairs if p.strip()]
        known = {p["pair"] for p in pairs}
        unknown = [p for p in wanted if p not in known]
        if unknown:
            raise ValueError(f"Not an online USD pair on Kraken: {', '.join(unknown)}")
        pairs = [p for p in pairs if p["pair"] in wanted]
    if max_pairs:
        pairs = pairs[:max_pairs]
    log(f"{len(pairs)} USD pairs to pull (about {len(pairs) * (sleep_seconds + 0.3) / 60:.0f} min)")

    frames, manifest_rows = {}, []
    for i, info in enumerate(pairs, 1):
        row = dict(info)
        try:
            df, n_raw = fetch_daily_candles(session, info["pair"], now_unix, sleep=sleep)
            row.update(summarize_pair(df, n_raw))
            row.update({"status": "ok" if len(df) else "empty", "error": ""})
            if len(df):
                frames[info["pair"]] = df
            log(f"[{i}/{len(pairs)}] {info['pair']}: {row['n_rows']} days "
                f"{row['first_date']} to {row['last_date']}")
        except (ValueError, requests.RequestException) as exc:
            row.update({"status": "error", "error": str(exc), "n_rows": 0})
            log(f"[{i}/{len(pairs)}] {info['pair']}: ERROR {exc}")
        manifest_rows.append(row)
        if i < len(pairs):
            sleep(sleep_seconds)

    manifest = pd.DataFrame(manifest_rows)
    manifest.to_csv(paths["manifest"], index=False)
    if not frames:
        raise RuntimeError(f"No pair returned data. Manifest written to {paths['manifest']}")

    wide = build_wide_close(frames)
    wide.to_csv(paths["prices"], index_label="date", date_format="%Y-%m-%d")
    long = pd.concat([f.assign(pair=p) for p, f in sorted(frames.items())], ignore_index=True)
    long = long[["date", "pair"] + [c for c in FRAME_COLUMNS if c != "date"]]
    long.to_csv(paths["ohlcv"], index=False, date_format="%Y-%m-%d")
    return {"paths": paths, "manifest": manifest, "wide": wide}


def main():
    here = Path(__file__).resolve().parent
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--out-dir", default=str(here))
    ap.add_argument("--tag", default=datetime.now(timezone.utc).strftime("%Y-%m-%d"),
                    help="goes in the file names; default is today's UTC date")
    ap.add_argument("--pairs", default="", help="comma-separated pair names, e.g. XBTUSD,ETHUSD")
    ap.add_argument("--max-pairs", type=int, default=0, help="stop after this many pairs (smoke test)")
    ap.add_argument("--sleep", type=float, default=1.0, help="seconds between requests")
    args = ap.parse_args()

    out = run(args.out_dir, args.tag,
              only_pairs=args.pairs.split(",") if args.pairs else None,
              max_pairs=args.max_pairs or None, sleep_seconds=args.sleep)

    m = out["manifest"]
    ok = m[m["status"] == "ok"]
    print(f"\n{len(ok)} pairs with data, {int((m['status'] == 'error').sum())} errors, "
          f"{int((m['status'] == 'empty').sum())} empty")
    print(f"Days of history per pair: min {int(ok['n_rows'].min())}, "
          f"median {int(ok['n_rows'].median())}, max {int(ok['n_rows'].max())}")
    print(f"Pairs that hit the {CANDLE_CAP}-candle cap (older history exists): {int(ok['hit_cap'].sum())}")
    print(f"Pairs with a gap longer than 3 days: {int((ok['max_gap_days'] > 3).sum())}")
    print(f"Pairs flagged stablecoin or fiat base: {int(ok['stable_or_fiat_base'].sum())}")
    for key, path in out["paths"].items():
        print(f"Saved {path}")


if __name__ == "__main__":
    main()

# ---------------------------------------------------------------------------
# TDD RESULTS (pytest, 2026-10-10, SANDBOX run: Python 3.13.16, pytest 9.1.1)
# NOTE: sandbox, NOT the mlfinlab env (py3.10.20 / pandas 1.5.3). Re-run there.
# $ cd portfolio ; pytest test_pull_crypto_daily.py -v
#
# platform linux -- Python 3.13.16, pytest-9.1.1, pluggy-1.6.0 -- /usr/bin/python3
# collecting ... collected 21 items
# test_pull_crypto_daily.py::test_select_usd_pairs_keeps_online_usd_quoted_pairs_only PASSED [  4%]
# test_pull_crypto_daily.py::test_stablecoin_and_fiat_bases_are_flagged_not_dropped PASSED [  9%]
# test_pull_crypto_daily.py::test_parse_converts_types_and_drops_the_in_progress_candle PASSED [ 14%]
# test_pull_crypto_daily.py::test_a_candle_that_has_just_closed_is_kept PASSED [ 19%]
# test_pull_crypto_daily.py::test_parse_sorts_and_removes_duplicate_times PASSED [ 23%]
# test_pull_crypto_daily.py::test_parse_of_no_rows_gives_an_empty_frame PASSED [ 28%]
# test_pull_crypto_daily.py::test_summarize_pair_known_values PASSED       [ 33%]
# test_pull_crypto_daily.py::test_hit_cap_is_true_when_the_raw_response_has_720_candles PASSED [ 38%]
# test_pull_crypto_daily.py::test_nonpositive_closes_are_counted PASSED    [ 42%]
# test_pull_crypto_daily.py::test_wide_close_leaves_nan_where_a_pair_has_no_candle PASSED [ 47%]
# test_pull_crypto_daily.py::test_fetch_reads_the_result_key_that_is_not_last PASSED [ 52%]
# test_pull_crypto_daily.py::test_fetch_retries_after_http_429_with_growing_backoff PASSED [ 57%]
# test_pull_crypto_daily.py::test_fetch_retries_when_the_rate_limit_error_comes_inside_a_200_response PASSED [ 61%]
# test_pull_crypto_daily.py::test_fetch_gives_up_after_max_retries PASSED  [ 66%]
# test_pull_crypto_daily.py::test_fetch_raises_on_an_ordinary_api_error_without_retrying PASSED [ 71%]
# test_pull_crypto_daily.py::test_check_outputs_free_refuses_an_existing_file PASSED [ 76%]
# test_pull_crypto_daily.py::test_run_refuses_to_overwrite_before_making_any_network_call PASSED [ 80%]
# test_pull_crypto_daily.py::test_run_writes_all_three_files_and_records_a_failed_pair PASSED [ 85%]
# test_pull_crypto_daily.py::test_run_can_be_limited_to_named_pairs_and_to_a_maximum PASSED [ 90%]
# test_pull_crypto_daily.py::test_run_raises_when_no_pair_succeeds_but_still_writes_the_manifest PASSED [ 95%]
# test_pull_crypto_daily.py::test_run_rejects_a_requested_pair_that_is_not_a_usd_pair PASSED [100%]
# ============================== 21 passed in 0.42s ==============================
# ---------------------------------------------------------------------------
