import pytest
import requests

from pull_crypto_daily import (
    build_wide_close,
    check_outputs_free,
    fetch_daily_candles,
    output_paths,
    parse_ohlc_rows,
    run,
    select_usd_pairs,
    summarize_pair,
)

# 2025-01-01 00:00:00 UTC. Kraken daily candles open at UTC midnight.
T0 = 1735689600
DAY = 86400


def row(day, close="100.0", vwap="100.0", volume="2.5", trades=7):
    """One Kraken OHLC row: [time, open, high, low, close, vwap, volume, count]."""
    return [T0 + day * DAY, "99.0", "101.0", "98.0", close, vwap, volume, trades]


class FakeResponse:
    def __init__(self, payload=None, status=200):
        self.status_code = status
        self._payload = payload if payload is not None else {}

    def json(self):
        return self._payload

    def raise_for_status(self):
        if self.status_code >= 400:
            raise requests.HTTPError(f"HTTP {self.status_code}")


class FakeSession:
    """Routes AssetPairs and OHLC calls to canned answers and counts every call."""

    def __init__(self, asset_pairs=None, ohlc_by_pair=None, ohlc_queue=None):
        self.asset_pairs = asset_pairs or {}
        self.ohlc_by_pair = ohlc_by_pair or {}
        self.ohlc_queue = list(ohlc_queue or [])
        self.calls = 0

    def get(self, url, params=None, timeout=None):
        self.calls += 1
        if url.endswith("/AssetPairs"):
            return FakeResponse({"error": [], "result": self.asset_pairs})
        if self.ohlc_queue:
            item = self.ohlc_queue.pop(0)
            return item if isinstance(item, FakeResponse) else FakeResponse(item)
        return FakeResponse(self.ohlc_by_pair[params["pair"]])


def ok_payload(key, rows):
    return {"error": [], "result": {key: rows, "last": T0}}


def no_sleep(_seconds):
    return None


# ---------------------------------------------------------------- pair selection

def test_select_usd_pairs_keeps_online_usd_quoted_pairs_only():
    asset_pairs = {
        "XXBTZUSD": {"altname": "XBTUSD", "wsname": "XBT/USD", "base": "XXBT", "quote": "ZUSD", "status": "online"},
        "XETHZUSD": {"altname": "ETHUSD", "wsname": "ETH/USD", "base": "XETH", "quote": "ZUSD", "status": "online"},
        "XXBTZEUR": {"altname": "XBTEUR", "wsname": "XBT/EUR", "base": "XXBT", "quote": "ZEUR", "status": "online"},
        "XXBTZUSD.d": {"altname": "XBTUSD.d", "wsname": "XBT/USD", "base": "XXBT", "quote": "ZUSD", "status": "online"},
        "TAOUSD": {"altname": "TAOUSD", "wsname": "TAO/USD", "base": "TAO", "quote": "ZUSD", "status": "post_only"},
        "SOLUSD": {"altname": "SOLUSD", "wsname": "SOL/USD", "base": "SOL", "quote": "ZUSD"},
        "USDTZUSD": {"altname": "USDTUSD", "wsname": "USDT/USD", "base": "USDT", "quote": "ZUSD", "status": "online"},
    }
    got = select_usd_pairs(asset_pairs)
    assert [p["pair"] for p in got] == ["ETHUSD", "SOLUSD", "USDTUSD", "XBTUSD"]
    by_pair = {p["pair"]: p for p in got}
    assert by_pair["XBTUSD"]["internal_name"] == "XXBTZUSD"
    assert by_pair["XBTUSD"]["base"] == "XXBT"


def test_stablecoin_and_fiat_bases_are_flagged_not_dropped():
    asset_pairs = {
        "USDTZUSD": {"altname": "USDTUSD", "base": "USDT", "quote": "ZUSD", "status": "online"},
        "ZEURZUSD": {"altname": "EURUSD", "base": "ZEUR", "quote": "ZUSD", "status": "online"},
        "XXBTZUSD": {"altname": "XBTUSD", "base": "XXBT", "quote": "ZUSD", "status": "online"},
    }
    flags = {p["pair"]: p["stable_or_fiat_base"] for p in select_usd_pairs(asset_pairs)}
    assert flags == {"EURUSD": True, "USDTUSD": True, "XBTUSD": False}


# ---------------------------------------------------------------- parsing

def test_parse_converts_types_and_drops_the_in_progress_candle():
    rows = [row(0), row(1), row(2)]
    now = T0 + 2 * DAY + 3600  # one hour into day 3, so day 3's candle is incomplete
    df, n_raw = parse_ohlc_rows(rows, now)
    assert n_raw == 3
    assert [d.strftime("%Y-%m-%d") for d in df["date"]] == ["2025-01-01", "2025-01-02"]
    assert df["close"].dtype.kind == "f" and df["trades"].dtype.kind == "i"
    assert df.loc[0, "dollar_volume"] == pytest.approx(250.0)  # vwap 100.0 x volume 2.5


def test_a_candle_that_has_just_closed_is_kept():
    df, n_raw = parse_ohlc_rows([row(0)], T0 + DAY)  # exactly at the close of day 1
    assert n_raw == 1 and len(df) == 1


def test_parse_sorts_and_removes_duplicate_times():
    rows = [row(2, close="3.0"), row(0, close="1.0"), row(1, close="2.0"), row(1, close="2.0")]
    df, n_raw = parse_ohlc_rows(rows, T0 + 10 * DAY)
    assert n_raw == 4
    assert list(df["close"]) == [1.0, 2.0, 3.0]


def test_parse_of_no_rows_gives_an_empty_frame():
    df, n_raw = parse_ohlc_rows([], T0 + 10 * DAY)
    assert n_raw == 0 and df.empty
    assert "dollar_volume" in df.columns


# ---------------------------------------------------------------- summaries

def test_summarize_pair_known_values():
    rows = [row(0, vwap="10.0", volume="1.0", trades=3),   # dollar volume 10
            row(1, vwap="10.0", volume="3.0", trades=4),   # 30
            row(4, vwap="10.0", volume="2.0", trades=5)]   # 20, after a 3-day gap
    df, n_raw = parse_ohlc_rows(rows, T0 + 10 * DAY)
    s = summarize_pair(df, n_raw)
    assert s["n_rows"] == 3
    assert s["first_date"] == "2025-01-01" and s["last_date"] == "2025-01-05"
    assert s["max_gap_days"] == 3
    assert s["median_dollar_volume"] == pytest.approx(20.0)
    assert s["total_trades"] == 12
    assert s["hit_cap"] is False and s["n_nonpositive_close"] == 0


def test_hit_cap_is_true_when_the_raw_response_has_720_candles():
    df, _ = parse_ohlc_rows([row(0)], T0 + 10 * DAY)
    assert summarize_pair(df, 720)["hit_cap"] is True
    assert summarize_pair(df, 719)["hit_cap"] is False


def test_nonpositive_closes_are_counted():
    df, n_raw = parse_ohlc_rows([row(0, close="0.0"), row(1)], T0 + 10 * DAY)
    assert summarize_pair(df, n_raw)["n_nonpositive_close"] == 1


def test_wide_close_leaves_nan_where_a_pair_has_no_candle():
    a, _ = parse_ohlc_rows([row(0, close="1.0"), row(1, close="2.0")], T0 + 10 * DAY)
    b, _ = parse_ohlc_rows([row(1, close="5.0")], T0 + 10 * DAY)
    wide = build_wide_close({"BBB": b, "AAA": a})
    assert list(wide.columns) == ["AAA", "BBB"]
    assert wide.loc["2025-01-01", "AAA"] == 1.0
    assert wide["BBB"].isna().tolist() == [True, False]
    assert wide.loc["2025-01-02", "BBB"] == 5.0


# ---------------------------------------------------------------- fetching and retries

def test_fetch_reads_the_result_key_that_is_not_last():
    payload = {"error": [], "result": {"last": T0, "XXBTZUSD": [row(0), row(1)]}}
    session = FakeSession(ohlc_queue=[payload])
    df, n_raw = fetch_daily_candles(session, "XBTUSD", T0 + 10 * DAY, sleep=no_sleep)
    assert n_raw == 2 and len(df) == 2


def test_fetch_retries_after_http_429_with_growing_backoff():
    sleeps = []
    session = FakeSession(ohlc_queue=[FakeResponse(status=429), FakeResponse(status=429),
                                      ok_payload("XXBTZUSD", [row(0)])])
    df, _ = fetch_daily_candles(session, "XBTUSD", T0 + 10 * DAY, sleep=sleeps.append,
                                retry_backoff_seconds=5.0)
    assert len(df) == 1
    assert session.calls == 3
    assert sleeps == [5.0, 10.0]


def test_fetch_retries_when_the_rate_limit_error_comes_inside_a_200_response():
    sleeps = []
    limited = {"error": ["EAPI:Rate limit exceeded"], "result": {}}
    session = FakeSession(ohlc_queue=[limited, ok_payload("XXBTZUSD", [row(0)])])
    df, _ = fetch_daily_candles(session, "XBTUSD", T0 + 10 * DAY, sleep=sleeps.append)
    assert len(df) == 1 and session.calls == 2 and len(sleeps) == 1


def test_fetch_gives_up_after_max_retries():
    session = FakeSession(ohlc_queue=[FakeResponse(status=429)] * 10)
    with pytest.raises(ValueError, match="rate-limited"):
        fetch_daily_candles(session, "XBTUSD", T0 + 10 * DAY, sleep=no_sleep, max_retries=2)
    assert session.calls == 3  # the first try plus two retries


def test_fetch_raises_on_an_ordinary_api_error_without_retrying():
    bad = {"error": ["EQuery:Unknown asset pair"], "result": {}}
    session = FakeSession(ohlc_queue=[bad])
    with pytest.raises(ValueError, match="Unknown asset pair"):
        fetch_daily_candles(session, "NOPEUSD", T0 + 10 * DAY, sleep=no_sleep)
    assert session.calls == 1


# ---------------------------------------------------------------- overwrite guard

def test_check_outputs_free_refuses_an_existing_file(tmp_path):
    existing = tmp_path / "a.csv"
    existing.write_text("x")
    with pytest.raises(FileExistsError):
        check_outputs_free([tmp_path / "new.csv", existing])
    check_outputs_free([tmp_path / "new.csv"])  # nothing exists, so no error


def test_run_refuses_to_overwrite_before_making_any_network_call(tmp_path):
    paths = output_paths(tmp_path, "2026-10-10")
    paths["manifest"].write_text("old run")
    session = FakeSession()
    with pytest.raises(FileExistsError):
        run(tmp_path, "2026-10-10", session=session, now_unix=T0 + 10 * DAY, sleep=no_sleep, log=lambda *_: None)
    assert session.calls == 0
    assert paths["manifest"].read_text() == "old run"


# ---------------------------------------------------------------- end to end

ASSET_PAIRS = {
    "XXBTZUSD": {"altname": "XBTUSD", "wsname": "XBT/USD", "base": "XXBT", "quote": "ZUSD", "status": "online"},
    "TAOUSD": {"altname": "TAOUSD", "wsname": "TAO/USD", "base": "TAO", "quote": "ZUSD", "status": "online"},
    "XXBTZEUR": {"altname": "XBTEUR", "wsname": "XBT/EUR", "base": "XXBT", "quote": "ZEUR", "status": "online"},
}


def test_run_writes_all_three_files_and_records_a_failed_pair(tmp_path):
    import pandas as pd

    now = T0 + 2 * DAY + 3600  # day 3 is still in progress
    session = FakeSession(
        asset_pairs=ASSET_PAIRS,
        ohlc_by_pair={
            "XBTUSD": ok_payload("XXBTZUSD", [row(0, close="10.0"), row(1, close="11.0"), row(2, close="12.0")]),
            "TAOUSD": {"error": ["EQuery:Unknown asset pair"], "result": {}},
        },
    )
    out = run(tmp_path, "2026-10-10", session=session, now_unix=now, sleep=no_sleep, log=lambda *_: None)

    wide = pd.read_csv(out["paths"]["prices"])
    assert list(wide.columns) == ["date", "XBTUSD"]
    assert wide["XBTUSD"].tolist() == [10.0, 11.0]  # the in-progress day is not included
    assert wide["date"].tolist() == ["2025-01-01", "2025-01-02"]

    long = pd.read_csv(out["paths"]["ohlcv"])
    assert len(long) == 2 and set(long["pair"]) == {"XBTUSD"}

    manifest = pd.read_csv(out["paths"]["manifest"]).set_index("pair")
    assert manifest.loc["XBTUSD", "status"] == "ok" and manifest.loc["XBTUSD", "n_rows"] == 2
    assert manifest.loc["TAOUSD", "status"] == "error"
    assert "Unknown asset pair" in manifest.loc["TAOUSD", "error"]
    assert "XBTEUR" not in manifest.index  # not USD-quoted


def test_run_can_be_limited_to_named_pairs_and_to_a_maximum(tmp_path):
    session = FakeSession(
        asset_pairs=ASSET_PAIRS,
        ohlc_by_pair={"XBTUSD": ok_payload("XXBTZUSD", [row(0)]), "TAOUSD": ok_payload("TAOUSD", [row(0)])},
    )
    out = run(tmp_path, "t1", only_pairs=["taousd"], session=session, now_unix=T0 + 10 * DAY,
              sleep=no_sleep, log=lambda *_: None)
    assert list(out["manifest"]["pair"]) == ["TAOUSD"]

    session2 = FakeSession(
        asset_pairs=ASSET_PAIRS,
        ohlc_by_pair={"XBTUSD": ok_payload("XXBTZUSD", [row(0)]), "TAOUSD": ok_payload("TAOUSD", [row(0)])},
    )
    out2 = run(tmp_path, "t2", max_pairs=1, session=session2, now_unix=T0 + 10 * DAY,
               sleep=no_sleep, log=lambda *_: None)
    assert list(out2["manifest"]["pair"]) == ["TAOUSD"]  # alphabetical, so TAOUSD comes first


def test_run_raises_when_no_pair_succeeds_but_still_writes_the_manifest(tmp_path):
    session = FakeSession(
        asset_pairs=ASSET_PAIRS,
        ohlc_by_pair={"XBTUSD": {"error": ["EGeneral:Internal error"], "result": {}},
                      "TAOUSD": {"error": ["EGeneral:Internal error"], "result": {}}},
    )
    with pytest.raises(RuntimeError, match="No pair"):
        run(tmp_path, "t3", session=session, now_unix=T0 + 10 * DAY, sleep=no_sleep, log=lambda *_: None)
    assert output_paths(tmp_path, "t3")["manifest"].exists()
    assert not output_paths(tmp_path, "t3")["prices"].exists()


def test_run_rejects_a_requested_pair_that_is_not_a_usd_pair(tmp_path):
    session = FakeSession(asset_pairs=ASSET_PAIRS)
    with pytest.raises(ValueError, match="XBTEUR"):
        run(tmp_path, "t4", only_pairs=["XBTEUR"], session=session, now_unix=T0 + 10 * DAY,
            sleep=no_sleep, log=lambda *_: None)

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
