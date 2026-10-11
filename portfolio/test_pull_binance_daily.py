"""Tests for pull_binance_daily.py. Fake HTTP only: no network. Run from the portfolio folder:
    pytest test_pull_binance_daily.py -v
"""
import io
import zipfile

import pandas as pd
import pytest

import pull_binance_daily as pb

D2024 = 1704067200000          # 2024-01-01 00:00 UTC in milliseconds
D2025_US = 1735689600000000    # 2025-01-01 00:00 UTC in microseconds
DAY_MS = 86_400_000


def kline_csv(rows, header=False):
    """rows: (open_time, close, quote_volume, trades). Other columns are filler."""
    lines = []
    if header:
        lines.append(",".join(pb.KLINE_COLUMNS))
    for t, c, qv, n in rows:
        lines.append(f"{t},{c},{c},{c},{c},10,{t},{qv},{n},1,1,0")
    return "\n".join(lines) + "\n"


def make_zip(csv_text, name="x.csv"):
    buf = io.BytesIO()
    with zipfile.ZipFile(buf, "w") as z:
        z.writestr(name, csv_text)
    return buf.getvalue()


def listing_xml(prefixes=(), keys=(), truncated=False, next_marker=None):
    ns = 'xmlns="http://s3.amazonaws.com/doc/2006-03-01/"'
    parts = [f"<ListBucketResult {ns}>", f"<IsTruncated>{'true' if truncated else 'false'}</IsTruncated>"]
    if next_marker:
        parts.append(f"<NextMarker>{next_marker}</NextMarker>")
    parts += [f"<Contents><Key>{k}</Key></Contents>" for k in keys]
    parts += [f"<CommonPrefixes><Prefix>{p}</Prefix></CommonPrefixes>" for p in prefixes]
    parts.append("</ListBucketResult>")
    return "".join(parts)


class FakeResponse:
    def __init__(self, status=200, text="", content=b""):
        self.status_code, self.text, self.content = status, text, content


class FakeArchive:
    """Behaves like the bucket listing plus the zip files. `files` maps key -> zip bytes."""
    def __init__(self, files, fail_first=None, page_size=1000, boom=()):
        self.files, self.calls, self.fail_first = dict(files), [], dict(fail_first or {})
        self.page_size, self.boom = page_size, set(boom)

    def get(self, url, params=None, timeout=None):
        self.calls.append((url, dict(params or {})))
        if url == pb.BUCKET_LISTING:
            prefix, delim, marker = params["prefix"], params.get("delimiter"), params.get("marker")
            if delim:
                syms = sorted({k[len(prefix):].split("/")[0] for k in self.files if k.startswith(prefix)})
                items = [prefix + s + "/" for s in syms]
                start = items.index(marker) + 1 if marker in items else 0
                page = items[start:start + self.page_size]
                more = start + self.page_size < len(items)
                return FakeResponse(200, listing_xml(prefixes=page, truncated=more, next_marker=page[-1] if more else None))
            keys = sorted(k for k in self.files if k.startswith(prefix))
            return FakeResponse(200, listing_xml(keys=keys))
        key = url[len(pb.ARCHIVE) + 1:]
        if any(b in key for b in self.boom):
            return FakeResponse(404)
        if self.fail_first.get(key, 0) > 0:
            self.fail_first[key] -= 1
            return FakeResponse(503)
        return FakeResponse(200, content=self.files[key]) if key in self.files else FakeResponse(404)


def K(sym, month):
    return f"{pb.KLINE_PREFIX}{sym}/1d/{sym}-1d-{month}.zip"


def three_days(start=D2024, close=100.0):
    return make_zip(kline_csv([(start + i * DAY_MS, close + i, 1000.0 * (i + 1), 5 + i) for i in range(3)]))


# ------------------------------------------------------------------ listing
def test_listing_page_parses_prefixes_keys_and_truncation():
    xml = listing_xml(prefixes=["a/BTCUSDT/"], keys=["a/k.zip"], truncated=True, next_marker="a/BTCUSDT/")
    prefixes, keys, nxt = pb.parse_listing_xml(xml)
    assert prefixes == ["a/BTCUSDT/"] and keys == ["a/k.zip"] and nxt == "a/BTCUSDT/"
    assert pb.parse_listing_xml(listing_xml())[2] is None


def test_list_all_follows_pagination_to_the_end():
    files = {K(s, "2024-01"): b"" for s in ["AUSDT", "BUSDT", "CUSDT", "DUSDT", "EUSDT"]}
    api = FakeArchive(files, page_size=2)
    prefixes, _ = pb.list_all(api, pb.KLINE_PREFIX, "/", sleep=lambda s: None)
    assert [p.split("/")[-2] for p in prefixes] == ["AUSDT", "BUSDT", "CUSDT", "DUSDT", "EUSDT"]
    assert len(api.calls) == 3


def test_select_symbols_keeps_usdt_drops_leveraged_and_flags_stables():
    prefixes = [pb.KLINE_PREFIX + s + "/" for s in
                ["BTCUSDT", "BTCUPUSDT", "ETHDOWNUSDT", "JUPUSDT", "SYRUPUSDT", "USDCUSDT", "ETHBTC", "EURUSDT"]]
    # ETH and BTC bases exist only if listed; add ETH so ETHDOWN is recognised as leveraged
    prefixes.append(pb.KLINE_PREFIX + "ETHUSDT/")
    got = {d["symbol"]: d for d in pb.select_symbols(prefixes)}
    assert "BTCUPUSDT" not in got and "ETHDOWNUSDT" not in got          # leveraged tokens dropped
    assert "JUPUSDT" in got and "SYRUPUSDT" in got                       # real coins that end in UP stay
    assert "ETHBTC" not in got                                           # not a USDT pair
    assert got["USDCUSDT"]["stable_or_fiat_base"] and got["EURUSDT"]["stable_or_fiat_base"]
    assert not got["BTCUSDT"]["stable_or_fiat_base"]


# ------------------------------------------------------------------- parsing
def test_parse_reads_milliseconds_and_microseconds_to_the_same_dates():
    ms = pb.parse_kline_zip(make_zip(kline_csv([(D2024, 5, 7.5, 3)])))
    us = pb.parse_kline_zip(make_zip(kline_csv([(D2025_US, 5, 7.5, 3)])))
    assert ms["date"].iloc[0] == pd.Timestamp("2024-01-01")
    assert us["date"].iloc[0] == pd.Timestamp("2025-01-01")             # unit read correctly
    assert ms["close"].iloc[0] == 5 and ms["quote_volume"].iloc[0] == 7.5 and ms["trades"].iloc[0] == 3


def test_parse_skips_a_header_row():
    df = pb.parse_kline_zip(make_zip(kline_csv([(D2024, 5, 1, 1), (D2024 + DAY_MS, 6, 1, 1)], header=True)))
    assert len(df) == 2 and df["close"].tolist() == [5, 6]


def test_parse_rejects_implausible_dates():
    with pytest.raises(ValueError, match="dates outside"):
        pb.parse_kline_zip(make_zip(kline_csv([(1000, 5, 1, 1)])), "XUSDT")   # 1970


def test_parse_rejects_a_corrupt_zip():
    blob = bytearray(three_days())
    blob[len(blob) // 2] ^= 0xFF
    with pytest.raises(Exception):
        pb.parse_kline_zip(bytes(blob), "XUSDT")


def test_month_helpers():
    assert pb.month_of("a/BTCUSDT-1d-2024-03.zip") == "2024-03"
    assert pb.month_of("a/BTCUSDT-1d-2024-03.zip.CHECKSUM") is None
    assert pb.missing_months(["2024-01", "2024-02", "2024-05"]) == ["2024-03", "2024-04"]
    assert pb.missing_months([]) == []


# ---------------------------------------------------------------- per-symbol
def test_fetch_symbol_joins_months_and_reports_missing_ones():
    files = {K("AUSDT", "2024-01"): three_days(D2024), K("AUSDT", "2024-03"): three_days(D2024 + 60 * DAY_MS)}
    df, missing, n = pb.fetch_symbol(FakeArchive(files), "AUSDT", sleep=lambda s: None)
    assert len(df) == 6 and n == 2 and missing == ["2024-02"]
    assert df["date"].is_monotonic_increasing


def test_fetch_symbol_retries_a_503_and_then_succeeds():
    key = K("AUSDT", "2024-01")
    api = FakeArchive({key: three_days()}, fail_first={key: 2})
    sleeps = []
    df, _, _ = pb.fetch_symbol(api, "AUSDT", sleep=sleeps.append, backoff=2.0)
    assert len(df) == 3 and sleeps == [2.0, 4.0]


def test_fetch_symbol_gives_up_after_max_retries():
    key = K("AUSDT", "2024-01")
    api = FakeArchive({key: three_days()}, fail_first={key: 99})
    with pytest.raises(RuntimeError, match="503"):
        pb.fetch_symbol(api, "AUSDT", sleep=lambda s: None, max_retries=2)
    assert sum(1 for u, _ in api.calls if u.endswith("2024-01.zip")) == 3


def test_start_year_filters_old_files():
    files = {K("AUSDT", "2018-05"): three_days(), K("AUSDT", "2024-01"): three_days()}
    _, _, n = pb.fetch_symbol(FakeArchive(files), "AUSDT", start_year=2020, sleep=lambda s: None)
    assert n == 1


def test_overlapping_months_are_deduplicated_by_date():
    files = {K("AUSDT", "2024-01"): three_days(D2024, 100), K("AUSDT", "2024-02"): three_days(D2024 + 2 * DAY_MS, 200)}
    df, _, _ = pb.fetch_symbol(FakeArchive(files), "AUSDT", sleep=lambda s: None)
    assert len(df) == 5 and df["date"].is_unique


def test_summary_known_values():
    df = pb.parse_kline_zip(three_days())
    s = pb.summarize_symbol(df, ["2024-02"], 2)
    assert s["n_rows"] == 3 and s["first_date"] == "2024-01-01" and s["last_date"] == "2024-01-03"
    assert s["max_gap_days"] == 1 and s["median_dollar_volume"] == 2000.0 and s["total_trades"] == 5 + 6 + 7
    assert s["n_missing_months"] == 1 and s["n_nonpositive_close"] == 0


# ----------------------------------------------------------------------- run
def two_symbol_archive():
    return FakeArchive({K("AUSDT", "2024-01"): three_days(D2024, 100), K("BUSDT", "2024-01"): three_days(D2024 + DAY_MS, 50),
                        K("BTCUPUSDT", "2024-01"): three_days(), K("BTCUSDT", "2024-01"): three_days()})


def test_run_writes_three_files_with_aligned_columns(tmp_path):
    paths = pb.run(tmp_path, "t", session_factory=lambda: two_symbol_archive(), workers=2, sleep=lambda s: None, log=lambda *a: None)
    wide = pd.read_csv(paths["prices"], index_col=0)
    assert set(wide.columns) == {"AUSDT", "BUSDT", "BTCUSDT"}                     # BTCUP dropped
    assert pd.isna(wide.loc["2024-01-01", "BUSDT"])                                # B starts a day later
    man = pd.read_csv(paths["manifest"]).set_index("symbol")
    assert man.loc["AUSDT", "status"] == "ok" and man.loc["AUSDT", "n_rows"] == 3
    long = pd.read_csv(paths["ohlcv"])
    assert list(long.columns) == ["date", "symbol", "open", "high", "low", "close", "volume", "quote_volume", "trades"]


def test_run_refuses_to_overwrite_before_any_network_call(tmp_path):
    (tmp_path / "binance_prices_daily_asof_t.csv").write_text("x")
    api = two_symbol_archive()
    with pytest.raises(FileExistsError):
        pb.run(tmp_path, "t", session_factory=lambda: api, log=lambda *a: None)
    assert api.calls == []


def test_run_records_a_failing_symbol_and_continues(tmp_path):
    api = FakeArchive({K("AUSDT", "2024-01"): three_days(), K("BUSDT", "2024-01"): b"not a zip"})
    paths = pb.run(tmp_path, "t", session_factory=lambda: api, workers=1, sleep=lambda s: None, log=lambda *a: None)
    man = pd.read_csv(paths["manifest"]).set_index("symbol")
    assert man.loc["AUSDT", "status"] == "ok" and man.loc["BUSDT", "status"] == "error"
    assert "BUSDT" not in pd.read_csv(paths["prices"], index_col=0).columns


def test_run_limits_symbols_and_rejects_unknown_ones(tmp_path):
    api = two_symbol_archive()
    paths = pb.run(tmp_path, "a", only_symbols=["AUSDT"], session_factory=lambda: api, workers=1, sleep=lambda s: None, log=lambda *a: None)
    assert list(pd.read_csv(paths["prices"], index_col=0).columns) == ["AUSDT"]
    with pytest.raises(ValueError, match="not USDT spot symbols"):
        pb.run(tmp_path, "b", only_symbols=["ZZZUSDT"], session_factory=lambda: api, log=lambda *a: None)
    paths = pb.run(tmp_path, "c", max_symbols=1, session_factory=lambda: api, workers=1, sleep=lambda s: None, log=lambda *a: None)
    assert pd.read_csv(paths["prices"], index_col=0).shape[1] == 1


def test_run_with_no_data_raises_but_still_writes_the_manifest(tmp_path):
    api = FakeArchive({K("AUSDT", "2024-01"): b"junk"})
    with pytest.raises(RuntimeError, match="No symbol returned data"):
        pb.run(tmp_path, "t", session_factory=lambda: api, workers=1, sleep=lambda s: None, log=lambda *a: None)
    assert (tmp_path / "binance_pairs_manifest_asof_t.csv").exists()
    assert not (tmp_path / "binance_prices_daily_asof_t.csv").exists()

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
