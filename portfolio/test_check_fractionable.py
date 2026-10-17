import sys
from pathlib import Path
from types import SimpleNamespace

import pytest

from check_fractionable import check_assets, run_check, summarize
from pull_prices import load_tickers


def asset(symbol, status="active", tradable=True, fractionable=True, shortable=True,
          min_order_size=None, min_trade_increment=None):
    """A stand-in for alpaca.trading.models.Asset. Real status values are enums with a .value."""
    st = SimpleNamespace(value=status) if status is not None else None
    return SimpleNamespace(symbol=symbol, status=st, tradable=tradable, fractionable=fractionable,
                           shortable=shortable, min_order_size=min_order_size,
                           min_trade_increment=min_trade_increment)


def test_everything_tradable_and_fractionable_has_no_issues():
    df = check_assets([asset("SPY"), asset("QQQ")], ["SPY", "QQQ"])
    assert list(df["symbol"]) == ["SPY", "QQQ"] and (df["issue"] == "").all()
    assert df["found"].all() and df["fractionable"].all() and df["tradable"].all()
    assert summarize(df)["blockers"] == []


def test_a_ticker_alpaca_does_not_list_is_found_false_and_a_blocker():
    df = check_assets([asset("SPY")], ["SPY", "ZZZZ"])
    row = df[df["symbol"] == "ZZZZ"].iloc[0]
    assert not row["found"] and "not in" in row["issue"]
    assert summarize(df)["blockers"] == ["ZZZZ"]


def test_a_non_fractionable_etf_is_a_blocker_and_says_why():
    df = check_assets([asset("SPY"), asset("XYZ", fractionable=False)], ["SPY", "XYZ"])
    row = df[df["symbol"] == "XYZ"].iloc[0]
    assert "not fractionable" in row["issue"] and row["tradable"]
    assert summarize(df)["blockers"] == ["XYZ"]


def test_a_non_tradable_etf_is_a_blocker():
    df = check_assets([asset("XYZ", tradable=False)], ["XYZ"])
    assert "not tradable" in df.iloc[0]["issue"] and summarize(df)["blockers"] == ["XYZ"]


def test_an_inactive_status_is_a_blocker_even_if_flags_look_fine():
    df = check_assets([asset("XYZ", status="inactive")], ["XYZ"])
    assert "status inactive" in df.iloc[0]["issue"] and summarize(df)["blockers"] == ["XYZ"]


def test_two_problems_are_both_reported():
    df = check_assets([asset("XYZ", tradable=False, fractionable=False)], ["XYZ"])
    issue = df.iloc[0]["issue"]
    assert "not tradable" in issue and "not fractionable" in issue


def test_the_hedge_is_listed_but_never_a_blocker():
    df = check_assets([asset("SPY"), asset("SH", fractionable=False)], ["SPY", "SH"])
    sh = df[df["symbol"] == "SH"].iloc[0]
    assert sh["role"].startswith("hedge") and "not fractionable" in sh["issue"]
    s = summarize(df)
    assert s["blockers"] == [] and s["hedge_issues"] == ["SH"]
    assert s["n_ranked"] == 1 and s["n_ok"] == 1                    # the hedge is not counted among the ranked


def test_symbols_match_regardless_of_case_and_row_order_follows_the_ticker_list():
    df = check_assets([asset("qqq"), asset("SPY")], ["SPY", "QQQ"])
    assert list(df["symbol"]) == ["SPY", "QQQ"] and df["found"].all()


def test_a_lower_case_ticker_in_the_list_still_matches_alpaca_s_upper_case_symbol():
    df = check_assets([asset("SPY")], ["spy"])
    assert df.iloc[0]["found"] and df.iloc[0]["issue"] == ""


def test_missing_order_size_limits_show_as_empty_text_not_the_word_None():
    df = check_assets([asset("SPY")], ["SPY"])
    assert df.iloc[0]["min_order_size"] == "" and df.iloc[0]["min_trade_increment"] == ""


def test_shortable_and_order_size_limits_are_reported_but_never_block():
    a = asset("SPY", shortable=False, min_order_size="1", min_trade_increment="0.0001")
    df = check_assets([a], ["SPY"])
    assert df.iloc[0]["shortable"] is False or not df.iloc[0]["shortable"]
    assert df.iloc[0]["min_trade_increment"] == "0.0001" and summarize(df)["blockers"] == []


def test_summary_counts():
    df = check_assets([asset("A"), asset("B", fractionable=False), asset("C")], ["A", "B", "C", "D"])
    s = summarize(df)
    assert s["n_ranked"] == 4 and s["n_ok"] == 2 and s["blockers"] == ["B", "D"]


def test_the_real_universe_file_parses_to_48_ranked_etfs_plus_the_hedge():
    here = Path(__file__).resolve().parent
    tickers = load_tickers(here / "universe.txt")
    df = check_assets([asset(t) for t in tickers], tickers)
    s = summarize(df)
    assert s["n_ranked"] == 48 and s["n_ok"] == 48 and s["blockers"] == []
    assert list(df[df["role"].str.startswith("hedge")]["symbol"]) == ["SH"]


def test_run_check_only_asks_for_assets_once_and_returns_the_table():
    calls = []
    def fake_get_assets():
        calls.append(1)
        return [asset("SPY"), asset("QQQ", fractionable=False)]
    df = run_check(fake_get_assets, ["SPY", "QQQ"])
    assert len(calls) == 1 and summarize(df)["blockers"] == ["QQQ"]


def test_the_module_never_places_orders_and_imports_no_trading_client_at_load():
    src = (Path(__file__).resolve().parent / "check_fractionable.py").read_text(encoding="utf-8")
    for forbidden in ("MarketOrderRequest", "LimitOrderRequest", "submit_order", "cancel_order", "close_position"):
        assert forbidden not in src
    import importlib
    for m in [m for m in sys.modules if m.startswith("alpaca")]:
        del sys.modules[m]
    importlib.reload(sys.modules["check_fractionable"])
    assert not [m for m in sys.modules if m.startswith("alpaca")]


# ------------------------------------------------------------------ main(), with a fake alpaca-py
def _install_fake_alpaca(monkeypatch, assets):
    """Register fake alpaca modules; return a log of what main() asked them to do."""
    import types
    log = {"client_kwargs": None, "calls": [], "request_kwargs": None}

    class FakeClient:
        def __init__(self, key, secret, paper=False):
            log["client_kwargs"] = dict(key=key, secret=secret, paper=paper)

        def get_all_assets(self, request):
            log["calls"].append("get_all_assets")
            return assets

        def __getattr__(self, name):                       # any other method (an order, say) is a test failure
            raise AssertionError(f"main() called client.{name}(), which a read-only check must never do")

    class FakeRequest:
        def __init__(self, **kw):
            log["request_kwargs"] = kw

    enums = types.SimpleNamespace(AssetClass=types.SimpleNamespace(US_EQUITY="us_equity"),
                                  AssetStatus=types.SimpleNamespace(ACTIVE="active"))
    for name, attrs in {"alpaca": {}, "alpaca.trading": {},
                        "alpaca.trading.client": {"TradingClient": FakeClient},
                        "alpaca.trading.enums": vars(enums),
                        "alpaca.trading.requests": {"GetAssetsRequest": FakeRequest}}.items():
        mod = types.ModuleType(name)
        mod.__dict__.update(attrs)
        monkeypatch.setitem(sys.modules, name, mod)
    return log


def _run_main(monkeypatch, tmp_path, universe_text, *extra):
    import check_fractionable
    uni = tmp_path / "universe.txt"
    uni.write_text(universe_text, encoding="utf-8")
    monkeypatch.setattr(sys, "argv", ["check_fractionable.py", "--universe", str(uni), *extra])
    with pytest.raises(SystemExit) as e:
        check_fractionable.main()
    return e.value.code


def test_main_exits_2_without_keys_and_never_touches_alpaca(monkeypatch, tmp_path, capsys):
    monkeypatch.delenv("APCA_API_KEY_ID", raising=False)
    monkeypatch.delenv("APCA_API_SECRET_KEY", raising=False)
    log = _install_fake_alpaca(monkeypatch, [])
    assert _run_main(monkeypatch, tmp_path, "SPY\n") == 2
    assert log["client_kwargs"] is None and "APCA_API_KEY_ID" in capsys.readouterr().out


def test_main_uses_the_paper_api_asks_once_and_exits_0_when_everything_is_fine(monkeypatch, tmp_path, capsys):
    monkeypatch.setenv("APCA_API_KEY_ID", "k")
    monkeypatch.setenv("APCA_API_SECRET_KEY", "s")
    log = _install_fake_alpaca(monkeypatch, [asset("SPY"), asset("QQQ")])
    assert _run_main(monkeypatch, tmp_path, "SPY\nQQQ  # comment\n") == 0
    assert log["client_kwargs"]["paper"] is True and log["calls"] == ["get_all_assets"]
    assert log["request_kwargs"] == {"status": "active", "asset_class": "us_equity"}
    assert "2 of 2 ranked ETFs" in capsys.readouterr().out


def test_main_exits_1_and_names_the_blocker_and_can_save_the_table(monkeypatch, tmp_path, capsys):
    monkeypatch.setenv("APCA_API_KEY_ID", "k")
    monkeypatch.setenv("APCA_API_SECRET_KEY", "s")
    _install_fake_alpaca(monkeypatch, [asset("SPY"), asset("QQQ", fractionable=False)])
    out_csv = tmp_path / "table.csv"
    assert _run_main(monkeypatch, tmp_path, "SPY\nQQQ\n", "--out", str(out_csv)) == 1
    assert "BLOCKERS: QQQ" in capsys.readouterr().out
    import pandas as pd
    saved = pd.read_csv(out_csv)
    assert list(saved["symbol"]) == ["SPY", "QQQ"] and "not fractionable" in saved.loc[1, "issue"]

# ---------------------------------------------------------------------------
# TDD RESULTS (pytest, 2026-10-17; sandbox run: Linux, Python 3.10.20, pandas 1.5.3, numpy 1.23.5,
# same versions as the mlfinlab env; alpaca-py is faked, so nothing here touched Alpaca)
# $ cd portfolio ; pytest test_check_fractionable.py -v
#
# collected 18 items
# test_check_fractionable.py::test_everything_tradable_and_fractionable_has_no_issues PASSED [  5%]
# test_check_fractionable.py::test_a_ticker_alpaca_does_not_list_is_found_false_and_a_blocker PASSED [ 11%]
# test_check_fractionable.py::test_a_non_fractionable_etf_is_a_blocker_and_says_why PASSED [ 16%]
# test_check_fractionable.py::test_a_non_tradable_etf_is_a_blocker PASSED  [ 22%]
# test_check_fractionable.py::test_an_inactive_status_is_a_blocker_even_if_flags_look_fine PASSED [ 27%]
# test_check_fractionable.py::test_two_problems_are_both_reported PASSED   [ 33%]
# test_check_fractionable.py::test_the_hedge_is_listed_but_never_a_blocker PASSED [ 38%]
# test_check_fractionable.py::test_symbols_match_regardless_of_case_and_row_order_follows_the_ticker_list PASSED [ 44%]
# test_check_fractionable.py::test_a_lower_case_ticker_in_the_list_still_matches_alpaca_s_upper_case_symbol PASSED [ 50%]
# test_check_fractionable.py::test_missing_order_size_limits_show_as_empty_text_not_the_word_None PASSED [ 55%]
# test_check_fractionable.py::test_shortable_and_order_size_limits_are_reported_but_never_block PASSED [ 61%]
# test_check_fractionable.py::test_summary_counts PASSED                   [ 66%]
# test_check_fractionable.py::test_the_real_universe_file_parses_to_48_ranked_etfs_plus_the_hedge PASSED [ 72%]
# test_check_fractionable.py::test_run_check_only_asks_for_assets_once_and_returns_the_table PASSED [ 77%]
# test_check_fractionable.py::test_the_module_never_places_orders_and_imports_no_trading_client_at_load PASSED [ 83%]
# test_check_fractionable.py::test_main_exits_2_without_keys_and_never_touches_alpaca PASSED [ 88%]
# test_check_fractionable.py::test_main_uses_the_paper_api_asks_once_and_exits_0_when_everything_is_fine PASSED [ 94%]
# test_check_fractionable.py::test_main_exits_1_and_names_the_blocker_and_can_save_the_table PASSED [100%]
# ============================== 18 passed in 0.27s ==============================
#
# Mutation checks (sandbox, 13): fractionable check removed, tradable check removed, status check removed,
#   case-sensitive lookup, hedge counted as ranked, hedge not recognised, missing ticker not flagged, live instead of
#   paper API, exit code always 0, missing keys not fatal, --out ignored, asset-class filter dropped, blank order size
#   shown as 'None' each made a test fail; original restored. Two survived the first pass (case-sensitive lookup, 'None'
#   text), so one test for each was added.
# NOT YET RUN against live Alpaca: the real answer for the 48 ETFs comes from your paper-key run of check_fractionable.py.
# ---------------------------------------------------------------------------
