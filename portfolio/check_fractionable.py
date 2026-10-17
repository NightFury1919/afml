"""Read-only check: can every ETF in the universe be bought in fractional dollars on Alpaca?

Why: at $1,000 split into two sleeves, the QF sleeve holds 48 ETFs at about $10 each, which only works if
every one of them is fractionable. Whole shares would be impossible for most of them (paper_protocol_v2.md).

Run from C:\\ws\\AFML\\portfolio in the mlfinlab env, with your PAPER keys set as in check_account.py:
    python check_fractionable.py
    python check_fractionable.py --out fractionable_check_asof_2026-10-17.csv

It makes ONE read-only request (the list of active US-equity assets) and compares it with universe.txt.
It never places, changes or cancels an order, and the alpaca imports happen inside main() so the tests need no
Alpaca install. Exit code 0 = all 48 ranked ETFs are active, tradable and fractionable; 1 = at least one is not.

What counts as a blocker (ranked ETFs only): not in Alpaca's active US-equity list, status not "active",
not tradable, or not fractionable. The hedge SH is listed but is never a blocker, because it is never bought.
`shortable`, `min_order_size` and `min_trade_increment` are shown for information only. Fractional order
minimums are Alpaca's rule and are not checked here (paper_runner_v2 sends nothing below $1).
"""
import argparse
import os
import sys
from pathlib import Path

import pandas as pd

from panel_data import HEDGE_TICKERS
from pull_prices import load_tickers

COLUMNS = ["symbol", "role", "found", "status", "tradable", "fractionable", "shortable",
           "min_order_size", "min_trade_increment", "issue"]


def _text(x):
    """Alpaca returns enums for status; tests pass plain strings. Return lower-case text or ''."""
    if x is None:
        return ""
    return str(getattr(x, "value", x)).lower()


def _blank(x):
    return "" if x is None else str(x)


def check_assets(assets, tickers, hedge=HEDGE_TICKERS):
    """One row per ticker, in the order given. `assets`: objects with symbol, status, tradable,
    fractionable, shortable (and optionally min_order_size, min_trade_increment)."""
    by_symbol = {str(a.symbol).upper(): a for a in assets}
    rows = []
    for t in tickers:
        role = "hedge (never bought)" if t in hedge else "ranked"
        a = by_symbol.get(t.upper())
        if a is None:
            rows.append(dict(symbol=t, role=role, found=False, status="", tradable=False, fractionable=False,
                             shortable=False, min_order_size="", min_trade_increment="",
                             issue="not in Alpaca's active US-equity list"))
            continue
        status = _text(a.status)
        problems = []
        if status != "active":
            problems.append(f"status {status or 'unknown'}")
        if not a.tradable:
            problems.append("not tradable")
        if not a.fractionable:
            problems.append("not fractionable")
        rows.append(dict(symbol=t, role=role, found=True, status=status, tradable=bool(a.tradable),
                         fractionable=bool(a.fractionable), shortable=bool(a.shortable),
                         min_order_size=_blank(getattr(a, "min_order_size", None)),
                         min_trade_increment=_blank(getattr(a, "min_trade_increment", None)),
                         issue="; ".join(problems)))
    return pd.DataFrame(rows, columns=COLUMNS)


def summarize(df):
    is_hedge = df["role"].str.startswith("hedge")
    ranked, hedge = df[~is_hedge], df[is_hedge]
    return dict(n_ranked=int(len(ranked)), n_ok=int((ranked["issue"] == "").sum()),
                blockers=list(ranked.loc[ranked["issue"] != "", "symbol"]),
                hedge_issues=list(hedge.loc[hedge["issue"] != "", "symbol"]))


def run_check(get_assets, tickers):
    """`get_assets` is called once and returns the active US-equity assets."""
    return check_assets(get_assets(), tickers)


def main():
    here = Path(__file__).resolve().parent
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--universe", default=str(here / "universe.txt"))
    ap.add_argument("--out", default=None, help="optional CSV of the full table")
    args = ap.parse_args()

    keys = [os.environ.get("APCA_API_KEY_ID"), os.environ.get("APCA_API_SECRET_KEY")]
    if not all(keys):
        print("Set APCA_API_KEY_ID and APCA_API_SECRET_KEY (paper keys), as check_account.py does.")
        sys.exit(2)

    from alpaca.trading.client import TradingClient           # imported here so the tests do not need alpaca-py
    from alpaca.trading.enums import AssetClass, AssetStatus
    from alpaca.trading.requests import GetAssetsRequest

    client = TradingClient(keys[0], keys[1], paper=True)
    request = GetAssetsRequest(status=AssetStatus.ACTIVE, asset_class=AssetClass.US_EQUITY)
    tickers = load_tickers(args.universe)
    df = run_check(lambda: client.get_all_assets(request), tickers)

    s = summarize(df)
    print(f"READ-ONLY. {len(tickers)} tickers from {Path(args.universe).name} checked against Alpaca's active US-equity list.\n")
    print(df[["symbol", "role", "found", "tradable", "fractionable", "shortable", "issue"]].to_string(index=False))
    print(f"\n{s['n_ok']} of {s['n_ranked']} ranked ETFs are active, tradable and fractionable.")
    if s["blockers"]:
        print("BLOCKERS: " + ", ".join(s["blockers"]))
    if s["hedge_issues"]:
        print("Hedge (not a blocker, never bought): " + ", ".join(s["hedge_issues"]))
    if args.out:
        df.to_csv(args.out, index=False)
        print(f"Saved {args.out}")
    sys.exit(1 if s["blockers"] else 0)


if __name__ == "__main__":
    main()

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
