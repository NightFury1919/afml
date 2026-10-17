import dataclasses
import json
import sys

import numpy as np
import pandas as pd
import pytest

from hrp_allocation import hrp_weights, trailing_returns
from paper_runner_v2 import (PROTOCOL, combined_weights, make_plan, qf_weights, scoreboard,
                             write_plan)

# --------------------------------------------------------------------------- toy universes
# Four ETFs in two classes. SPY is in the QF universe too (it is a US-equity ETF), as in the real one.
G4 = {"SPY": "eq", "E1": "eq", "B1": "bd", "B2": "bd"}
# Three equity ETFs and ONE bond ETF: unequal class sizes, so size and equal budgets differ.
G31 = {"SPY": "eq", "E1": "eq", "E2": "eq", "B1": "bd"}
W = 5                                                # tiny window so the toy frames can be short


def P(**kw):
    return dataclasses.replace(PROTOCOL, **{"window": W, **kw})


def prices_from_returns(rets, start=100.0):
    """rets: {ticker: list of daily returns}. Day 0 has no return; prices compound exactly."""
    n = len(next(iter(rets.values())))
    idx = pd.bdate_range("2026-01-05", periods=n + 1)
    px = pd.DataFrame({t: start * np.cumprod([1.0] + [1.0 + r for r in rs]) for t, rs in rets.items()},
                      index=idx)
    px.index.name, px.columns.name = "date", "asset"
    return px


def flat(tickers, n=12):
    return prices_from_returns({t: [0.0] * n for t in tickers})


def random_prices(tickers, n=80, seed=0):
    rng = np.random.default_rng(seed)
    idx = pd.bdate_range("2026-01-05", periods=n)
    px = pd.DataFrame(100 * np.exp(np.cumsum(rng.normal(0.0003, 0.01, (n, len(tickers))), axis=0)),
                      index=idx, columns=tickers)
    px.index.name, px.columns.name = "date", "asset"
    return px


PX4 = flat(list(G4))
TODAY = PX4.index[-1] + pd.Timedelta(days=1)


# --------------------------------------------------------------------- the QF sleeve, by rule
def test_size_rule_holds_every_etf_equally():
    w = qf_weights(PX4, P(qf_rule="size"), G4)
    assert w == pytest.approx({"SPY": 0.25, "E1": 0.25, "B1": 0.25, "B2": 0.25})


def test_equal_rule_gives_each_class_the_same_share_then_splits_it_equally():
    w = qf_weights(flat(list(G31)), P(qf_rule="equal"), G31)
    assert w == pytest.approx({"SPY": 1 / 6, "E1": 1 / 6, "E2": 1 / 6, "B1": 0.5})


def test_size_rule_on_unequal_classes_is_plain_equal_weight():
    w = qf_weights(flat(list(G31)), P(qf_rule="size"), G31)
    assert w == pytest.approx({t: 0.25 for t in G31})


def test_inverse_vol_gives_the_calmer_class_the_larger_budget():
    n = 30
    up_down = lambda r: [r if t % 2 == 0 else -r for t in range(n)]       # exactly +r, -r, +r, ...
    px = prices_from_returns({"SPY": up_down(0.01), "E1": up_down(0.01),
                              "B1": up_down(0.005), "B2": up_down(0.005)})
    w = qf_weights(px, P(qf_rule="inverse_vol", window=10), G4)
    # equity vol is twice bond vol: budgets are 1/3 and 2/3, split equally inside each class
    assert w == pytest.approx({"SPY": 1 / 6, "E1": 1 / 6, "B1": 1 / 3, "B2": 1 / 3})


def test_inverse_vol_uses_only_the_trailing_window():
    n = 40
    calm = [0.005 if t % 2 == 0 else -0.005 for t in range(n)]
    wild = [0.02 if t % 2 == 0 else -0.02 for t in range(n)]
    # equity was wild early and calm in the last 10 days; bonds the opposite
    eq = wild[:30] + calm[30:]
    bd = calm[:30] + wild[30:]
    px = prices_from_returns({"SPY": eq, "E1": eq, "B1": bd, "B2": bd})
    w = qf_weights(px, P(qf_rule="inverse_vol", window=10), G4)
    # in the last 10 days equity vol is a quarter of bond vol: budgets 0.8 / 0.2, split inside each class
    assert w == pytest.approx({"SPY": 0.4, "E1": 0.4, "B1": 0.1, "B2": 0.1})


def test_hrp_rule_is_the_repo_hrp_weights_over_the_trailing_window():
    tickers = ["SPY", "E1", "E2", "B1", "B2", "B3"]
    groups = {t: ("eq" if t[0] in "SE" else "bd") for t in tickers}
    px = random_prices(tickers)
    w = qf_weights(px, P(qf_rule="hrp", window=60), groups)
    expected = hrp_weights(trailing_returns(px, tickers, 60))
    assert sum(w.values()) == pytest.approx(1.0)
    assert w == pytest.approx(expected.to_dict())


def test_an_unknown_rule_is_rejected():
    with pytest.raises(ValueError, match="qf_rule"):
        qf_weights(PX4, P(qf_rule="momentum"), G4)


def test_an_etf_without_a_full_window_of_prices_is_left_out():
    px = flat(list(G4))
    px.loc[px.index[:4], "B2"] = np.nan                    # B2 only has 8 of the 13 rows
    short = px.copy()
    short.loc[short.index[:9], "B2"] = np.nan              # now fewer than window + 1 = 6 rows
    assert "B2" in qf_weights(px, P(qf_rule="size"), G4)
    assert "B2" not in qf_weights(short, P(qf_rule="size"), G4)
    assert qf_weights(short, P(qf_rule="size"), G4) == pytest.approx({"SPY": 1 / 3, "E1": 1 / 3, "B1": 1 / 3})


# ------------------------------------------------------------------ exposure and cap (QF sleeve)
def test_cap_moves_the_excess_to_the_others_in_proportion():
    w = qf_weights(flat(list(G31)), P(qf_rule="equal", max_weight=0.3), G31)
    assert w["B1"] == pytest.approx(0.3)                   # capped, was 0.5
    assert w["SPY"] == w["E1"] == w["E2"] == pytest.approx(0.7 / 3)
    assert sum(w.values()) == pytest.approx(1.0)


def test_exposure_scales_the_sleeve_and_the_cap_applies_to_the_final_weights():
    w = qf_weights(PX4, P(qf_rule="size", exposure=0.8), G4)
    assert w == pytest.approx({t: 0.2 for t in G4}) and sum(w.values()) == pytest.approx(0.8)
    capped = qf_weights(PX4, P(qf_rule="size", exposure=0.8, max_weight=0.18), G4)
    assert capped == pytest.approx({t: 0.18 for t in G4})  # everything capped: the rest stays cash


def test_hrp_rule_honours_exposure_and_cap_too():
    tickers = ["SPY", "E1", "E2", "B1", "B2", "B3"]
    groups = {t: ("eq" if t[0] in "SE" else "bd") for t in tickers}
    px = random_prices(tickers)
    half = qf_weights(px, P(qf_rule="hrp", window=60, exposure=0.5), groups)
    assert sum(half.values()) == pytest.approx(0.5)
    capped = qf_weights(px, P(qf_rule="hrp", window=60, max_weight=0.2), groups)
    assert max(capped.values()) <= 0.2 + 1e-9 and sum(capped.values()) == pytest.approx(1.0)


# ---------------------------------------------------------------------- the two sleeves together
def test_combined_weights_add_the_spy_sleeve_to_the_scaled_qf_sleeve():
    qf = {"SPY": 0.25, "E1": 0.25, "B1": 0.25, "B2": 0.25}
    c = combined_weights(qf, P(spy_share=0.4))
    assert c == pytest.approx({"SPY": 0.4 + 0.6 * 0.25, "E1": 0.15, "B1": 0.15, "B2": 0.15})
    assert sum(c.values()) == pytest.approx(1.0)


def test_all_spy_sleeve_is_just_spy_and_no_spy_sleeve_is_just_the_qf_book():
    qf = {"SPY": 0.25, "E1": 0.25, "B1": 0.25, "B2": 0.25}
    assert combined_weights(qf, P(spy_share=1.0)) == pytest.approx({"SPY": 1.0})
    assert combined_weights(qf, P(spy_share=0.0)) == pytest.approx(qf)


def test_cap_is_a_share_of_the_qf_sleeve_not_of_the_account():
    qf = qf_weights(flat(list(G31)), P(qf_rule="equal", max_weight=0.3), G31)
    c = combined_weights(qf, P(spy_share=0.5))
    assert c["B1"] == pytest.approx(0.5 * 0.3)             # 15% of the account
    assert c["E1"] == pytest.approx(0.5 * 0.7 / 3)


def test_cash_left_by_exposure_stays_outside_both_sleeves():
    qf = qf_weights(PX4, P(qf_rule="size", exposure=0.8), G4)
    c = combined_weights(qf, P(spy_share=0.4))
    assert sum(c.values()) == pytest.approx(0.4 + 0.6 * 0.8)


# ------------------------------------------------------------------------------------- the plan
def test_empty_account_plan_buys_both_sleeves_and_spends_all_the_cash():
    plan = make_plan(PX4, {}, 1000.0, P(spy_share=0.4, qf_rule="size"), today=TODAY, groups=G4)
    assert plan["spy_dollars"] == pytest.approx(400.0) and plan["qf_dollars"] == pytest.approx(600.0)
    buys = {o["symbol"]: o["notional"] for o in plan["orders"]}
    assert buys == pytest.approx({"SPY": 550.0, "E1": 150.0, "B1": 150.0, "B2": 150.0})
    assert plan["cash_after"] == pytest.approx(0.0, abs=0.05)
    assert plan["as_of"] == PX4.index[-1].date().isoformat() and plan["version"] == 2


def test_cent_rounding_never_spends_more_than_the_account_holds():
    g7 = {f"E{i}": "eq" for i in range(7)}                 # 1000 / 7 = 142.857..., each order rounds UP to .86
    px = flat(list(g7))
    plan = make_plan(px, {}, 1000.0, P(spy_share=0.0, qf_rule="size"), today=TODAY, groups=g7)
    spent = sum(o["notional"] for o in plan["orders"])
    assert spent <= 1000.0 + 1e-9 and plan["cash_after"] >= 0
    assert spent == pytest.approx(1000.0, abs=0.005)       # trimmed by the 2 cents, not by more
    notionals = sorted(o["notional"] for o in plan["orders"])
    assert notionals[0] == pytest.approx(142.84) and notionals[1:] == [142.86] * 6   # one order carries the trim


def test_the_cent_trim_leaves_an_exact_account_alone():
    plan = make_plan(PX4, {}, 1000.0, P(spy_share=0.4, qf_rule="size"), today=TODAY, groups=G4)
    assert {o["symbol"]: o["notional"] for o in plan["orders"]} == pytest.approx(
        {"SPY": 550.0, "E1": 150.0, "B1": 150.0, "B2": 150.0})


def test_running_again_with_the_resulting_holdings_orders_nothing():
    first = make_plan(PX4, {}, 1000.0, P(), today=TODAY, groups=G4)
    held = {s: w * 1000.0 for s, w in first["weights"].items()}
    assert make_plan(PX4, held, 1000.0, P(), today=TODAY, groups=G4)["orders"] == []


def test_the_hedge_is_never_bought_and_a_held_hedge_is_closed():
    px = flat(list(G4) + ["SH"])
    plan = make_plan(px, {"SH": 200.0}, 1000.0, P(), today=TODAY, groups=G4)
    assert "SH" not in plan["weights"]
    assert {"symbol": "SH", "side": "sell", "close_all": True} in plan["orders"]


def test_bad_settings_and_inputs_are_rejected():
    for bad in (-0.1, 1.1):
        with pytest.raises(ValueError, match="spy_share"):
            make_plan(PX4, {}, 1000.0, P(spy_share=bad), today=TODAY, groups=G4)
    with pytest.raises(ValueError, match="exposure"):
        make_plan(PX4, {}, 1000.0, P(exposure=1.5), today=TODAY, groups=G4)
    with pytest.raises(ValueError, match="max_weight"):
        make_plan(PX4, {}, 1000.0, P(max_weight=0.0), today=TODAY, groups=G4)
    with pytest.raises(ValueError, match="equity"):
        make_plan(PX4, {}, 0.0, P(), today=TODAY, groups=G4)
    with pytest.raises(ValueError, match="rebalance_every"):
        make_plan(PX4, {}, 1000.0, P(rebalance_every=0), today=TODAY, groups=G4)


def test_a_spy_sleeve_needs_a_spy_price_on_the_last_date():
    px = flat(list(G4))
    px.loc[px.index[-1], "SPY"] = np.nan
    with pytest.raises(ValueError, match="SPY"):
        make_plan(px, {}, 1000.0, P(spy_share=0.5), today=TODAY, groups=G4)
    assert make_plan(px.drop(columns="SPY"), {}, 1000.0, P(spy_share=0.0), today=TODAY,
                     groups={"E1": "eq", "B1": "bd", "B2": "bd"})["orders"]      # no SPY sleeve, no SPY needed


def test_stale_data_produces_a_warning_but_still_a_plan():
    late = PX4.index[-1] + pd.Timedelta(days=30)
    plan = make_plan(PX4, {}, 1000.0, P(), today=late, groups=G4)
    assert any("old" in w for w in plan["warnings"]) and plan["orders"]
    assert make_plan(PX4, {}, 1000.0, P(), today=TODAY, groups=G4)["warnings"] == []


def test_the_plan_records_the_protocol_it_was_made_under():
    plan = make_plan(PX4, {}, 1000.0, P(spy_share=0.3), today=TODAY, groups=G4)
    assert plan["protocol"] == dataclasses.asdict(P(spy_share=0.3))


def test_making_a_plan_never_loads_any_trading_client():
    make_plan(PX4, {}, 1000.0, P(), today=TODAY, groups=G4)
    assert not [m for m in sys.modules if "alpaca" in m.lower()]


def test_write_plan_round_trips_and_never_overwrites(tmp_path):
    plan = make_plan(PX4, {}, 1000.0, P(), today=TODAY, groups=G4)
    path = write_plan(plan, tmp_path)
    assert path.name == f"plan_v2_{plan['as_of']}.json"
    assert json.loads(path.read_text(encoding="utf-8"))["weights"] == pytest.approx(plan["weights"])
    with pytest.raises(FileExistsError):
        write_plan(plan, tmp_path)


# ------------------------------------------------------------------------------- the scoreboard
G2 = {"SPY": "eq", "B1": "bd"}                   # equal rule: QF sleeve is half SPY, half B1


def steady(n=8, spy=0.01, bond=0.0):
    return prices_from_returns({"SPY": [spy] * n, "B1": [bond] * n})


def test_first_row_is_the_start_date_with_each_sleeve_at_its_share_of_equity():
    px = steady()
    sb = scoreboard(px, px.index[2], P(spy_share=0.5, qf_rule="equal", window=1), 1000.0, G2)
    assert sb.index[0] == px.index[2]
    assert sb.iloc[0]["spy_value"] == pytest.approx(500.0) and sb.iloc[0]["qf_value"] == pytest.approx(500.0)
    assert sb.iloc[0]["spy_return"] == 0 and sb.iloc[0]["qf_minus_spy"] == 0


def test_the_sleeves_start_unequal_when_the_split_is_unequal():
    px = steady()
    sb = scoreboard(px, px.index[1], P(spy_share=0.25, qf_rule="equal", window=1), 1000.0, G2)
    assert sb.iloc[0]["spy_value"] == pytest.approx(250.0) and sb.iloc[0]["qf_value"] == pytest.approx(750.0)


def test_spy_sleeve_compounds_spy_and_a_daily_rebalanced_qf_sleeve_compounds_the_blend():
    px = steady(n=8)                                       # start on day 1, so day 0 supplies the window
    sb = scoreboard(px, px.index[1], P(spy_share=0.5, qf_rule="equal", window=1), 1000.0, G2)
    k = np.arange(len(sb))
    assert len(sb) == 8
    assert sb["spy_value"].to_numpy() == pytest.approx(500.0 * 1.01 ** k)
    assert sb["qf_value"].to_numpy() == pytest.approx(500.0 * 1.005 ** k)       # half SPY, half flat
    assert sb["qf_minus_spy"].iloc[-1] == pytest.approx(1.005 ** 7 - 1.01 ** 7)


def test_without_rebalancing_the_qf_sleeve_drifts_with_the_winner():
    px = steady(n=8)
    sb = scoreboard(px, px.index[1], P(spy_share=0.5, qf_rule="equal", window=1, rebalance_every=100), 1000.0, G2)
    k = np.arange(len(sb))
    assert sb["qf_value"].to_numpy() == pytest.approx(250.0 * 1.01 ** k + 250.0)


def test_rebalancing_every_third_day_resets_the_weights_on_those_days():
    px = steady(n=8)
    sb = scoreboard(px, px.index[1], P(spy_share=0.5, qf_rule="equal", window=1, rebalance_every=3), 1000.0, G2)
    day3 = 250.0 * 1.01 ** 3 + 250.0                                           # drifted, then reset to 50/50
    day6 = (day3 / 2) * 1.01 ** 3 + day3 / 2
    assert sb["qf_value"].iloc[3] == pytest.approx(day3) and sb["qf_value"].iloc[6] == pytest.approx(day6)


def test_drawdown_is_the_fall_from_the_running_peak():
    px = prices_from_returns({"SPY": [0.10, 0.10, -0.20, 0.0], "B1": [0.0, 0.0, 0.0, 0.0]})
    sb = scoreboard(px, px.index[1], P(spy_share=0.5, qf_rule="equal", window=1), 1000.0, G2)
    assert sb["spy_drawdown"].to_numpy() == pytest.approx([0.0, 0.0, 1 - 0.88 / 1.10, 1 - 0.88 / 1.10])


def test_the_scoreboard_never_looks_ahead():
    n = 40
    px = random_prices(["SPY", "E1", "B1", "B2"], n=n, seed=3)
    full = scoreboard(px, px.index[5], P(qf_rule="inverse_vol", window=4), 1000.0, G4)
    cut = scoreboard(px.iloc[:25], px.index[5], P(qf_rule="inverse_vol", window=4), 1000.0, G4)
    pd.testing.assert_frame_equal(cut, full.loc[cut.index])


def test_scoreboard_matches_an_independent_day_by_day_reference():
    # Each day's weights come from prices up to THAT day only and earn the NEXT day's return.
    px = random_prices(["SPY", "E1", "B1", "B2"], n=30, seed=7)
    proto = P(spy_share=0.4, qf_rule="inverse_vol", window=4)
    sb = scoreboard(px, px.index[6], proto, 1000.0, G4)
    value, ref = 600.0, [600.0]
    for pos in range(6, len(px) - 1):
        w = qf_weights(px.iloc[:pos + 1], proto, G4)
        nxt = px.iloc[pos + 1] / px.iloc[pos] - 1.0
        value *= 1.0 + sum(wt * nxt[t] for t, wt in w.items())
        ref.append(value)
    assert sb["qf_value"].to_numpy() == pytest.approx(ref)
    spy = 400.0 * px["SPY"].iloc[6:] / px["SPY"].iloc[6]
    assert sb["spy_value"].to_numpy() == pytest.approx(spy.to_numpy())


def test_scoreboard_start_must_exist_and_inputs_are_checked():
    px = steady()
    with pytest.raises(ValueError, match="start"):
        scoreboard(px, px.index[-1] + pd.Timedelta(days=5), P(qf_rule="equal"), 1000.0, G2)
    with pytest.raises(ValueError, match="equity"):
        scoreboard(px, px.index[0], P(qf_rule="equal"), -1.0, G2)


def test_a_missing_price_while_holding_is_an_error_not_a_silent_zero():
    px = steady(n=8)
    px.loc[px.index[4], "B1"] = np.nan
    with pytest.raises(ValueError, match="missing price"):
        scoreboard(px, px.index[1], P(qf_rule="equal", window=1), 1000.0, G2)


def test_an_all_spy_split_leaves_the_qf_columns_undefined_not_divided_by_zero():
    px = steady()
    sb = scoreboard(px, px.index[1], P(spy_share=1.0, qf_rule="equal", window=1), 1000.0, G2)
    assert (sb["qf_value"] == 0).all() and sb["qf_return"].isna().all()
    assert sb["spy_value"].iloc[-1] == pytest.approx(1000.0 * 1.01 ** (len(sb) - 1))


def test_whole_share_mode_spends_no_more_than_equity():
    px = random_prices(list(G4), n=30, seed=1)
    plan = make_plan(px, {}, 5000.0, P(mode="whole", spy_share=0.4), today=px.index[-1] + pd.Timedelta(days=1), groups=G4)
    last = px.iloc[-1]
    spent = sum(o["qty"] * last[o["symbol"]] for o in plan["orders"])
    assert 0 < spent <= 5000.0 and all(float(o["qty"]).is_integer() for o in plan["orders"])


# ------------------------------------------------------------------------------------- the CLI
def _write_real_shaped_snapshot(folder, n=300):
    """Synthetic prices for the real 48 ETFs plus SH, written the way pull_prices.py does (CLI tests only)."""
    from etf_classes import CLASS_OF
    px = random_prices(list(CLASS_OF) + ["SH"], n=n, seed=11)
    path = folder / f"prices_daily_asof_{px.index[-1].date().isoformat()}.csv"
    px.to_csv(path)
    return path, px


def _run_cli(monkeypatch, tmp_path, *extra):
    import paper_runner_v2
    monkeypatch.setattr(sys, "argv", ["paper_runner_v2.py", "--folder", str(tmp_path),
                                      "--plans-dir", str(tmp_path / "plans"), *extra])
    paper_runner_v2.main()


def test_cli_prints_the_plan_and_saves_it(monkeypatch, tmp_path, capsys):
    path, px = _write_real_shaped_snapshot(tmp_path)
    _run_cli(monkeypatch, tmp_path)
    out = capsys.readouterr().out
    assert "DRY RUN" in out and "Cash after: $0.00" in out
    assert (tmp_path / "plans" / f"plan_v2_{px.index[-1].date().isoformat()}.json").exists()


def test_cli_scoreboard_with_a_start_date_that_has_prices_prints_rows(monkeypatch, tmp_path, capsys):
    path, px = _write_real_shaped_snapshot(tmp_path)
    _run_cli(monkeypatch, tmp_path, "--start", px.index[-4].date().isoformat())
    out = capsys.readouterr().out
    assert "SCOREBOARD from" in out and "qf_minus_spy" in out


def test_cli_says_so_instead_of_crashing_when_the_scoreboard_start_is_in_the_future(monkeypatch, tmp_path, capsys):
    path, px = _write_real_shaped_snapshot(tmp_path)
    future = (px.index[-1] + pd.Timedelta(days=3)).date().isoformat()
    _run_cli(monkeypatch, tmp_path, "--start", future)                 # must not raise
    out = capsys.readouterr().out
    assert "Saved" in out                                              # the plan was still written
    assert "no scoreboard yet" in out and future in out

# ---------------------------------------------------------------------------
# TDD RESULTS (pytest, 2026-10-17; sandbox run: Linux, Python 3.10.20, pandas 1.5.3, numpy 1.23.5,
# scikit-learn 1.2.2, the same versions as the mlfinlab env)
# $ cd portfolio ; pytest test_paper_runner_v2.py -v
#
# collected 41 items
# test_paper_runner_v2.py::test_size_rule_holds_every_etf_equally PASSED   [  2%]
# test_paper_runner_v2.py::test_equal_rule_gives_each_class_the_same_share_then_splits_it_equally PASSED [  4%]
# test_paper_runner_v2.py::test_size_rule_on_unequal_classes_is_plain_equal_weight PASSED [  7%]
# test_paper_runner_v2.py::test_inverse_vol_gives_the_calmer_class_the_larger_budget PASSED [  9%]
# test_paper_runner_v2.py::test_inverse_vol_uses_only_the_trailing_window PASSED [ 12%]
# test_paper_runner_v2.py::test_hrp_rule_is_the_repo_hrp_weights_over_the_trailing_window PASSED [ 14%]
# test_paper_runner_v2.py::test_an_unknown_rule_is_rejected PASSED         [ 17%]
# test_paper_runner_v2.py::test_an_etf_without_a_full_window_of_prices_is_left_out PASSED [ 19%]
# test_paper_runner_v2.py::test_cap_moves_the_excess_to_the_others_in_proportion PASSED [ 21%]
# test_paper_runner_v2.py::test_exposure_scales_the_sleeve_and_the_cap_applies_to_the_final_weights PASSED [ 24%]
# test_paper_runner_v2.py::test_hrp_rule_honours_exposure_and_cap_too PASSED [ 26%]
# test_paper_runner_v2.py::test_combined_weights_add_the_spy_sleeve_to_the_scaled_qf_sleeve PASSED [ 29%]
# test_paper_runner_v2.py::test_all_spy_sleeve_is_just_spy_and_no_spy_sleeve_is_just_the_qf_book PASSED [ 31%]
# test_paper_runner_v2.py::test_cap_is_a_share_of_the_qf_sleeve_not_of_the_account PASSED [ 34%]
# test_paper_runner_v2.py::test_cash_left_by_exposure_stays_outside_both_sleeves PASSED [ 36%]
# test_paper_runner_v2.py::test_empty_account_plan_buys_both_sleeves_and_spends_all_the_cash PASSED [ 39%]
# test_paper_runner_v2.py::test_cent_rounding_never_spends_more_than_the_account_holds PASSED [ 41%]
# test_paper_runner_v2.py::test_the_cent_trim_leaves_an_exact_account_alone PASSED [ 43%]
# test_paper_runner_v2.py::test_running_again_with_the_resulting_holdings_orders_nothing PASSED [ 46%]
# test_paper_runner_v2.py::test_the_hedge_is_never_bought_and_a_held_hedge_is_closed PASSED [ 48%]
# test_paper_runner_v2.py::test_bad_settings_and_inputs_are_rejected PASSED [ 51%]
# test_paper_runner_v2.py::test_a_spy_sleeve_needs_a_spy_price_on_the_last_date PASSED [ 53%]
# test_paper_runner_v2.py::test_stale_data_produces_a_warning_but_still_a_plan PASSED [ 56%]
# test_paper_runner_v2.py::test_the_plan_records_the_protocol_it_was_made_under PASSED [ 58%]
# test_paper_runner_v2.py::test_making_a_plan_never_loads_any_trading_client PASSED [ 60%]
# test_paper_runner_v2.py::test_write_plan_round_trips_and_never_overwrites PASSED [ 63%]
# test_paper_runner_v2.py::test_first_row_is_the_start_date_with_each_sleeve_at_its_share_of_equity PASSED [ 65%]
# test_paper_runner_v2.py::test_the_sleeves_start_unequal_when_the_split_is_unequal PASSED [ 68%]
# test_paper_runner_v2.py::test_spy_sleeve_compounds_spy_and_a_daily_rebalanced_qf_sleeve_compounds_the_blend PASSED [ 70%]
# test_paper_runner_v2.py::test_without_rebalancing_the_qf_sleeve_drifts_with_the_winner PASSED [ 73%]
# test_paper_runner_v2.py::test_rebalancing_every_third_day_resets_the_weights_on_those_days PASSED [ 75%]
# test_paper_runner_v2.py::test_drawdown_is_the_fall_from_the_running_peak PASSED [ 78%]
# test_paper_runner_v2.py::test_the_scoreboard_never_looks_ahead PASSED    [ 80%]
# test_paper_runner_v2.py::test_scoreboard_matches_an_independent_day_by_day_reference PASSED [ 82%]
# test_paper_runner_v2.py::test_scoreboard_start_must_exist_and_inputs_are_checked PASSED [ 85%]
# test_paper_runner_v2.py::test_a_missing_price_while_holding_is_an_error_not_a_silent_zero PASSED [ 87%]
# test_paper_runner_v2.py::test_an_all_spy_split_leaves_the_qf_columns_undefined_not_divided_by_zero PASSED [ 90%]
# test_paper_runner_v2.py::test_whole_share_mode_spends_no_more_than_equity PASSED [ 92%]
# test_paper_runner_v2.py::test_cli_prints_the_plan_and_saves_it PASSED    [ 95%]
# test_paper_runner_v2.py::test_cli_scoreboard_with_a_start_date_that_has_prices_prints_rows PASSED [ 97%]
# test_paper_runner_v2.py::test_cli_says_so_instead_of_crashing_when_the_scoreboard_start_is_in_the_future PASSED [100%]
# ============================== 41 passed in 1.30s ==============================
#
# Mutation checks (sandbox, 19): SPY slice overwritten instead of added, QF weights not scaled by (1 - spy_share),
#   HRP ignoring exposure, HRP ignoring the cap, class rules ignoring exposure, class rules ignoring the cap, HRP
#   window ignored, scoreboard weights one day ahead, positions never drifting, rebalance_every ignored, missing price
#   as a zero return, plan overwrite allowed, any single price qualifying an ETF, drawdown from the trough, spy_share
#   not validated, overspend trim removed, overspend trim by the wrong amount, CLI future-start guard removed, sleeve dollars swapped each made a
#   test fail; original restored. Two tests were added after review: the one-day look-ahead mutation survived the
#   truncated-prices test (it cannot see a look-ahead of one day), so a day-by-day reference test was added.
# Real snapshot (prices_daily_asof_2026-10-08.csv, 48 ETFs): qf_weights for all 4 rules x {no cap, 5%, 10%} match the
#   earlier independent cap scenarios to 5e-5 (that file is rounded to 4 dp). The first real run found cash_after
#   -$0.16 (48 orders rounded up to the cent); fixed by trimming the largest buy, with a test. HRP with no cap leaves
#   ~$8.92 uninvested at $1,000 because 20 of its ETFs fall below the $1 minimum order (expected, not a bug).
# Windows mlfinlab env (2026-10-17, your run): 38 passed in 7.62s before the 3 CLI tests below were added; the real run on
#   prices_daily_asof_2026-10-16.csv gave cash_after $0.00 with the SPY order trimmed to $510.26. The CLI crashed on a
#   --start after the last price date (2026-10-19 vs 10-16); it now prints a note instead, with 3 CLI tests added.
# The scoreboard ran on the real snapshot as a mechanics check only. Its numbers are NOT a finding.
# ---------------------------------------------------------------------------
