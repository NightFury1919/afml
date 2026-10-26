"""Performance page: a short, honest, labelled summary of how the paper portfolios are doing.

Why this exists. A return number on its own proves little, especially early on. This page puts each portfolio next to
two yardsticks (SPY = "just buy the market", SHY = "a cash-like short Treasury ETF") and shows the risk side first-class:
how volatile it was, the worst fall from a peak, and how far the account is from the loss floor. It says what the sample
length is, and it refuses to print annualized return or volatility until there is enough history for them to mean
anything. The point is a page you can show without being embarrassed by a question.

Inputs are daily dollar values of the portfolio (the model replay from paper_runner_v2.scoreboard, or the real account
equity from a CSV) and the adjusted closes of the benchmarks. Everything is price data already in the pipeline. No orders.

Run:  python performance_page.py --folder <price folder> --config <run_config.json> --out <output folder>
"""
import argparse
import json
import math
from pathlib import Path

import numpy as np
import pandas as pd

TRADING_DAYS = 252
MIN_DAYS_ANNUALIZED = 126      # about six months of daily returns before an annualized return is shown
MIN_DAYS_VOL = 20              # about a month of daily returns before a volatility is shown
DEFAULT_BENCHMARKS = ("SPY", "SHY")


# ------------------------------------------------------------------------------------ metrics
def _check(values):
    if not isinstance(values, pd.Series):
        raise ValueError("values must be a pandas Series indexed by date")
    if len(values) < 2:
        raise ValueError("need at least two dated values")
    if not values.index.is_monotonic_increasing or values.index.has_duplicates:
        raise ValueError("dates must be strictly increasing")
    v = values.astype(float)
    if not np.isfinite(v.to_numpy()).all():
        raise ValueError("values contain a missing or infinite number")
    if (v <= 0).any():
        raise ValueError("values must be positive dollars")
    return v


def series_metrics(values, floor=None):
    """Plain numbers for one dated series of dollar values. None means "not enough history to say"."""
    v = _check(values)
    r = v.pct_change().dropna()
    days = len(r)
    peak = v.cummax()
    fall = 1.0 - v / peak
    out = {
        "start_date": str(v.index[0].date()), "end_date": str(v.index[-1].date()), "days": days,
        "first_value": float(v.iloc[0]), "last_value": float(v.iloc[-1]), "lowest_value": float(v.min()),
        "total_return": float(v.iloc[-1] / v.iloc[0] - 1.0),
        "annualized_return": (float((v.iloc[-1] / v.iloc[0]) ** (TRADING_DAYS / days) - 1.0)
                              if days >= MIN_DAYS_ANNUALIZED else None),
        "volatility": float(r.std(ddof=1) * math.sqrt(TRADING_DAYS)) if days >= MIN_DAYS_VOL else None,
        "max_drawdown": float(fall.max()), "current_drawdown": float(fall.iloc[-1]),
        "floor": None, "floor_distance": None, "floor_distance_pct": None,
    }
    if floor is not None:
        out["floor"] = float(floor)
        out["floor_distance"] = float(v.iloc[-1] - floor)
        out["floor_distance_pct"] = float((v.iloc[-1] - floor) / v.iloc[-1])
    return out


def align(series_by_name, start_value):
    """Cut every series to the dates they all have and rescale each to `start_value` on the first common date,
    so they can be compared (and charted) on one scale."""
    frame = pd.concat({k: _check(s) for k, s in series_by_name.items()}, axis=1, join="inner")
    if len(frame) < 2:
        raise ValueError("the series share fewer than two dates")
    return frame / frame.iloc[0] * float(start_value)


def benchmark_series(prices, tickers, start):
    """{ticker: adjusted close series} from the first price date on or after `start`."""
    missing = [t for t in tickers if t not in prices.columns]
    if missing:
        raise ValueError(f"no prices for: {', '.join(missing)}")
    pos = int(prices.index.searchsorted(pd.Timestamp(start)))
    if pos >= len(prices):
        raise ValueError(f"start {pd.Timestamp(start).date()} is after the last price date {prices.index[-1].date()}")
    return {t: prices[t].iloc[pos:].astype(float) for t in tickers}


def has_live_days(prices, start):
    """True when there is at least one price date after the first on-or-after-`start` date, so a live line can be drawn."""
    pos = int(prices.index.searchsorted(pd.Timestamp(start)))
    return pos < len(prices) - 1


def read_equity_csv(path):
    """A CSV with columns date,equity (one row per day) -> Series. Used for the real account equity."""
    df = pd.read_csv(path)
    if not {"date", "equity"} <= set(df.columns):
        raise ValueError("equity CSV needs the columns: date, equity")
    try:
        s = pd.Series(df["equity"].astype(float).to_numpy(), index=pd.to_datetime(df["date"]))
    except (ValueError, TypeError) as exc:
        raise ValueError(f"equity CSV has an unreadable value: {exc}") from None
    return s.sort_index()


# ------------------------------------------------------------------------------------ rendering
def _pct(x, signed=True):
    return "n/a" if x is None else (f"{x:+.1%}" if signed else f"{x:.1%}")


def _usd(x):
    return "n/a" if x is None else f"${x:,.2f}"


def render_markdown(metrics, title, as_of, start, notes=()):
    names = list(metrics)
    first = metrics[names[0]]
    short = any(m["days"] < MIN_DAYS_ANNUALIZED for m in metrics.values())
    floor = next((m["floor"] for m in metrics.values() if m["floor"] is not None), None)

    def row(label, fn):
        return "| " + label + " | " + " | ".join(fn(metrics[n]) for n in names) + " |"

    lines = [f"# {title}", "",
             "**Alpaca paper accounts: simulated money, not real money.** Returns shown are before any taxes and "
             "assume the fills the paper account gave.", "",
             f"Sample: {first['days']} trading days ({start} to {as_of}). Dollar figures are rescaled so every column "
             f"starts at the same value.", ""]
    if short:
        lines += [f"> **Sample too short to judge.** With fewer than {MIN_DAYS_ANNUALIZED} trading days, annualized "
                  f"return is not shown and with fewer than {MIN_DAYS_VOL} volatility is not shown. Treat this page as "
                  "a monitor of risk and behaviour, not as a result.", ""]
    lines += ["| Measure | " + " | ".join(names) + " |", "|---|" + "---|" * len(names),
              row("Total return", lambda m: _pct(m["total_return"])),
              row("Annualized return", lambda m: _pct(m["annualized_return"])),
              row("Volatility (annualized)", lambda m: _pct(m["volatility"], signed=False)),
              row("Worst fall from a peak", lambda m: _pct(m["max_drawdown"], signed=False)),
              row("Fall from peak today", lambda m: _pct(m["current_drawdown"], signed=False)),
              row("Lowest value", lambda m: _usd(m["lowest_value"])),
              row("Latest value", lambda m: _usd(m["last_value"])),
              row("Distance above the loss floor",
                  lambda m: "n/a" if m["floor_distance"] is None
                  else f"{_usd(m['floor_distance'])} ({_pct(m['floor_distance_pct'], signed=False)})"),
              ""]
    if floor is not None:
        lines += [f"Loss floor: {_usd(floor)}. If the account equity falls below it, execution refuses to place orders.", ""]
    lines += ["SPY is the market yardstick. SHY (short Treasuries) is the cash-like yardstick: a portfolio that cannot "
              "beat SHY with more risk has not earned that risk. Benchmarks use adjusted closes (dividends included).", ""]
    for n in notes:
        lines += [f"- {n}"]
    return "\n".join(lines).rstrip() + "\n"


def _chart(frame, path, title, floor):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    palette = ["#0072B2", "#E69F00", "#009E73", "#CC79A7", "#56B4E9"]
    fig, ax = plt.subplots(figsize=(8, 4), dpi=130)
    for i, col in enumerate(frame.columns):
        ax.plot(frame.index, frame[col], color=palette[i % len(palette)], lw=2 if i == 0 else 1.4, label=col)
        ax.annotate(col, (frame.index[-1], frame[col].iloc[-1]), xytext=(4, 0), textcoords="offset points",
                    fontsize=8, color=palette[i % len(palette)], va="center")
    if floor is not None:
        ax.axhline(floor, color="#888888", ls="--", lw=1)
        ax.annotate("loss floor", (frame.index[-1], floor), xytext=(0, 3), textcoords="offset points",
                    fontsize=8, color="#666666", ha="right")
    ax.set_title(title, fontsize=10, loc="left")
    ax.set_ylabel("Dollars, equal start", fontsize=9)
    ax.grid(alpha=0.25)
    for side in ("top", "right"):
        ax.spines[side].set_visible(False)
    import matplotlib.dates as mdates
    loc = mdates.AutoDateLocator(minticks=3, maxticks=7)
    ax.xaxis.set_major_locator(loc)
    ax.xaxis.set_major_formatter(mdates.ConciseDateFormatter(loc))
    fig.tight_layout()
    fig.savefig(path)
    plt.close(fig)


def _json_safe(o):
    if isinstance(o, dict):
        return {k: _json_safe(v) for k, v in o.items()}
    if isinstance(o, (np.floating, float)):
        return None if not math.isfinite(float(o)) else float(o)
    if isinstance(o, np.integer):
        return int(o)
    return o


def build_report(series_by_name, out_dir, title, as_of, start, floor, start_value, notes, chart=True,
                 floor_series=None):
    """Write performance.md, performance.json (and performance.png). Only those files are touched."""
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    frame = align(series_by_name, start_value)
    floor_name = floor_series or frame.columns[0]
    metrics = {c: series_metrics(frame[c], floor=floor if (floor is not None and c == floor_name) else None)
               for c in frame.columns}
    md = out_dir / "performance.md"
    md.write_text(render_markdown(metrics, title, as_of, start, notes), encoding="utf-8")
    js = out_dir / "performance.json"
    js.write_text(json.dumps({"title": title, "as_of": as_of, "start": start, "paper": True,
                              "series": _json_safe(metrics)}, indent=2, allow_nan=False) + "\n", encoding="utf-8")
    png = out_dir / "performance.png"
    if chart:
        _chart(frame, png, title + (" (paper trading)" if "replay" not in title else ""), floor)
    return {"markdown": str(md), "json": str(js), "chart": str(png) if chart else None, "metrics": metrics}


# ------------------------------------------------------------------------------------ command line
def main():
    from daily_run import load_config, protocol_from_config, sleeves_held
    from execute_plan import DEFAULT_EQUITY_FLOOR
    from paper_runner_v2 import scoreboard
    from panel_data import load_prices
    from update_prices import latest_snapshot

    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--folder", required=True, help="folder holding prices_daily_asof_*.csv")
    ap.add_argument("--config", required=True, help="the account's run_config.json")
    ap.add_argument("--out", required=True, help="folder for performance.md / .json / .png")
    ap.add_argument("--equity-csv", default=None, help="optional date,equity CSV of the real account equity")
    ap.add_argument("--replay-start", default=None, help="also show a historical replay from this date (hindsight)")
    ap.add_argument("--floor", type=float, default=DEFAULT_EQUITY_FLOOR)
    ap.add_argument("--title", default=None)
    a = ap.parse_args()

    cfg, found = load_config(a.config)
    if not found:
        raise SystemExit(f"{a.config} not found")
    protocol = protocol_from_config(cfg)
    prices = load_prices(latest_snapshot(a.folder))
    held = sleeves_held(protocol.spy_share)
    col = "spy_value" if held == "spy_only" else "qf_value"
    label = "Baseline: SPY only" if held == "spy_only" else "Portfolio (model)"
    equity = float(cfg["equity"])

    def run(start, name, extra_notes, live):
        sb = scoreboard(prices, start, protocol, equity)
        series = {name: sb[col]}
        if a.equity_csv and live:
            series = {"Account (actual)": read_equity_csv(a.equity_csv), **series}
        bench = benchmark_series(prices, [t for t in DEFAULT_BENCHMARKS if not (held == "spy_only" and t == "SPY")],
                                 sb.index[0])
        return build_report({**series, **bench}, a.out if live else Path(a.out) / "replay",
                            (a.title or "Portfolio performance") + ("" if live else " - historical replay, not live"), str(sb.index[-1].date()), str(sb.index[0].date()),
                            a.floor, equity, extra_notes, floor_series=name)

    live_notes = ["The 'model' line replays the portfolio rules on closing prices from the start date. It is not the "
                  "account's actual fills."]
    if has_live_days(prices, cfg["start"]):
        run(cfg["start"], label, live_notes, True)
    else:
        print(f"No live days yet: the live test starts {cfg['start']} and the newest prices are "
              f"{prices.index[-1].date()}. Run the nightly update after the close, then run this again.")
    if a.replay_start:
        run(a.replay_start, label, ["HISTORICAL REPLAY, NOT LIVE. The rules and the ETF list were chosen with hindsight, "
                                    "with no trading costs or taxes. It shows how the rules behaved in the past, "
                                    "not what they will do."], False)
    print(f"wrote {a.out}")


if __name__ == "__main__":
    main()

# ---------------------------------------------------------------------------------------------
# pytest -v test_performance_page.py
# ============================= test session starts ==============================
# platform linux -- Python 3.10.20, pytest-9.1.1, pluggy-1.7.0 -- /tmp/claude-0/-home-claude/186a30d5-aecd-5688-830a-09fb4cd892bd/scratchpad/perf/../venv310/bin/python
# cachedir: .pytest_cache
# rootdir: /tmp/claude-0/-home-claude/186a30d5-aecd-5688-830a-09fb4cd892bd/scratchpad/perf
# collecting ... collected 27 items
# 
# test_performance_page.py::test_total_return_and_days PASSED              [  3%]
# test_performance_page.py::test_worst_fall_is_peak_to_trough PASSED       [  7%]
# test_performance_page.py::test_no_fall_means_zero PASSED                 [ 11%]
# test_performance_page.py::test_volatility_of_constant_growth_is_zero PASSED [ 14%]
# test_performance_page.py::test_volatility_matches_definition PASSED      [ 18%]
# test_performance_page.py::test_too_short_for_volatility_is_none_not_a_number PASSED [ 22%]
# test_performance_page.py::test_annualized_return_needs_enough_days_and_matches_formula PASSED [ 25%]
# test_performance_page.py::test_distance_to_floor PASSED                  [ 29%]
# test_performance_page.py::test_floor_breached_gives_negative_distance PASSED [ 33%]
# test_performance_page.py::test_no_floor_means_no_floor_fields PASSED     [ 37%]
# test_performance_page.py::test_rejects_bad_input PASSED                  [ 40%]
# test_performance_page.py::test_align_cuts_to_common_dates_and_rebases_to_start_value PASSED [ 44%]
# test_performance_page.py::test_align_with_no_overlap_is_an_error PASSED  [ 48%]
# test_performance_page.py::test_render_says_paper_and_gives_dates_and_sample_length PASSED [ 51%]
# test_performance_page.py::test_render_short_sample_shows_na_and_a_warning_not_numbers PASSED [ 55%]
# test_performance_page.py::test_render_is_free_of_nan_and_none PASSED     [ 59%]
# test_performance_page.py::test_render_shows_floor_row_only_when_a_floor_exists PASSED [ 62%]
# test_performance_page.py::test_long_sample_shows_annualized_and_no_short_warning PASSED [ 66%]
# test_performance_page.py::test_benchmark_series_comes_from_prices_from_start_date PASSED [ 70%]
# test_performance_page.py::test_benchmark_missing_ticker_is_an_error PASSED [ 74%]
# test_performance_page.py::test_start_after_last_price_is_an_error PASSED [ 77%]
# test_performance_page.py::test_read_actual_equity_csv PASSED             [ 81%]
# test_performance_page.py::test_read_equity_csv_bad_files_are_errors PASSED [ 85%]
# test_performance_page.py::test_build_report_writes_markdown_chart_and_json PASSED [ 88%]
# test_performance_page.py::test_build_report_never_overwrites_silently_it_replaces_only_its_own_files PASSED [ 92%]
# test_performance_page.py::test_duplicate_dates_are_rejected PASSED       [ 96%]
# test_performance_page.py::test_has_live_days_needs_two_prices_from_the_start PASSED [100%]
# 
# ============================== 27 passed in 0.84s ==============================
