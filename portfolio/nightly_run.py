"""Nightly job: pull fresh prices, then write the shadow score log for every trading day that has none yet.

Run from C:\\ws\\AFML\\portfolio in the mlfinlab env (Task Scheduler runs it for you, see nightly_setup.md):
    python nightly_run.py

What it does, in order:
  1. Updates the price snapshot (the same update daily_run.py does). If this fails (no internet, Yahoo down) the
     failure is logged and it carries on with the newest snapshot it already has.
  2. Takes the newest snapshot and lists its trading days from 2026-10-16 onward that have no
     shadow/scores_asof_<date>.csv. Nights the computer was off are caught up, oldest first: each day is scored using
     prices up to that day only, so a catch-up score equals what the night-of run would have produced; the log header
     names the snapshot actually used.
  3. Writes each missing log with shadow_scoring.run_shadow. Existing logs are never touched.
  4. Appends one line per event to shadow/nightly_log.txt. Exit code 0 = nothing wrong, 1 = something to look at.

It sends NO orders and does not run git. Committing the new logs stays a manual step.
"""
import argparse
import sys
from datetime import datetime
from pathlib import Path

import pandas as pd

from panel_data import load_prices
from shadow_scoring import FIRST_ASOF, run_shadow


def missing_dates(prices_path, shadow_folder, first_asof=FIRST_ASOF):
    """Trading days in the price file, on or after first_asof, with no score log yet (oldest first)."""
    first = pd.Timestamp(first_asof)
    have = {p.name for p in Path(shadow_folder).glob("scores_asof_*.csv")}
    days = [d for d in load_prices(prices_path).index if d >= first]
    return [d for d in days if f"scores_asof_{d.date()}.csv" not in have]


def nightly(folder, shadow_folder, update_fn, run_fn=run_shadow, snapshot_fn=None, first_asof=FIRST_ASOF,
            now_fn=datetime.now):
    """Do the nightly job. Returns (exit_code, list of log lines). Never raises for a failed update or a refused day."""
    lines, bad = [], False

    def say(text):
        lines.append(f"{now_fn():%Y-%m-%d %H:%M:%S}  {text}")

    try:
        say(f"price update: {update_fn(folder)}")
    except Exception as err:                          # no internet, Yahoo change, missing yfinance ...
        bad = True
        say(f"PRICE UPDATE FAILED ({type(err).__name__}: {err}); continuing with the newest snapshot on disk")
    if snapshot_fn is None:
        from update_prices import latest_snapshot as snapshot_fn
    try:
        snap = snapshot_fn(folder)
    except Exception as err:
        say(f"NO PRICE SNAPSHOT ({type(err).__name__}: {err}); nothing scored")
        return 1, lines
    todo = missing_dates(snap, shadow_folder, first_asof)
    if not todo:
        say(f"nothing to score ({Path(snap).name}; every trading day since {first_asof} has a log)")
        return (1 if bad else 0), lines
    for day in todo:
        try:
            path = run_fn(snap, day, shadow_folder)
            say(f"scored {day.date()} -> {Path(path).name} (from {Path(snap).name})")
        except Exception as err:                      # one bad day must not stop the later ones being tried
            bad = True
            say(f"SCORING FAILED for {day.date()} ({type(err).__name__}: {err})")
    return (1 if bad else 0), lines


def append_log(path, lines):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "a", encoding="utf-8") as f:
        f.write("\n".join(lines) + "\n")


def main(argv=None):
    here = Path(__file__).resolve().parent
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--folder", default=str(here), help="folder holding the prices_daily_asof_*.csv snapshots")
    ap.add_argument("--shadow-folder", default=str(here / "shadow"))
    args = ap.parse_args(argv)
    from daily_run import default_update
    code, lines = nightly(args.folder, args.shadow_folder, default_update)
    append_log(Path(args.shadow_folder) / "nightly_log.txt", lines)
    print("\n".join(lines))
    return code


if __name__ == "__main__":
    sys.exit(main())
