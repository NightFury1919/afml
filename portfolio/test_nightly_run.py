"""Tests for nightly_run.py. The price update and the scoring are replaced by fakes so no network or model is needed."""
from datetime import datetime

import numpy as np
import pandas as pd
import pytest

import nightly_run as nr

NOW = lambda: datetime(2026, 10, 20, 21, 0, 0)
FIRST = "2026-10-16"


def make_snapshot(folder, last, start="2026-10-12"):
    idx = pd.bdate_range(start, last)
    df = pd.DataFrame({"AAA": np.arange(len(idx), dtype=float) + 10, "BBB": np.arange(len(idx), dtype=float) + 20},
                      index=idx)
    path = folder / f"prices_daily_asof_{idx[-1].date()}.csv"
    df.to_csv(path, index_label="date")
    return path


class Recorder:
    def __init__(self, fail_on=()):
        self.calls, self.fail_on = [], set(fail_on)

    def __call__(self, snap, day, shadow_folder):
        if str(day.date()) in self.fail_on:
            raise ValueError("boom")
        self.calls.append((snap.name, str(day.date())))
        out = shadow_folder / f"scores_asof_{day.date()}.csv"
        out.parent.mkdir(parents=True, exist_ok=True)
        out.write_text("x")
        return out


@pytest.fixture
def dirs(tmp_path):
    shadow = tmp_path / "shadow"
    shadow.mkdir()
    return tmp_path, shadow


def ok_update(folder):
    return "added 1 day(s)"


def test_missing_dates_starts_on_first_date_and_skips_logged(dirs):
    folder, shadow = dirs
    snap = make_snapshot(folder, "2026-10-20")
    (shadow / "scores_asof_2026-10-16.csv").write_text("x")
    got = [str(d.date()) for d in nr.missing_dates(snap, shadow, FIRST)]
    assert got == ["2026-10-19", "2026-10-20"]          # nothing before 10-16 and 10-16 already logged


def test_missing_dates_nothing_before_first_even_if_in_file(dirs):
    folder, shadow = dirs
    snap = make_snapshot(folder, "2026-10-15")
    assert nr.missing_dates(snap, shadow, FIRST) == []


def test_scores_each_missing_day_oldest_first(dirs):
    folder, shadow = dirs
    snap = make_snapshot(folder, "2026-10-20")
    rec = Recorder()
    code, lines = nr.nightly(folder, shadow, ok_update, rec, first_asof=FIRST, now_fn=NOW)
    assert code == 0
    assert [d for _, d in rec.calls] == ["2026-10-16", "2026-10-19", "2026-10-20"]
    assert all(s == snap.name for s, _ in rec.calls)
    assert sum("scored" in l for l in lines) == 3


def test_second_run_same_night_does_nothing(dirs):
    folder, shadow = dirs
    make_snapshot(folder, "2026-10-20")
    nr.nightly(folder, shadow, ok_update, Recorder(), first_asof=FIRST, now_fn=NOW)
    rec = Recorder()
    code, lines = nr.nightly(folder, shadow, ok_update, rec, first_asof=FIRST, now_fn=NOW)
    assert code == 0 and rec.calls == []
    assert "nothing to score" in lines[-1]


def test_weekend_with_no_new_bar_is_quiet_success(dirs):
    folder, shadow = dirs
    make_snapshot(folder, "2026-10-16")
    nr.nightly(folder, shadow, ok_update, Recorder(), first_asof=FIRST, now_fn=NOW)
    rec = Recorder()
    code, _ = nr.nightly(folder, shadow, ok_update, rec, first_asof=FIRST, now_fn=NOW)
    assert code == 0 and rec.calls == []


def test_failed_update_is_logged_exit_1_but_still_scores_from_disk(dirs):
    folder, shadow = dirs
    make_snapshot(folder, "2026-10-19")

    def bad_update(_):
        raise ConnectionError("no internet")

    rec = Recorder()
    code, lines = nr.nightly(folder, shadow, bad_update, rec, first_asof=FIRST, now_fn=NOW)
    assert code == 1
    assert any("PRICE UPDATE FAILED" in l and "no internet" in l for l in lines)
    assert [d for _, d in rec.calls] == ["2026-10-16", "2026-10-19"]


def test_failed_update_with_nothing_new_still_exit_1(dirs):
    folder, shadow = dirs
    make_snapshot(folder, "2026-10-16")
    (shadow / "scores_asof_2026-10-16.csv").write_text("x")

    def bad_update(_):
        raise ConnectionError("down")

    code, _ = nr.nightly(folder, shadow, bad_update, Recorder(), first_asof=FIRST, now_fn=NOW)
    assert code == 1


def test_one_failing_day_does_not_stop_later_days(dirs):
    folder, shadow = dirs
    make_snapshot(folder, "2026-10-20")
    rec = Recorder(fail_on={"2026-10-19"})
    code, lines = nr.nightly(folder, shadow, ok_update, rec, first_asof=FIRST, now_fn=NOW)
    assert code == 1
    assert [d for _, d in rec.calls] == ["2026-10-16", "2026-10-20"]
    assert any("SCORING FAILED for 2026-10-19" in l for l in lines)


def test_no_snapshot_at_all_exit_1(dirs):
    folder, shadow = dirs
    code, lines = nr.nightly(folder, shadow, ok_update, Recorder(), first_asof=FIRST, now_fn=NOW)
    assert code == 1 and any("NO PRICE SNAPSHOT" in l for l in lines)


def test_uses_newest_snapshot(dirs):
    folder, shadow = dirs
    make_snapshot(folder, "2026-10-16")
    newest = make_snapshot(folder, "2026-10-19")
    rec = Recorder()
    nr.nightly(folder, shadow, ok_update, rec, first_asof=FIRST, now_fn=NOW)
    assert {s for s, _ in rec.calls} == {newest.name}


def test_log_lines_carry_timestamp_and_append(dirs):
    folder, shadow = dirs
    make_snapshot(folder, "2026-10-16")
    _, lines = nr.nightly(folder, shadow, ok_update, Recorder(), first_asof=FIRST, now_fn=NOW)
    assert all(l.startswith("2026-10-20 21:00:00  ") for l in lines)
    log = shadow / "nightly_log.txt"
    nr.append_log(log, lines)
    nr.append_log(log, ["second"])
    text = log.read_text(encoding="utf-8").splitlines()
    assert text[:len(lines)] == lines and text[-1] == "second"


def test_existing_score_file_is_never_rewritten(dirs):
    folder, shadow = dirs
    make_snapshot(folder, "2026-10-16")
    (shadow / "scores_asof_2026-10-16.csv").write_text("original")
    nr.nightly(folder, shadow, ok_update, Recorder(), first_asof=FIRST, now_fn=NOW)
    assert (shadow / "scores_asof_2026-10-16.csv").read_text() == "original"


def test_real_scoring_function_is_the_default():
    import shadow_scoring
    assert nr.nightly.__defaults__[0] is shadow_scoring.run_shadow          # run_fn default
    assert nr.FIRST_ASOF == shadow_scoring.FIRST_ASOF == "2026-10-16"


# ---------------------------------------------------------------------------------------------- results
# Sandbox (py3.10 venv): 13 passed. Mutation checks: 10 mutants (scores days before 2026-10-16, ignores existing logs,
# ignores "nothing to score", update failure exit 0, stops after one failed day, nothing-new hides a failed update,
# newest-first order, no-snapshot exit 0, log overwritten, wrong first date): all caught.
# Not yet run in the mlfinlab env or against Yahoo/Task Scheduler on Ethan's machine.
