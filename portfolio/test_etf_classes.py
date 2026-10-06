from etf_classes import CLASS_OF, CLASS_NAMES, class_sizes, validate_classes


def universe_tickers():
    out = []
    for line in open("universe.txt", encoding="utf-8").read().splitlines():
        t = line.split("#", 1)[0].strip().upper()
        if t:
            out.append(t)
    return out


def test_every_ranked_etf_has_exactly_one_class_and_the_hedge_has_none():
    ranked = [t for t in universe_tickers() if t != "SH"]
    assert len(ranked) == 48
    assert sorted(CLASS_OF) == sorted(ranked)
    assert "SH" not in CLASS_OF
    assert set(CLASS_OF.values()) == set(CLASS_NAMES)


def test_class_sizes_are_the_frozen_design():
    assert class_sizes() == {"US equity": 20, "International developed": 7, "Emerging markets": 6,
                             "Bonds": 8, "Commodities": 4, "Currencies": 3}


def test_known_assignments():
    assert CLASS_OF["SPY"] == "US equity" and CLASS_OF["XLE"] == "US equity" and CLASS_OF["IYR"] == "US equity"
    assert CLASS_OF["EWJ"] == "International developed" and CLASS_OF["EFA"] == "International developed"
    assert CLASS_OF["EEM"] == "Emerging markets" and CLASS_OF["FXI"] == "Emerging markets"
    assert CLASS_OF["TLT"] == "Bonds" and CLASS_OF["EMB"] == "Bonds"
    assert CLASS_OF["GLD"] == "Commodities" and CLASS_OF["DBA"] == "Commodities"
    assert CLASS_OF["UUP"] == "Currencies" and CLASS_OF["FXY"] == "Currencies"


def test_validate_rejects_an_unclassified_ticker_and_accepts_the_universe():
    import pytest
    validate_classes(sorted(CLASS_OF))
    with pytest.raises(ValueError):
        validate_classes(sorted(CLASS_OF) + ["ZZZ"])

# ---------------------------------------------------------------------------
# TDD RESULTS (pytest, 2026-10-06, mlfinlab env: Python 3.10.20, pytest 9.0.3)
# $ cd portfolio ; pytest test_sizing.py test_rebalance.py test_positive_control_etf.py test_etf_classes.py test_etf_features.py test_etf_labels.py test_panel_data.py -v
# Seven test files were run together (88 items); the lines for this file's tests are shown.
#
# platform win32 -- Python 3.10.20, pytest-9.0.3, pluggy-1.6.0
# rootdir: C:\ws\AFML\portfolio
# collected 88 items
#
# test_etf_classes.py::test_every_ranked_etf_has_exactly_one_class_and_the_hedge_has_none PASSED [ 48%]
# test_etf_classes.py::test_class_sizes_are_the_frozen_design PASSED       [ 50%]
# test_etf_classes.py::test_known_assignments PASSED                       [ 51%]
# test_etf_classes.py::test_validate_rejects_an_unclassified_ticker_and_accepts_the_universe PASSED [ 52%]
#
# 88 passed in 17.89s (all seven files)
# ---------------------------------------------------------------------------
