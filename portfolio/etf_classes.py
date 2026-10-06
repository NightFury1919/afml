"""Asset-class map for the 48 ranked ETFs (frozen design, 6 classes).

Used when each ETF is ranked and labeled against its own class instead of one pool of 48.
SH (the hedge) has no class and is never ranked.

    US equity               20   SPY QQQ MDY IWM IWD IWF, the nine XL sectors, XBI KRE XHB GDX, IYR
    International developed  7   EFA EWJ EWG EWU EWC EWA EWL
    Emerging markets         6   EEM EWZ EWT EWY FXI EWW
    Bonds                    8   SHY IEF TLT TIP LQD HYG EMB MUB
    Commodities              4   GLD SLV DBC DBA
    Currencies               3   UUP FXE FXY

Measured on the real returns, ranking within these classes raises effective breadth from
16.5 (one pool) to 30.2. Real estate (IYR) sits with US equity so that no class has one member.
"""
from collections import Counter

CLASS_NAMES = ["US equity", "International developed", "Emerging markets",
               "Bonds", "Commodities", "Currencies"]

_MEMBERS = {
    "US equity": "SPY QQQ MDY IWM IWD IWF XLB XLE XLF XLI XLK XLP XLU XLV XLY XBI KRE XHB GDX IYR",
    "International developed": "EFA EWJ EWG EWU EWC EWA EWL",
    "Emerging markets": "EEM EWZ EWT EWY FXI EWW",
    "Bonds": "SHY IEF TLT TIP LQD HYG EMB MUB",
    "Commodities": "GLD SLV DBC DBA",
    "Currencies": "UUP FXE FXY",
}

CLASS_OF = {ticker: cls for cls in CLASS_NAMES for ticker in _MEMBERS[cls].split()}


def class_sizes():
    counts = Counter(CLASS_OF.values())
    return {cls: counts[cls] for cls in CLASS_NAMES}


def validate_classes(tickers):
    """Raise ValueError if any ticker has no class."""
    missing = [t for t in tickers if t not in CLASS_OF]
    if missing:
        raise ValueError(f"no asset class for: {missing}")

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
