"""
Chapter 18, Section 18.8.4 -- Market microstructure adverse-selection
feature.

WHY (plain-English, before the math):
A market maker gets hurt when the person on the other side of a trade
knows something they don't (adverse selection). The book's proposed
signal for this: look at how PREDICTABLE the buy/sell order-flow
imbalance has been recently. If recent order flow has been highly
repetitive (low entropy), a market maker CAN predict it and isn't
getting picked off. If recent order flow has been unpredictable (high
entropy), informed traders may be hiding in the noise. This module
builds that pipeline end-to-end: real per-bar buy/sell volume (Ch19) ->
order-flow-imbalance ratio -> quantized into a discrete alphabet
(reusing 18.5.2's quantile encoding) -> Kontoyiannis LZ entropy
(18.1-18.4) computed on a ROLLING basis -> empirical CDF of that
entropy series, which is the final per-bar feature.

FORMULA-ONLY, WORKFLOW-ONLY section -- the book (Sec 18.8.4) describes
six prose steps but gives no printed code and, critically, no algorithm
for turning ONE entropy estimate into "the time series {F[H[Xtau]]}"
its own step 6 requires. The rolling-window mechanics below are this
project's own judgment call, not a literal book formula -- documented
explicitly per this project's convention, not silently resolved.

DESIGN DECISIONS (confirmed with Ethan 2026-08-04):
  - n_quantiles=5 (default): number of letters for the order-flow-
    imbalance quantization (book's own prose uses a generic "q" without
    committing to a value).
  - roll_window=30 (default, must be even -- konto's own expanding-
    window mode requires len(msg)%2==0): trailing window of bars used
    to compute each entropy estimate H_tau. With ~249 real bars this
    leaves ~220 output points -- a genuine "how much history is enough"
    tension, same caveat class as Ch13/Ch15/Ch17 Part C.
"""
import numpy as np
import pandas as pd

from encoding_schemes import quantile_encode
from entropy_estimators import konto


def order_flow_imbalance(buy_volume, sell_volume):
    """
    vB_tau = VB_tau / (VB_tau + VS_tau), the fraction of a bar's volume
    that was buy-initiated (Sec 18.8.4, step 1).

    Bars with zero total volume (both buy and sell volume are 0) are a
    degenerate edge case the book doesn't address -- guarded here by
    assigning the neutral value 0.5 rather than dividing by zero.

    Returns
    -------
    pd.Series of float in [0, 1], same length/index as the inputs.
    """
    buy_volume = pd.Series(buy_volume, dtype=float).reset_index(drop=True)
    sell_volume = pd.Series(sell_volume, dtype=float).reset_index(drop=True)
    if len(buy_volume) != len(sell_volume):
        raise ValueError("buy_volume and sell_volume must be the same length")
    total = buy_volume + sell_volume
    vB = pd.Series(np.where(total > 0, buy_volume / total.replace(0, np.nan), 0.5),
                    index=buy_volume.index)
    return vB


def adverse_selection_feature(buy_volume, sell_volume, n_quantiles=5, roll_window=30):
    """
    Full Sec 18.8.4 pipeline: order-flow imbalance -> quantile encoding
    -> rolling Kontoyiannis entropy -> empirical CDF.

    Parameters
    ----------
    buy_volume, sell_volume : array-like, per-bar volumes (real Ch19
        columns: bars['BuyVolume'], bars['SellVolume']).
    n_quantiles : int, quantization alphabet size (step 2-4 of 18.8.4;
        reuses 18.5.2's quantile_encode, full-sample per this project's
        small-sample convention).
    roll_window : int, must be even. Trailing window (in bars) used for
        each Kontoyiannis entropy estimate (step 5).

    Returns
    -------
    pd.Series, index = original bar index from roll_window-1 through the
        last bar, values = F[H[X_tau]] in [0,1] (step 6: the empirical
        CDF of the rolling entropy series -- 0 = least entropic window
        seen, 1 = most entropic window seen, RELATIVE TO THIS SAMPLE
        ONLY, same in-sample caveat as quantile_encode itself).
    """
    if roll_window % 2 != 0:
        raise ValueError(
            "roll_window must be even -- konto's expanding-window mode "
            "requires len(msg) % 2 == 0."
        )
    if roll_window < 4:
        raise ValueError("roll_window must be >= 4 for a meaningful entropy estimate")

    vB = order_flow_imbalance(buy_volume, sell_volume)
    n = len(vB)
    if n < roll_window:
        raise ValueError(
            f"need at least roll_window={roll_window} bars, got {n}"
        )

    quantized = quantile_encode(vB.values, n_letters=n_quantiles)

    h_values = []
    end_positions = range(roll_window, n + 1)  # tau = last bar in each window
    for tau in end_positions:
        window_msg = quantized[tau - roll_window:tau]
        h = konto(window_msg)['h']
        h_values.append(h)

    h_series = pd.Series(h_values)
    # Empirical CDF via percentile rank (average rank for ties, scaled
    # to (0,1]) -- standard, simple estimator of F[H[X_tau]].
    cdf = h_series.rank(pct=True)

    out_index = vB.index[roll_window - 1: n]
    return pd.Series(cdf.values, index=out_index, name='adverse_selection_cdf')


# -----------------------------------------------------------------------
# pytest -v output (sandbox, Python 3.12.3) -- 12/12 passed (this file's
# tests only; 45/45 when run together with test_entropy_estimators.py
# and test_encoding_schemes.py).
# Real-machine confirmation under mlfinlab (Python 3.10.20) still
# pending as of this commit -- see chapter README for status.
# -----------------------------------------------------------------------
# ============================= test session starts ==============================
# platform linux -- Python 3.12.3, pytest-9.1.1, pluggy-1.6.0 -- /usr/bin/python3
# rootdir: /home/claude/ch18/entropy_features
# collecting ... collected 12 items
#
# test_adverse_selection.py::TestOrderFlowImbalance::test_basic_ratio PASSED [  8%]
# test_adverse_selection.py::TestOrderFlowImbalance::test_all_buy_is_one PASSED [ 16%]
# test_adverse_selection.py::TestOrderFlowImbalance::test_all_sell_is_zero PASSED [ 25%]
# test_adverse_selection.py::TestOrderFlowImbalance::test_zero_total_volume_guarded_to_neutral PASSED [ 33%]
# test_adverse_selection.py::TestOrderFlowImbalance::test_mismatched_lengths_raise PASSED [ 41%]
# test_adverse_selection.py::TestAdverseSelectionFeatureValidation::test_odd_roll_window_rejected PASSED [ 50%]
# test_adverse_selection.py::TestAdverseSelectionFeatureValidation::test_too_small_roll_window_rejected PASSED [ 58%]
# test_adverse_selection.py::TestAdverseSelectionFeatureValidation::test_fewer_bars_than_window_rejected PASSED [ 66%]
# test_adverse_selection.py::TestAdverseSelectionFeatureCrossChecked::test_symmetric_alternating_flow_all_windows_tied PASSED [ 75%]
# test_adverse_selection.py::TestAdverseSelectionFeatureCrossChecked::test_output_length_and_index_alignment PASSED [ 83%]
# test_adverse_selection.py::TestAdverseSelectionFeatureCrossChecked::test_cdf_values_bounded_zero_to_one PASSED [ 91%]
# test_adverse_selection.py::TestAdverseSelectionFeatureCrossChecked::test_repetitive_flow_reads_lower_cdf_than_random_flow PASSED [100%]
#
# ============================== 12 passed in 0.54s ===============================
