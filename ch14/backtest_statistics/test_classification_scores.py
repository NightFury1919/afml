"""
TDD suite for classification_scores, hand-traced against AFML Table 14.1's
four degenerate cases of binary classification.
"""
# --- import the module under test ------------------------------------------
# Derive the repo root from __file__, put it on sys.path, then import
# fully-qualified.
#
# LOAD-BEARING (2026-07-22): this used to be a bare 'from
# classification_scores import ...' with no sys.path handling, which the
# 2026-07-21 audit flagged as fragile -- it only passes when pytest is
# invoked from inside this module's own folder; from the repo root it
# raises ModuleNotFoundError (both this folder and ch14/ have __init__.py,
# per the audit's verified truth table). See ch10/bet_sizing/test_bet_sizing.py
# or ch13/otr/test_otr.py for the pattern this was generalized from.
import os
import sys

ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '..'))
if ROOT not in sys.path:
    sys.path.insert(0, ROOT)

import numpy as np
import pytest
from ch14.backtest_statistics.classification_scores import classification_scores  # noqa: E402


class TestClassificationScores:
    def test_observed_all_ones(self):
        """TN=FP=0: accuracy==recall, precision=1, recall/F1 in [0,1]."""
        y_true = [1, 1, 1, 1]
        y_pred = [1, 1, 0, 1]  # 3 correct, 1 missed
        s = classification_scores(y_true, y_pred)
        assert s['tn'] == 0 and s['fp'] == 0
        assert s['accuracy'] == pytest.approx(0.75)
        assert s['precision'] == pytest.approx(1.0)
        assert s['recall'] == pytest.approx(0.75)
        assert s['accuracy'] == pytest.approx(s['recall'])  # book's stated identity
        assert s['f1'] == pytest.approx(2 * 1.0 * 0.75 / (1.0 + 0.75))

    def test_observed_all_zeros(self):
        """TP=FN=0: precision=0 (defined), recall=NaN, F1=NaN."""
        y_true = [0, 0, 0, 0]
        y_pred = [0, 1, 0, 1]
        s = classification_scores(y_true, y_pred)
        assert s['tp'] == 0 and s['fn'] == 0
        assert s['accuracy'] == pytest.approx(0.5)
        assert s['precision'] == pytest.approx(0.0)
        assert np.isnan(s['recall'])
        assert np.isnan(s['f1'])

    def test_predicted_all_ones(self):
        """TN=FN=0: accuracy==precision, recall=1, precision/F1 in [0,1]."""
        y_true = [1, 1, 0, 0]
        y_pred = [1, 1, 1, 1]
        s = classification_scores(y_true, y_pred)
        assert s['tn'] == 0 and s['fn'] == 0
        assert s['accuracy'] == pytest.approx(0.5)
        assert s['precision'] == pytest.approx(0.5)
        assert s['recall'] == pytest.approx(1.0)
        assert s['accuracy'] == pytest.approx(s['precision'])  # book's stated identity
        assert s['f1'] == pytest.approx(2 * 0.5 * 1.0 / (0.5 + 1.0))

    def test_predicted_all_zeros(self):
        """TP=FP=0: precision=NaN, recall=0 (defined), F1=NaN."""
        y_true = [1, 1, 0, 0]
        y_pred = [0, 0, 0, 0]
        s = classification_scores(y_true, y_pred)
        assert s['tp'] == 0 and s['fp'] == 0
        assert s['accuracy'] == pytest.approx(0.5)
        assert np.isnan(s['precision'])
        assert s['recall'] == pytest.approx(0.0)
        assert np.isnan(s['f1'])

    def test_neg_log_loss_included_when_proba_given(self):
        y_true = [0, 1, 0, 1]
        y_pred = [0, 1, 0, 0]
        y_proba = [[0.9, 0.1], [0.2, 0.8], [0.7, 0.3], [0.6, 0.4]]
        s = classification_scores(y_true, y_pred, y_proba=y_proba)
        assert 'neg_log_loss' in s
        assert s['neg_log_loss'] < 0  # negative log-loss is <=0 by construction
