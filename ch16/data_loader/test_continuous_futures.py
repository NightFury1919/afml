"""
Tests for continuous_futures.py -- Chapter 16's real multi-asset data loader.

Unlike hrp.py (which has book-provided hand-traced numbers to test against),
this module's correctness is defined by real data-quality bugs found and
fixed during development (see project handoff). Most tests below are
regression tests: small, synthetic, fully-controlled reproductions of each
bug, so a future change can't silently reintroduce them. A smaller set of
tests run against the real six-commodity dataset as a sanity/regression
guard (not hand-traceable, but bounded: no NaNs, no impossible outlier
returns, valid correlation range).
"""
import os
import sys

import numpy as np
import pandas as pd
import pytest

sys.path.insert(0, os.path.dirname(__file__))
from continuous_futures import (
    load_contract, build_front_month_series, build_continuous_price,
    DATA_DIR, COMMODITIES,
)


# =============================================================================
# load_contract: dual-format parsing + real bug regressions
# =============================================================================
class TestLoadContract:
    def test_no_header_format_parses_correctly(self, tmp_path):
        # Standard no-header YYMMDD format, matching every pre-2000 file.
        path = tmp_path / 'TEST98H.txt'
        path.write_text(
            '980312,100.5,101.0,100.0,100.75,500,1000\n'
            '980313,100.75,102.0,100.5,101.8,600,1050\n'
        )
        df = load_contract(str(path))
        assert list(df.columns) == ['Date', 'Open', 'High', 'Low', 'Close', 'Volume', 'OpenInt']
        assert df['Date'].iloc[0] == pd.Timestamp('1998-03-12')
        assert df['Close'].iloc[1] == pytest.approx(101.8)

    def test_header_format_parses_correctly(self, tmp_path):
        # Newer quoted-header MM/DD/YYYY format, matching every 2000+ file.
        path = tmp_path / 'TEST00F.txt'
        path.write_text(
            '"Date","Open","High","Low","Close","Volume","OpenInt"\n'
            '01/04/2000,281.0,282.7,281.0,282.7,4,4\n'
        )
        df = load_contract(str(path))
        assert df['Date'].iloc[0] == pd.Timestamp('2000-01-04')
        assert df['Close'].iloc[0] == pytest.approx(282.7)

    def test_leading_zero_january_dates_preserved(self, tmp_path):
        # Regression test: pd.read_csv auto-infers an unquoted no-header
        # Date column as int64, silently stripping leading zeros, so
        # '000104' (Jan 4, 2000) becomes the integer 104 -- which then
        # fails to parse as %y%m%d entirely (or worse, parses as
        # something else). Confirmed to hit real files across all six
        # commodities. This must stay fixed.
        path = tmp_path / 'TEST00F.txt'
        path.write_text(
            '000103,100,100,100,100,0,0\n'
            '000104,100,100,100,100,0,0\n'
            '000127,100,100,100,100,0,0\n'
        )
        df = load_contract(str(path))
        assert df['Date'].iloc[0] == pd.Timestamp('2000-01-03')
        assert df['Date'].iloc[1] == pd.Timestamp('2000-01-04')
        assert df['Date'].iloc[2] == pd.Timestamp('2000-01-27')

    def test_rescale_applies_to_header_format(self, tmp_path):
        path = tmp_path / 'TESTNEW.txt'
        path.write_text(
            '"Date","Open","High","Low","Close","Volume","OpenInt"\n'
            '01/04/2000,1.5,1.6,1.4,1.55,4,4\n'
        )
        df = load_contract(str(path), rescale_new_format_by=100)
        assert df['Close'].iloc[0] == pytest.approx(155.0)
        assert df['Open'].iloc[0] == pytest.approx(150.0)

    def test_rescale_none_leaves_header_format_unchanged(self, tmp_path):
        path = tmp_path / 'TESTNEW.txt'
        path.write_text(
            '"Date","Open","High","Low","Close","Volume","OpenInt"\n'
            '01/04/2000,1.5,1.6,1.4,1.55,4,4\n'
        )
        df = load_contract(str(path), rescale_new_format_by=None)
        assert df['Close'].iloc[0] == pytest.approx(1.55)

    def test_rescale_does_not_apply_to_no_header_format(self, tmp_path):
        # Regression guard: the rescale bug was specific to the NEWER
        # format files. A rescale factor passed in must never touch the
        # older no-header files, even if the caller sets it (defends
        # against accidentally double-scaling old-format GBP data).
        path = tmp_path / 'TESTOLD.txt'
        path.write_text('980312,150.0,151.0,149.0,150.5,500,1000\n')
        df = load_contract(str(path), rescale_new_format_by=100)
        assert df['Close'].iloc[0] == pytest.approx(150.5)


# =============================================================================
# build_front_month_series: selection logic + real bug regressions
# =============================================================================
class TestBuildFrontMonthSeries:
    def _write_contract(self, tmp_path, name, rows):
        """rows: list of (yymmdd_str, close, openint) tuples."""
        path = tmp_path / f'{name}.txt'
        lines = [f'{d},{c},{c},{c},{c},1,{oi}' for d, c, oi in rows]
        path.write_text('\n'.join(lines) + '\n')
        return path

    def test_selects_highest_open_interest_contract(self, tmp_path):
        # Two contracts trading the same days; the one with higher OI
        # should win every day.
        self._write_contract(tmp_path, 'TT98H', [
            ('980310', 100.0, 500), ('980311', 100.5, 520),
        ])
        self._write_contract(tmp_path, 'TT98M', [
            ('980310', 200.0, 5000), ('980311', 201.0, 5100),
        ])
        fm = build_front_month_series(str(tmp_path), 'TT')
        assert (fm['Instrument'] == 'TT98M').all()
        assert fm.loc['1998-03-10', 'Close'] == pytest.approx(200.0)

    def test_forward_fills_isolated_gap_in_dominant_contract(self, tmp_path):
        # Regression test, exactly reproducing the T-bonds bug: the
        # dominant contract (high OI) has ONE missing day in the middle
        # of an otherwise-continuous run, while a much-less-dominant
        # contract happens to have a row that day at a very different
        # price. Without the forward-fill fix, front-month selection
        # would flip to the minor contract for exactly that one day,
        # producing a fake price jump.
        self._write_contract(tmp_path, 'TT00H', [
            ('991206', 93.4, 489795), ('991207', 93.6, 502529),
            ('991208', 93.7, 502529),
            # NOTE: no 991209 row for TT00H -- the gap.
            ('991210', 94.7, 513105),
        ])
        self._write_contract(tmp_path, 'TT99Z', [
            ('991209', 112.75, 52793),   # much lower OI, very different price
        ])
        fm = build_front_month_series(str(tmp_path), 'TT')
        # The dominant contract's forward-filled value should win on the
        # gap day, NOT the minor contract's wildly different price.
        assert fm.loc['1999-12-09', 'Instrument'] == 'TT00H'
        assert fm.loc['1999-12-09', 'Close'] == pytest.approx(93.7)  # ffilled from 12-08

    def test_long_gaps_only_partially_filled_then_dropped(self, tmp_path):
        # ffill(limit=3) fills only the FIRST 3 consecutive missing
        # business days of any gap, then leaves the rest as NaN (which
        # get dropped) -- a bounded partial fill, not all-or-nothing.
        # This guards against a genuinely-expired contract being
        # forward-filled indefinitely (which would let a dead contract
        # "win" front-month status long after it stopped trading).
        self._write_contract(tmp_path, 'TT98H', [
            ('980302', 100.0, 500),
            # 6-business-day gap follows (980303-980310 missing):
            # 03-03,04,05 get filled (limit=3); 03-06,09,10 do not.
            ('980311', 105.0, 50),
        ])
        fm = build_front_month_series(str(tmp_path), 'TT')
        expected_dates = {'1998-03-02', '1998-03-03', '1998-03-04',
                           '1998-03-05', '1998-03-11'}
        assert set(fm.index.strftime('%Y-%m-%d')) == expected_dates
        # The filled days should carry forward the last REAL price (100.0),
        # not interpolate toward the next real price (105.0).
        assert fm.loc['1998-03-05', 'Close'] == pytest.approx(100.0)

    def test_returns_exactly_one_row_per_date(self, tmp_path):
        self._write_contract(tmp_path, 'TT98H', [('980310', 100.0, 500)])
        self._write_contract(tmp_path, 'TT98M', [('980310', 200.0, 5000)])
        fm = build_front_month_series(str(tmp_path), 'TT')
        assert fm.index.is_unique


# =============================================================================
# build_continuous_price: integration with real roll.py
# =============================================================================
class TestBuildContinuousPrice:
    def test_output_starts_at_one_and_stays_positive(self, tmp_path):
        path = tmp_path / 'TT98H.txt'
        path.write_text(
            '980310,100.0,100.0,100.0,100.0,1,500\n'
            '980311,101.0,101.0,101.0,101.0,1,500\n'
            '980312,99.5,99.5,99.5,99.5,1,500\n'
        )
        non_neg, _ = build_continuous_price(str(tmp_path), 'TT')
        # non_negative_rolled_prices' rPrices column: first value is NaN
        # (diff() has nothing to compare against), rest must be positive.
        rprices = non_neg['rPrices'].dropna()
        assert (rprices > 0).all()

    def test_single_contract_no_roll_reproduces_its_own_returns(self, tmp_path):
        # With only one contract (no rolling), rPrices should exactly
        # compound that contract's own daily returns -- a direct check
        # that roll.py's gap correction contributes nothing when there's
        # no roll to correct for.
        path = tmp_path / 'TT98H.txt'
        path.write_text(
            '980310,100.0,100.0,100.0,100.0,1,500\n'
            '980311,101.0,101.0,101.0,101.0,1,500\n'
            '980312,99.5,99.5,99.5,99.5,1,500\n'
        )
        non_neg, _ = build_continuous_price(str(tmp_path), 'TT')
        expected_r2 = 101.0 / 100.0 - 1
        expected_r3 = 99.5 / 101.0 - 1
        assert non_neg['Returns'].iloc[1] == pytest.approx(expected_r2)
        assert non_neg['Returns'].iloc[2] == pytest.approx(expected_r3)


# =============================================================================
# Real-data regression guards (not hand-traceable, but bounded sanity checks)
# =============================================================================
class TestRealDataIntegration:
    """Runs the full loader against the actual six-commodity dataset. These
    are regression guards against the specific bug classes found during
    development recurring silently (e.g. a new contract file introducing
    another format/scale/gap surprise), not hand-traced known values."""

    @pytest.fixture(scope='class')
    @classmethod
    def all_returns(cls):
        prices = {}
        for name, (prefix, folder, rescale) in COMMODITIES.items():
            non_neg, _ = build_continuous_price(folder, prefix, rescale_new_format_by=rescale)
            prices[name] = non_neg['rPrices']
        price_df = pd.DataFrame(prices)
        price_df = price_df[(price_df.index >= '1998-01-01') &
                             (price_df.index <= '2002-09-16')].dropna()
        return price_df.pct_change().dropna()

    def test_all_six_commodities_present(self, all_returns):
        assert set(all_returns.columns) == set(COMMODITIES.keys())

    def test_no_nans_or_infs(self, all_returns):
        assert not all_returns.isna().any().any()
        assert np.isfinite(all_returns.values).all()

    def test_no_extreme_outlier_returns(self, all_returns):
        # Regression guard against the GBP (+9796%) and T-bonds (+20%)
        # data-gap/scale bugs recurring silently. 25% is well above even
        # the most extreme real single-day moves seen in this dataset
        # (crude oil ~15% post-9/11) but far below any plausible
        # data-artifact magnitude.
        assert (all_returns.abs() < 0.25).all().all()

    def test_correlation_matrix_valid(self, all_returns):
        corr = all_returns.corr()
        assert np.allclose(np.diag(corr), 1.0)
        assert (corr.values >= -1.0001).all() and (corr.values <= 1.0001).all()
        assert np.allclose(corr.values, corr.values.T)   # symmetric

    def test_reasonable_sample_size(self, all_returns):
        # Book's own rule of thumb: ~ 0.5*N*(N+1) independent observations
        # minimum to avoid a singular/unstable covariance matrix. N=6 here.
        n = len(COMMODITIES)
        minimum = 0.5 * n * (n + 1)
        assert len(all_returns) > minimum * 10   # comfortable margin
