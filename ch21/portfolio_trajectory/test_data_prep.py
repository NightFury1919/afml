"""
TDD tests for AFML Chapter 21's real-data preparation module (data_prep.py).

Unit-level tests (parsing, front-month selection, horizon-window slicing)
use small synthetic fixture files with hand-traceable expected results --
this isolates the PLUMBING logic from any question about real-data content.
The final class runs the actual pipeline against the real gold/crude oil/
US T-bonds files and is skipped automatically if those folders aren't
present relative to the repo root (e.g. if this test file is copied
somewhere without the surrounding repo).
"""
import os
import sys

import numpy as np
import pandas as pd
import pytest

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import data_prep  # noqa: E402


# ---------------------------------------------------------------------------
# _parse_headerless_date -- century-cutoff rule
# ---------------------------------------------------------------------------
class TestParseHeaderlessDate:
    def test_1900s_date(self):
        assert data_prep._parse_headerless_date('770822') == pd.Timestamp('1977-08-22')

    def test_2000s_date(self):
        assert data_prep._parse_headerless_date('991103') == pd.Timestamp('1999-11-03')

    def test_2000s_date_low_year(self):
        assert data_prep._parse_headerless_date('000124') == pd.Timestamp('2000-01-24')

    def test_cutoff_boundary_low(self):
        # yy=30 -> 2030, the book-dataset-safe cutoff used by this module.
        assert data_prep._parse_headerless_date('300101') == pd.Timestamp('2030-01-01')

    def test_cutoff_boundary_high(self):
        # yy=31 -> 1931
        assert data_prep._parse_headerless_date('310101') == pd.Timestamp('1931-01-01')


# ---------------------------------------------------------------------------
# load_contract_file -- both real raw formats
# ---------------------------------------------------------------------------
class TestLoadContractFile:
    def test_headerless_format(self, tmp_path):
        f = tmp_path / 'AB99Z.txt'
        f.write_text('991103,294.1,295.0,293.0,294.5,10,20\n991104,294.4,295.5,293.5,294.9,15,25\n')
        df = data_prep.load_contract_file(str(f))
        assert list(df.index) == [pd.Timestamp('1999-11-03'), pd.Timestamp('1999-11-04')]
        assert df.iloc[0]['Close'] == 294.5
        assert df.iloc[0]['Instrument'] == 'AB99Z'
        assert df.iloc[1]['Volume'] == 15

    def test_headered_format(self, tmp_path):
        f = tmp_path / 'CD00H.txt'
        f.write_text(
            '"Date","Open","High","Low","Close","Volume","OpenInt"\n'
            '11/27/2001,99.1875,99.625,99.1875,99.625,1,0\n'
            '11/28/2001,99.625,99.625,99.625,99.625,0,0\n'
        )
        df = data_prep.load_contract_file(str(f))
        assert list(df.index) == [pd.Timestamp('2001-11-27'), pd.Timestamp('2001-11-28')]
        assert df.iloc[0]['Close'] == pytest.approx(99.625)
        assert df.iloc[0]['Instrument'] == 'CD00H'

    def test_sorted_ascending_even_if_file_is_not(self, tmp_path):
        f = tmp_path / 'EF01M.txt'
        f.write_text('010102,10,10,10,10,5,5\n010101,9,9,9,9,5,5\n')
        df = data_prep.load_contract_file(str(f))
        assert list(df.index) == sorted(df.index)


# ---------------------------------------------------------------------------
# load_commodity_panel / select_front_month
# ---------------------------------------------------------------------------
class TestFrontMonthSelection:
    def _write_contract(self, tmp_path, name, rows):
        f = tmp_path / f'{name}.txt'
        lines = [f'{d},{o},{h},{l},{c},{v},{oi}' for d, o, h, l, c, v, oi in rows]
        f.write_text('\n'.join(lines) + '\n')

    def test_picks_higher_volume_contract_on_overlapping_date(self, tmp_path):
        # Two contracts both trade on 010102: contract A has volume 5,
        # contract B has volume 50 -> B must be selected as front month.
        self._write_contract(tmp_path, 'AA01F', [('010101', 10, 10, 10, 10, 5, 5),
                                                    ('010102', 11, 11, 11, 11, 5, 5)])
        self._write_contract(tmp_path, 'BB01F', [('010102', 20, 20, 20, 20, 50, 50)])
        panel = data_prep.load_commodity_panel(str(tmp_path))
        front = data_prep.select_front_month(panel)
        assert front.loc[pd.Timestamp('2001-01-02'), 'Instrument'] == 'BB01F'
        assert front.loc[pd.Timestamp('2001-01-02'), 'Close'] == 20
        # 010101 only has one contract trading -> that one wins by default.
        assert front.loc[pd.Timestamp('2001-01-01'), 'Instrument'] == 'AA01F'

    def test_one_row_per_date(self, tmp_path):
        self._write_contract(tmp_path, 'AA01F', [('010101', 1, 1, 1, 1, 1, 1),
                                                    ('010102', 1, 1, 1, 1, 1, 1)])
        self._write_contract(tmp_path, 'BB01F', [('010102', 1, 1, 1, 1, 1, 1),
                                                    ('010103', 1, 1, 1, 1, 1, 1)])
        panel = data_prep.load_commodity_panel(str(tmp_path))
        front = data_prep.select_front_month(panel)
        assert len(front) == 3  # dates 01/01, 01/02, 01/03 -- one row each
        assert front.index.is_unique

    def test_raises_on_empty_directory(self, tmp_path):
        with pytest.raises(FileNotFoundError):
            data_prep.load_commodity_panel(str(tmp_path))


# ---------------------------------------------------------------------------
# build_horizon_params -- window slicing on synthetic returns
# ---------------------------------------------------------------------------
class TestBuildHorizonParams:
    def _synthetic_returns(self, n_days=200, n_assets=3, seed=0):
        rng = np.random.default_rng(seed)
        dates = pd.date_range('2020-01-01', periods=n_days, freq='B')
        data = rng.normal(loc=0.0005, scale=0.01, size=(n_days, n_assets))
        return pd.DataFrame(data, index=dates, columns=[f'asset_{i}' for i in range(n_assets)])

    def test_shape_and_horizon_count(self):
        returns_df = self._synthetic_returns(n_days=200)
        params, meta = data_prep.build_horizon_params(returns_df, horizon=2, lookback=50)
        assert len(params) == 2
        for p in params:
            assert p['mean'].shape == (3, 1)
            assert p['cov'].shape == (3, 3)
            assert p['c'].shape == (3,)
        assert len(meta['window_dates']) == 2

    def test_windows_are_non_overlapping_and_chronological(self):
        returns_df = self._synthetic_returns(n_days=200)
        _, meta = data_prep.build_horizon_params(returns_df, horizon=2, lookback=50)
        (start1, end1), (start2, end2) = meta['window_dates']
        assert start1 < end1 < start2 < end2  # strictly increasing, non-overlapping

    def test_mean_matches_manual_computation(self):
        # Hand-check: horizon params must equal a direct .mean()/.cov() call
        # on the exact same tail slice, no off-by-one drift.
        returns_df = self._synthetic_returns(n_days=100)
        params, meta = data_prep.build_horizon_params(returns_df, horizon=1, lookback=30)
        expected_window = returns_df.iloc[-30:]
        np.testing.assert_allclose(params[0]['mean'].ravel(), expected_window.mean().values)
        np.testing.assert_allclose(params[0]['cov'], expected_window.cov().values)

    def test_cost_scale_multiplies_volatility(self):
        returns_df = self._synthetic_returns(n_days=100)
        params_lo, _ = data_prep.build_horizon_params(returns_df, horizon=1, lookback=30, cost_scale=0.01)
        params_hi, _ = data_prep.build_horizon_params(returns_df, horizon=1, lookback=30, cost_scale=0.02)
        np.testing.assert_allclose(params_hi[0]['c'], 2 * params_lo[0]['c'])

    def test_raises_when_insufficient_history(self):
        returns_df = self._synthetic_returns(n_days=50)
        with pytest.raises(ValueError):
            data_prep.build_horizon_params(returns_df, horizon=2, lookback=30)


# ---------------------------------------------------------------------------
# Real-data integration check (skipped if the repo's input_data isn't found)
# ---------------------------------------------------------------------------
def _find_input_data_dirs():
    repo_root = data_prep._repo_root()
    base = os.path.join(repo_root, 'input_data')
    dirs = {
        'gold': os.path.join(base, 'gold'),
        'crude_oil': os.path.join(base, 'crude oil'),
        'us_t_bonds': os.path.join(base, 'US-T bonds'),
    }
    if all(os.path.isdir(d) for d in dirs.values()):
        return dirs
    return None


_REAL_DIRS = _find_input_data_dirs()


@pytest.mark.skipif(_REAL_DIRS is None, reason='real commodity input_data not found relative to repo root')
class TestRealDataIntegration:
    def test_full_pipeline_produces_finite_aligned_returns(self):
        returns_df = data_prep.align_returns(_REAL_DIRS)
        assert len(returns_df) > 0
        assert returns_df.isna().sum().sum() == 0
        assert np.isfinite(returns_df.values).all()

    def test_horizon_params_are_well_formed_on_real_data(self):
        returns_df = data_prep.align_returns(_REAL_DIRS)
        params, meta = data_prep.build_horizon_params(returns_df, horizon=2, lookback=60, cost_scale=0.02)
        assert len(params) == 2
        for p in params:
            # covariance must be symmetric and positive-semidefinite (real
            # sample covariance of real returns).
            np.testing.assert_allclose(p['cov'], p['cov'].T)
            eigvals = np.linalg.eigvalsh(p['cov'])
            assert (eigvals >= -1e-10).all()
