"""
AFML Chapter 21 -- real-data preparation.

NOT part of the book's printed snippets. The book hands you `params`
(mu_h, V_h, c_h per horizon) as a given; it never says how to derive them
from real market data. This module is the missing plumbing: it turns raw
per-contract commodity files (gold, crude oil, US T-bonds -- from
`input_data/`) into the multi-horizon parameter list that
brute_force.dyn_opt_port() and static_solution.stat_opt_trajectory() expect.

Design decisions made here (confirmed with Ethan, documented per project
convention rather than left implicit):

1. FRONT-MONTH SELECTION: the raw files are one file per individual futures
   contract (e.g. GC02Z.txt = the gold contract expiring Dec 2002), and
   several contracts' date ranges overlap (a contract can trade for years
   before its expiry). roll.py's roll-adjustment logic expects a SINGLE
   continuous series -- one row per date, tagged with whichever contract
   was "front month" that day. This module selects the front-month contract
   per date using the highest trading VOLUME that day (a standard, real
   roll-selection rule, since no fixed expiry calendar is available in the
   raw files).
2. RAW FILE FORMAT: two incompatible formats coexist in the same commodity
   folder (not split cleanly by era) -- some files have no header and a
   6-digit YYMMDD date (e.g. `770822,101.78125,...`), others have a quoted
   header and MM/DD/YYYY dates (e.g. `"Date","Open",...` then
   `11/27/2001,99.1875,...`). Both are parsed and normalized here.
3. mu_h / V_h / c_h FROM REAL RETURNS: the book doesn't specify this. Per
   design discussion, horizons are built from NON-OVERLAPPING trailing
   windows of real daily returns near the end of the aligned history (most
   recent windows first, walking backward), so mu_h/V_h are honest rolling
   sample statistics, not fabricated. c_h (per-asset transaction cost
   factor) is proxied as cost_scale * (that window's realized daily
   volatility per asset) -- costs scale with how volatile/illiquid an asset
   was over that window, a standard real-world proxy.
"""

import os
import sys

import numpy as np
import pandas as pd


def _repo_root():
    """Derive the repo root from this file's location: ch21/portfolio_trajectory/ -> repo root."""
    return os.path.abspath(os.path.join(os.path.dirname(os.path.abspath(__file__)), '..', '..'))


def _import_roll_module():
    """
    Import ch02/multi_product/roll.py using the project's robust,
    __file__-derived import pattern (Ch10/Ch13 style) -- works whether
    pytest is run from the repo root or from inside this module's folder,
    without needing ch02 to be pip-installed or on a hardcoded path.
    """
    roll_dir = os.path.join(_repo_root(), 'ch02', 'multi_product')
    if roll_dir not in sys.path:
        sys.path.insert(0, roll_dir)
    import roll  # noqa: E402
    return roll


# ---------------------------------------------------------------------------
# Step 1: robust raw contract file parsing
# ---------------------------------------------------------------------------
_HEADERLESS_COLS = ['Date', 'Open', 'High', 'Low', 'Close', 'Volume', 'OpenInt']


def _parse_headerless_date(raw):
    """
    Parse a 6-digit YYMMDD date (e.g. 770822 -> 1977-08-22, 991103 -> 1999-11-03,
    000124 -> 2000-01-24). Century-cutoff rule: YY <= 30 -> 20YY, else 19YY.
    Safe for this dataset's observed range (1959-2002): the only 20xx years
    present are 00, 01, 02, all comfortably under the cutoff.
    """
    raw = str(int(raw)).zfill(6)
    yy, mm, dd = int(raw[0:2]), int(raw[2:4]), int(raw[4:6])
    year = 2000 + yy if yy <= 30 else 1900 + yy
    return pd.Timestamp(year=year, month=mm, day=dd)


def load_contract_file(path):
    """
    Parse a single raw contract file into a standardized DataFrame, handling
    BOTH formats found in this dataset (see module docstring, point 2).

    Returns
    -------
    pd.DataFrame
        Columns: Instrument, Open, High, Low, Close, Volume, OpenInt.
        Indexed by Date (pd.Timestamp), sorted ascending.
    """
    instrument = os.path.splitext(os.path.basename(path))[0]
    with open(path, 'r') as f:
        first_line = f.readline()

    if first_line.lstrip().startswith('"'):
        # Headered format: quoted columns, MM/DD/YYYY dates.
        df = pd.read_csv(path)
        df['Date'] = pd.to_datetime(df['Date'], format='%m/%d/%Y')
    else:
        # Headerless format: 6-digit YYMMDD dates, fixed column order.
        df = pd.read_csv(path, header=None, names=_HEADERLESS_COLS)
        df['Date'] = df['Date'].apply(_parse_headerless_date)

    df['Instrument'] = instrument
    df = df.set_index('Date').sort_index()
    return df[['Instrument', 'Open', 'High', 'Low', 'Close', 'Volume', 'OpenInt']]


def load_commodity_panel(commodity_dir):
    """
    Load and concatenate EVERY contract file in a commodity's raw data
    folder into one long panel (multiple rows per date -- one per contract
    trading that day).

    Parameters
    ----------
    commodity_dir : str
        Path to a folder of raw per-contract .txt files (e.g. input_data/gold).

    Returns
    -------
    pd.DataFrame
        Columns: Instrument, Open, High, Low, Close, Volume, OpenInt.
        Indexed by Date. NOT yet reduced to one row per date.
    """
    files = sorted(
        f for f in os.listdir(commodity_dir)
        if f.lower().endswith('.txt')
    )
    if not files:
        raise FileNotFoundError(f'No .txt contract files found in {commodity_dir}')
    frames = [load_contract_file(os.path.join(commodity_dir, f)) for f in files]
    panel = pd.concat(frames, axis=0)
    return panel.sort_index()


def select_front_month(panel):
    """
    Step: for each date, select the single row with the HIGHEST trading
    volume as the front-month contract for that day (design decision,
    see module docstring point 1). Ties broken arbitrarily (pandas'
    stable sort keeps the first-encountered row, i.e. the alphabetically
    first Instrument, which is an acceptable tie-break for same-volume rows).

    Parameters
    ----------
    panel : pd.DataFrame
        Output of load_commodity_panel (may have multiple rows per date).

    Returns
    -------
    pd.DataFrame
        One row per date, columns: Instrument, Open, Close (subset needed
        by roll.py), sorted ascending by date.
    """
    panel_sorted = panel.sort_values('Volume', ascending=False)
    front = panel_sorted.groupby(panel_sorted.index).first()
    front = front.sort_index()
    return front[['Instrument', 'Open', 'Close']]


def build_continuous_series(commodity_dir, match_end=True):
    """
    Full pipeline for one commodity: raw contract files -> front-month
    selection -> roll-gap-adjusted continuous series with real daily returns
    (via roll.py's non_negative_rolled_prices, which guards against the
    negative-price artifacts that plain roll adjustment can produce in
    contango markets -- see roll.py's own docstring).

    Parameters
    ----------
    commodity_dir : str
        Path to input_data/<commodity> folder.
    match_end : bool, default True
        Passed through to roll.py (True = backward/most-common convention:
        latest prices are left unadjusted).

    Returns
    -------
    pd.DataFrame
        Indexed by Date; includes 'Returns' (daily % return, roll-adjusted)
        and 'rPrices' ($1-compounded continuous series) columns.
    """
    roll = _import_roll_module()
    panel = load_commodity_panel(commodity_dir)
    front = select_front_month(panel)
    rolled = roll.non_negative_rolled_prices(front)
    return rolled


def align_returns(commodity_dirs, match_end=True):
    """
    Build continuous series for several commodities and inner-join their
    daily returns onto a shared date index (only dates all assets have data
    for are kept -- required for a well-defined covariance matrix).

    Parameters
    ----------
    commodity_dirs : dict[str, str]
        Maps a short asset name (e.g. 'gold') to its input_data folder path.
    match_end : bool, default True
        Passed through to build_continuous_series.

    Returns
    -------
    pd.DataFrame
        Columns = asset names (in commodity_dirs' order), index = Date,
        values = daily returns. No NaNs (inner join + drop the leading NaN
        from each series' own first-day return).
    """
    returns = {}
    for name, path in commodity_dirs.items():
        series = build_continuous_series(path, match_end=match_end)
        returns[name] = series['Returns'].dropna()
    returns_df = pd.concat(returns, axis=1, join='inner')
    returns_df.columns = list(commodity_dirs.keys())
    return returns_df.dropna()


def build_horizon_params(returns_df, horizon, lookback, cost_scale=0.02):
    """
    Turn a multi-asset daily-returns DataFrame into the H-horizon params
    list dyn_opt_port/stat_opt_trajectory expect, using NON-OVERLAPPING
    trailing windows near the end of the available history (most recent
    window is the LAST horizon, walking backward for earlier horizons --
    see module docstring point 3).

    Parameters
    ----------
    returns_df : pd.DataFrame
        Output of align_returns: columns = assets, index = Date.
    horizon : int
        Number of horizons H. Needs horizon * lookback <= len(returns_df).
    lookback : int
        Trading days per window used to estimate mu_h, V_h, c_h.
    cost_scale : float, default 0.02
        Scales realized per-window volatility into a transaction-cost
        factor c_h (proxy: costlier to trade a more volatile/illiquid asset).

    Returns
    -------
    list[dict]
        Length-H list of {'mean': (N,1), 'cov': (N,N), 'c': (N,)} dicts,
        ordered horizon 1 .. H (oldest window first, most recent window last).
    dict
        Metadata: {'window_dates': [(start, end), ...], 'assets': [...]}
        for the driver script/notebook to report real, verifiable numbers.
    """
    n_needed = horizon * lookback
    if n_needed > len(returns_df):
        raise ValueError(
            f'Need {n_needed} trading days (horizon={horizon} x lookback={lookback}) '
            f'but only {len(returns_df)} are available after alignment.'
        )
    tail = returns_df.iloc[-n_needed:]
    params = []
    window_dates = []
    for h in range(horizon):
        window = tail.iloc[h * lookback:(h + 1) * lookback]
        mean_ = window.mean().values.reshape(-1, 1)
        cov_ = window.cov().values
        c_ = cost_scale * window.std().values
        params.append({'mean': mean_, 'cov': cov_, 'c': c_})
        window_dates.append((window.index[0], window.index[-1]))
    metadata = {'window_dates': window_dates, 'assets': list(returns_df.columns)}
    return params, metadata
