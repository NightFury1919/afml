"""
portfolio_oversight/oversight.py

*** THIS MODULE IS NOT AFML CONTENT. ***

Everything in this file is genuinely NEW material this project invented,
living in its own top-level directory (sibling to ch01-ch22 and pipeline/)
specifically so it can never be confused with the book-fidelity codebase.
No book snippet was ever supplied for any of this, because none exists to
supply -- AFML Ch1 Sec 1.3.1.6 NAMES a "Portfolio Oversight" production
station, with an embargo -> paper trading -> graduation -> re-allocation
-> decommission lifecycle, but deliberately never gives it an algorithm
(the book's own stated reasoning: "it would be unreasonable for a book to
reveal specific investment strategies"). Everything below is this
project's own reasonable-but-arbitrary attempt to fill that named-but-
unspecified gap, for teaching purposes only. See the 2026-08-14 handoff,
Part 5, for the full decision to keep this clearly separated from the
real AFML chapters.

*** NO REAL CAPITAL, NO REAL EXECUTION ***
This entire project only ever READS market data (the Binance.US API key
is read-only -- see pipeline/orchestration/ingestion.py). There is no
order-execution code anywhere in this repo. Every function below is
PURELY INFORMATIONAL: a suggested dollar figure, a suggested lifecycle
label, a suggested "you might want to pause" flag -- never an action
taken, never a real halt, never a real trade. This matches
pipeline/orchestration/report.py's own established scope discipline
(states evidence, never issues a buy/sell directive) -- deliberately
extended here, not relaxed, even though this content is new.

*** SINGLE-RUN ONLY, NO PERSISTED STATE (decided 2026-08-15) ***
A "real" circuit breaker (running drawdown across trades) or a "real"
lifecycle tracker (how many runs has this strategy survived, minimum
time-in-stage before promotion) would need to remember something across
multiple live runs -- new persisted-state infrastructure this pipeline
doesn't have. Deliberately NOT built today: every function here
classifies/suggests from THIS RUN's own already-computed numbers alone
(eval_result, strategy_risk_result -- reused, not recomputed) and forgets
everything the moment the run ends. If real persisted state is wanted
later, that's a separate, larger addition, not an extension of this file.

*** Thresholds are invented, not derived ***
classify_lifecycle_stage's DSR bands (0.5, 0.95) deliberately reuse
pipeline/orchestration/report.py's OWN _confidence_band thresholds (low
<0.5, moderate 0.5-0.95, high >=0.95) for internal consistency across the
report, rather than inventing a second, different threshold system. This
is a design choice for coherence, not a derivation from anything in AFML.
check_circuit_breaker's thresholds (PBO>0.5, P[fail]>0.5) are similarly
simple, round, and arbitrary -- adjust freely; they are not book-derived
either.
"""

DEFAULT_MAX_POSITION_FRACTION = 0.10   # arbitrary, invented judgment call:
                                         # the max fraction of paper capital
                                         # this add-on will ever suggest
                                         # risking on a single highest-
                                         # confidence (signal=+/-1) bet


def suggest_capital_allocation(signal, paper_capital_usd,
                                 max_position_fraction=DEFAULT_MAX_POSITION_FRACTION):
    """Translates Ch10's abstract [-1,+1] getSignal confidence scale into
    a suggested DOLLAR amount, given a paper trading capital figure. Pure
    arithmetic, no new statistics: suggested_usd = signal * capital *
    max_position_fraction.

    Parameters
    ----------
    signal : float in [-1, 1], or None (no current signal available --
        see stages.latest_bet_signal, which can return None)
    paper_capital_usd : float > 0, this run's configured paper-trading
        capital figure (e.g. run_pipeline_live.py's PAPER_CAPITAL_USD)
    max_position_fraction : float in (0, 1], the max fraction of capital
        suggested at full confidence (signal = +/-1)

    Returns
    -------
    dict with keys:
      'has_signal'         : bool
      'suggested_usd'       : float or None (signed: + long, - short),
        None if has_signal is False
      'direction'           : 'long', 'short', 'flat', or 'no signal'
      'paper_capital_usd'   : float, echoed back
      'max_position_fraction' : float, echoed back
    """
    if not (0 < max_position_fraction <= 1):
        raise ValueError('max_position_fraction must be in (0, 1]')
    if paper_capital_usd <= 0:
        raise ValueError('paper_capital_usd must be positive')

    if signal is None:
        return {
            'has_signal': False, 'suggested_usd': None, 'direction': 'no signal',
            'paper_capital_usd': paper_capital_usd,
            'max_position_fraction': max_position_fraction,
        }

    suggested_usd = signal * paper_capital_usd * max_position_fraction
    direction = 'long' if signal > 0 else ('short' if signal < 0 else 'flat')

    return {
        'has_signal': True, 'suggested_usd': suggested_usd, 'direction': direction,
        'paper_capital_usd': paper_capital_usd,
        'max_position_fraction': max_position_fraction,
    }


def check_circuit_breaker(eval_result, strategy_risk_result=None,
                            pbo_threshold=0.5, p_fail_threshold=0.5):
    """Informational-only 'would a cautious operator pause here?' flag,
    from THIS RUN's own PBO (eval_result) and, if available, THIS RUN's
    own Ch15 P[fail] (strategy_risk_result). Never halts anything for
    real -- see module docstring.

    Parameters
    ----------
    eval_result : dict, stages.evaluate_overfitting() output (needs
        'prob_overfit')
    strategy_risk_result : dict or None, risk_context.compute_strategy_risk()
        output (needs 'p_fail'). None skips that check (e.g. if it
        couldn't be computed this run) rather than crashing.
    pbo_threshold, p_fail_threshold : float, arbitrary round-number
        defaults (see module docstring) -- adjust freely

    Returns
    -------
    dict with keys:
      'triggered' : bool, True if ANY check fired
      'reasons'   : list of str, one entry per fired check (empty if none)
      'pbo'       : float, echoed from eval_result
      'p_fail'    : float or None, echoed from strategy_risk_result
    """
    pbo = eval_result['prob_overfit']
    p_fail = strategy_risk_result['p_fail'] if strategy_risk_result is not None else None

    reasons = []
    if pbo > pbo_threshold:
        reasons.append(
            f'PBO {pbo:.2%} exceeds the {pbo_threshold:.2%} threshold -- '
            'the winning configuration is more likely than not to be noise.'
        )
    if p_fail is not None and p_fail > p_fail_threshold:
        reasons.append(
            f'Ch15 P[fail] {p_fail:.2%} exceeds the {p_fail_threshold:.2%} '
            'threshold -- realized precision is unlikely to sustain this '
            "run's own claimed Sharpe."
        )

    return {'triggered': len(reasons) > 0, 'reasons': reasons, 'pbo': pbo, 'p_fail': p_fail}


# *** LOAD-BEARING (2026-08-18): min_reliable_T corrected 150 -> 30 ***
# This default had drifted out of sync with report.py's build_report(),
# whose min_reliable_T was changed 150->30 on 2026-08-17 (see stages.py's
# DSR uniqueness-weighting fix and calibrate_min_reliable_T.py). This
# function's own docstring already claimed the two were "kept in sync
# deliberately" -- they were not, until this fix. build_oversight_section()
# calls this with no explicit min_reliable_T, so the DEFAULT is what
# actually governs EMBARGO/PAPER_TRADING/GRADUATION_CANDIDATE
# classification -- the stale default silently overrode the corrected
# report.py threshold for every live run since 2026-08-17. No shared
# constant exists to import (report.py's 30 is a bare function default,
# not a module-level constant) -- if report.py's threshold changes again,
# this default must be updated manually, same as it should have been then.
def classify_lifecycle_stage(eval_result, min_reliable_T=30, min_reliable_trials=10,
                               paper_trading_dsr=0.5, graduation_dsr=0.95):
    """Single-run, heuristic, NEW (not book-derived) classification into
    one of AFML Ch1 Sec 1.3.1.6's named-but-unalgorithmized lifecycle
    stages, using only THIS RUN's own already-computed eval_result --
    see module docstring for why the DSR bands deliberately match
    report.py's own _confidence_band thresholds.

    Rules (checked in order; first match wins):
      1. Sample too small to trust (T < min_reliable_T or n_trials <
         min_reliable_trials) -> EMBARGO, regardless of DSR -- an
         impressive DSR on a tiny sample isn't evidence of anything yet.
      2. DSR < paper_trading_dsr -> EMBARGO -- no meaningful evidence of
         edge; matches report.py's 'low' confidence band.
      3. paper_trading_dsr <= DSR < graduation_dsr -> PAPER_TRADING --
         promising but unconfirmed; matches report.py's 'moderate' band.
      4. DSR >= graduation_dsr -> GRADUATION_CANDIDATE -- matches
         report.py's 'high' band.

    Parameters
    ----------
    eval_result : dict, stages.evaluate_overfitting() output (needs 'T',
        'n_trials', 'dsr')
    min_reliable_T, min_reliable_trials : same meaning/defaults as
        report.py's build_report() small-sample check -- kept in sync
        deliberately, not re-derived
    paper_trading_dsr, graduation_dsr : float, DSR breakpoints (see
        module docstring)

    Returns
    -------
    dict with keys:
      'stage'  : 'EMBARGO', 'PAPER_TRADING', or 'GRADUATION_CANDIDATE'
      'reason' : str, plain-English reason for the classification
      'dsr'    : float, echoed from eval_result
      'small_sample' : bool
    """
    T = eval_result['T']
    n_trials = eval_result['n_trials']
    dsr = eval_result['dsr']
    small_sample = T < min_reliable_T or n_trials < min_reliable_trials

    if small_sample:
        return {
            'stage': 'EMBARGO',
            'reason': (
                f'Only {T} out-of-sample observations / {n_trials} trial '
                'configuration(s) -- too small a sample to trust ANY DSR '
                'value yet, regardless of what it currently reads.'
            ),
            'dsr': dsr, 'small_sample': True,
        }
    if dsr < paper_trading_dsr:
        return {
            'stage': 'EMBARGO',
            'reason': f'DSR {dsr:.4f} is below the paper-trading bar of {paper_trading_dsr} -- no meaningful evidence of edge yet.',
            'dsr': dsr, 'small_sample': False,
        }
    if dsr < graduation_dsr:
        return {
            'stage': 'PAPER_TRADING',
            'reason': f'DSR {dsr:.4f} is promising (>= {paper_trading_dsr}) but below the graduation bar of {graduation_dsr} -- would need real-time monitoring before real capital.',
            'dsr': dsr, 'small_sample': False,
        }
    return {
        'stage': 'GRADUATION_CANDIDATE',
        'reason': f'DSR {dsr:.4f} clears the graduation bar of {graduation_dsr}.',
        'dsr': dsr, 'small_sample': False,
    }


def build_oversight_section(signal, eval_result, strategy_risk_result=None,
                              paper_capital_usd=10_000.0,
                              max_position_fraction=DEFAULT_MAX_POSITION_FRACTION):
    """Plain-English presentation of the three functions above, banner-
    labeled as explicitly non-AFML, experimental content -- meant to be
    concatenated onto pipeline/orchestration/report.py's build_report()
    output as a clearly separate section (NOT merged into report.py
    itself -- see module docstring). Never issues a buy/sell directive,
    same scope discipline as report.py.

    Parameters
    ----------
    signal : float or None, stages.latest_bet_signal() output
    eval_result : dict, stages.evaluate_overfitting() output
    strategy_risk_result : dict or None, risk_context.compute_strategy_risk()
        output
    paper_capital_usd : float, this run's configured paper capital
    max_position_fraction : float, see suggest_capital_allocation

    Returns
    -------
    str, plain-English text (no leading/trailing blank lines).
    """
    alloc = suggest_capital_allocation(signal, paper_capital_usd, max_position_fraction)
    breaker = check_circuit_breaker(eval_result, strategy_risk_result)
    lifecycle = classify_lifecycle_stage(eval_result)

    lines = []
    lines.append('=' * 74)
    lines.append('EXPERIMENTAL -- NOT FROM AFML (portfolio_oversight/, added 2026-08-15)')
    lines.append('=' * 74)
    lines.append(
        "AFML Ch1 names a 'Portfolio Oversight' station (embargo -> paper "
        'trading -> graduation -> re-allocation -> decommission) but never '
        "gives it an algorithm. Everything in this section is this "
        'project\'s OWN invented heuristic filling that gap -- informational '
        'only, no real capital, no real order execution anywhere in this '
        'repo. Treat this section with LESS confidence than everything '
        'above it, which is real, book-fidelity AFML content.'
    )
    lines.append('')

    if alloc['has_signal']:
        lines.append(
            f"Suggested paper-capital allocation: ${alloc['suggested_usd']:,.2f} "
            f"({alloc['direction']}) on a ${alloc['paper_capital_usd']:,.2f} paper "
            f"capital base, capped at {alloc['max_position_fraction']:.0%} of "
            "capital at full signal confidence. This is a suggested FIGURE, "
            "NOT investment advice or a directive to allocate capital."
        )
    else:
        lines.append('Suggested paper-capital allocation: none (no current signal).')
    lines.append('')

    lines.append(f"Lifecycle stage (this run only, no persisted history): {lifecycle['stage']}.")
    lines.append(f"  {lifecycle['reason']}")
    lines.append('')

    if breaker['triggered']:
        lines.append('Circuit breaker: FLAGGED (informational only -- nothing is actually halted).')
        for reason in breaker['reasons']:
            lines.append(f'  - {reason}')
    else:
        lines.append('Circuit breaker: not flagged on this run.')

    return '\n'.join(lines)
