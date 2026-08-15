"""
pipeline/orchestration/report.py

Turns the orchestration layer's raw statistics (trial Sharpes, PBO
probability, deflated Sharpe ratio, latest bet signal) into a plain-English
"edge / risk / confidence" writeup for a non-technical trading-club reader.

No new AFML formula lives here -- this is presentation logic over
stages.py's real, already-computed numbers (and, as of Phase 4, over
risk_context.py's real, already-computed OTR/strategy-risk/PT-SL numbers
-- see risk_context.py's own module docstring for why those live in a
separate module rather than here). Deliberately does NOT output a
buy/sell directive: per this project's scope, the report states the
evidence (edge, risk, confidence) and lets the reader decide.

PHASE 1b (2026-08-12): stages.py now reuses Ch11's real, established
20-trial bar-level construction (not Phase 1a's 3-trial event-level proxy),
reconciling PBO to ~0.83 and dropping DSR from a false 0.9995 to ~0.54 --
consistent with this codebase's own honestly-reported near-null results on
this same dataset (Ch11-15).

PHASE 4 (2026-08-15): added an optional risk-context section (Ch13 OTR,
Ch15 strategy risk, rebuild.py's PT/SL) -- see the 2026-08-14 handoff,
Part 5. All three new parameters default to None so run_pipeline.py's
existing static-data call (which doesn't pass them) is unaffected.
"""


def _confidence_band(dsr):
    if dsr is None or dsr != dsr:  # NaN check without a numpy import here
        return "undetermined (insufficient data for a deflated Sharpe estimate)"
    if dsr >= 0.95:
        return "high"
    if dsr >= 0.5:
        return "moderate"
    return "low"


def _risk_context_section(otr_result, strategy_risk_result, pt_sl_result):
    """Builds the plain-English risk-context section (Ch13 OTR, Ch15
    strategy risk, rebuild.py's PT/SL) -- pure presentation over
    risk_context.py's already-computed dicts, no new computation. Any of
    the three inputs may be None (that sub-section is simply omitted),
    so a partial risk-context call (e.g. PT/SL surfaced but OTR/Ch15
    skipped for some reason) still produces a coherent section."""
    lines = []
    lines.append("Risk Context (Portfolio-Oversight-adjacent, from real AFML content)")
    lines.append("-" * 68)
    lines.append("")

    if pt_sl_result is not None:
        pt_pct = pt_sl_result['implied_pt_pct']
        sl_pct = pt_sl_result['implied_sl_pct']
        lines.append(
            f"Stop-loss / take-profit (Ch3 triple-barrier, PT_SL="
            f"{pt_sl_result['pt_sl']}): this run's most recent triple-"
            f"barrier event implies a profit-take at +{pt_pct:.2%} and a "
            f"stop-loss at -{sl_pct:.2%} from entry, scaled to current "
            "volatility (this is the SAME barrier width already driving "
            "every triple-barrier label in this pipeline -- not a new "
            "calculation, just surfaced explicitly here for the reader)."
        )
        lines.append("")

    if otr_result is not None:
        phi = otr_result['phi_hat']
        if otr_result['stationary']:
            pt, sl, mean, std, sharpe = otr_result['best_node']
            lines.append(
                f"Optimal Trading Rule (Ch13): this run's price behavior "
                f"came back STATIONARY (phi_hat={phi:.4f}, half-life="
                f"{otr_result['half_life']:.1f} bars) -- a genuine change "
                "from every prior run on this pipeline. A fittable "
                f"synthetic-backtested rule was found: profit-take="
                f"{pt:.2f}, stop-loss={sl:.2f} (real price units), Sharpe="
                f"{sharpe:.4f} across {otr_result['n_opportunities']} "
                "real opportunities. Worth investigating further given "
                "this is not the established finding."
            )
        else:
            lines.append(
                f"Optimal Trading Rule (Ch13): phi_hat={phi:.4f} -- "
                "NON-STATIONARY (outside the O-U requirement of (-1,1)), "
                f"across {otr_result['n_opportunities']} real "
                "opportunities. Consistent with every prior run on this "
                "pipeline: this asset's bar-level price behavior over "
                "typical trade-holding windows is close to a random walk, "
                "so no fittable optimal profit-take/stop-loss rule exists "
                "here (the book's own degenerate case, Sec 13.6.1 -- not "
                "a gap in this pipeline)."
            )
        lines.append("")

    if strategy_risk_result is not None:
        p_fail = strategy_risk_result['p_fail']
        p_bar = strategy_risk_result['p_bar']
        freq = strategy_risk_result['freq_real']
        verdict = "TOO RISKY per the book's own >.05 rule of thumb" if p_fail > 0.05 else "within the book's own <=.05 rule of thumb"
        lines.append(
            f"Probability of strategy failure (Ch15): at this run's own "
            f"best-trial Sharpe as the target, P[true precision too low "
            f"to sustain it] = {p_fail:.4f} -- {verdict}. Realized "
            f"precision on this run's {strategy_risk_result['n_events']} "
            f"real bets was {p_bar:.2%}, annualized to ~{freq:.0f} "
            f"bets/year over {strategy_risk_result['elapsed_years']:.3f} "
            "years of real data (short-window annualization -- treat as "
            "a genuine but heavily-extrapolated estimate, not a full-year "
            "track record)."
        )
        lines.append("")

    return lines


def build_report(eval_result, signal, asset_label='this asset',
                  min_reliable_T=150, min_reliable_trials=10,
                  otr_result=None, strategy_risk_result=None,
                  pt_sl_result=None):
    """
    Parameters
    ----------
    eval_result : dict, output of stages.evaluate_overfitting
    signal : float or None, output of stages.latest_bet_signal
    asset_label : str, human-readable asset name for the writeup
    min_reliable_T : int, below this nonzero-bet count, PBO/DSR are flagged
        as statistically less reliable. Lowered from Phase 1a's 250 to 150
        now that T is a real nonzero-bet count bounded by this dataset's
        ~238-bar ceiling (Phase 1a's 250 threshold could never be met by
        this dataset at all, which defeated the point of the check) -- see
        this project's own ch11/backtest_dangers/pbo.py TDD notes on
        estimator imprecision for why any threshold here is a heuristic,
        not a statistically derived cutoff
    min_reliable_trials : int, below this trial count, DSR's multiple-
        testing correction has too little information to be meaningful.
        Phase 1b's real trial count is 20 (vs Phase 1a's 3)
    otr_result : dict or None, risk_context.compute_otr_finding() output.
        None (default) omits the OTR sub-section entirely.
    strategy_risk_result : dict or None, risk_context.compute_strategy_risk()
        output. None (default) omits that sub-section entirely.
    pt_sl_result : dict or None, risk_context.compute_pt_sl_context()
        output. None (default) omits that sub-section entirely.

    Returns
    -------
    str, a plain-English report.
    """
    sr_hat = eval_result['sr_hat']
    prob_overfit = eval_result['prob_overfit']
    dsr = eval_result['dsr']
    n_trials = eval_result['n_trials']
    best_trial = eval_result['best_trial']
    T = eval_result['T']
    skew = eval_result.get('skew')
    kurtosis = eval_result.get('kurtosis')
    small_sample = T < min_reliable_T or n_trials < min_reliable_trials

    lines = []
    header = f"AFML Pipeline Assessment -- {asset_label}"
    lines.append(header)
    lines.append("=" * len(header))
    lines.append("")
    lines.append(
        f"Across {n_trials} model configuration(s) tested on {T} "
        f"out-of-sample (purged, embargoed) observations, the "
        f"best-performing configuration was '{best_trial}', with an "
        f"unannualized Sharpe ratio of {sr_hat:.4f}."
    )
    lines.append("")
    lines.append(
        f"Probability of Backtest Overfitting (PBO): {prob_overfit:.2%}. "
        "This estimates the chance that the best in-sample configuration "
        "underperforms the median configuration out-of-sample -- i.e. that "
        "picking the 'winning' model was a matter of noise, not genuine "
        "skill."
    )
    lines.append("")
    lines.append(
        f"Deflated Sharpe Ratio (DSR): {dsr:.4f}. This corrects the best "
        f"trial's Sharpe ratio for the fact that {n_trials} configurations "
        "were tried and the best one was cherry-picked. DSR estimates the "
        "probability that the TRUE Sharpe ratio exceeds zero once that "
        "selection bias is accounted for. Values near 1.0 indicate strong "
        "evidence of genuine, non-overfit edge; values near 0.5 or below "
        "indicate the apparent outperformance is statistically "
        "indistinguishable from noise."
    )
    if skew is not None and kurtosis is not None:
        lines.append(
            f"(Computed using the winning trial's real realized-return "
            f"skew={skew:.4f} and kurtosis={kurtosis:.4f}, not a Gaussian "
            "assumption -- fatter tails or negative skew in real returns "
            "would otherwise cause DSR to overstate confidence.)"
        )
    lines.append("")

    if small_sample:
        confidence = "UNRELIABLE (sample too small to trust)"
        lines.append(
            f"** SAMPLE SIZE WARNING: only {T} observations and {n_trials} "
            f"trial configuration(s) went into this estimate. This "
            "project's own PBO test suite documents that a SINGLE PBO/DSR "
            "estimate on a small sample can range roughly 4%-99% even for "
            "a genuinely zero-edge strategy, purely from sampling noise. "
            "The specific PBO/DSR values above should NOT be read as "
            "reliable evidence of edge (or lack of it) until this is "
            "re-checked with a substantially larger out-of-sample count "
            "and more trial configurations. **"
        )
        lines.append("")
    else:
        confidence = _confidence_band(dsr)

    lines.append(f"Overall confidence in a real, exploitable edge: {confidence}.")
    lines.append("")

    if signal is None:
        lines.append(
            "No current bet-sizing signal is available (no recent "
            "out-of-sample prediction to size)."
        )
    else:
        direction = "long" if signal > 0 else ("short" if signal < 0 else "flat")
        lines.append(
            f"Most recent model-implied position size: {signal:+.2f} "
            f"({direction}) on a [-1, 1] scale, discretized per Ch10's "
            "getSignal. This is a statistically-derived position size, NOT "
            "investment advice or a directive to trade -- it reflects this "
            "specific model's confidence given real historical purged-CV "
            "performance, nothing more."
        )
    lines.append("")

    if otr_result is not None or strategy_risk_result is not None or pt_sl_result is not None:
        lines.extend(_risk_context_section(otr_result, strategy_risk_result, pt_sl_result))

    lines.append(
        "IMPORTANT: This report reflects rigorous statistical testing, not "
        "a guarantee of future performance. Prior real-data runs on this "
        "same BTC/TUSD dataset found convergent evidence of NO exploitable "
        "signal across five independent methods (Ch11-15 of this project) "
        "-- a genuine, honestly-reported null result. A low PBO/DSR score "
        "here is not a bug; it may be the correct, honest answer, and "
        "should be treated as such rather than re-run until a better "
        "number appears."
    )
    return "\n".join(lines)
