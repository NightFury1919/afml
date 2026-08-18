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
            # *** LOAD-BEARING (2026-08-18): removed the "genuine change
            # from every prior run" claim *** -- this pipeline is single-
            # run/stateless (no persisted history across live runs, see
            # portfolio_oversight/oversight.py's own docstring), so this
            # text can't actually verify novelty/rarity across runs. By
            # 2026-08-18, multiple consecutive live runs had in fact come
            # back stationary, making the old claim actively false, not
            # just unverifiable. Replaced with a comparison this function
            # CAN honestly make: against the project's documented static-
            # dataset finding (Ch13 non-stationary, matching book Sec
            # 13.6.1's random-walk-like prediction).
            lines.append(
                f"Optimal Trading Rule (Ch13): this run's price behavior "
                f"came back STATIONARY (phi_hat={phi:.4f}, half-life="
                f"{otr_result['half_life']:.1f} bars). A fittable "
                f"synthetic-backtested rule was found: profit-take="
                f"{pt:.2f}, stop-loss={sl:.2f} (real price units), Sharpe="
                f"{sharpe:.4f} across {otr_result['n_opportunities']} "
                "real opportunities. This differs from this project's "
                "established static-dataset finding (Ch13's phi_hat "
                "non-stationary, per book Sec 13.6.1's random-walk-like "
                "prediction) -- worth continued tracking across live runs "
                "before treating as a stable pattern."
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
                  min_reliable_T=30, min_reliable_trials=10,
                  otr_result=None, strategy_risk_result=None,
                  pt_sl_result=None):
    """
    Parameters
    ----------
    eval_result : dict, output of stages.evaluate_overfitting
    signal : float or None, output of stages.latest_bet_signal
    asset_label : str, human-readable asset name for the writeup
    min_reliable_T : int, below this UNIQUENESS-WEIGHTED effective T
        (see stages.py's 2026-08-17 LOAD-BEARING note -- T is no longer a
        raw bar count), PBO/DSR are flagged as statistically less
        reliable.

        *** LOAD-BEARING (2026-08-17): simulation-derived, not a guess ***
        Replaces the old min_reliable_T=150, which was picked to "feel
        right" against a ~238-bar dataset ceiling -- a leftover from when
        T meant something else entirely (a raw bar count). 30 is derived
        from calibrate_min_reliable_T.py's null-hypothesis Monte Carlo
        (N=20 trials, matching this project's real C_GRID x STEP_GRID,
        20,000 reps per T, genuinely zero true edge): DSR's own multiple-
        testing correction is ALREADY well-calibrated at any T tested
        (P[DSR>0.5]~=50%, P[DSR>0.95]~=0% under the null, uniformly across
        T=5..200) -- the sqrt(T-1) term in the PSR formula dampens
        confidence exactly as it should at small T, so small T does not
        produce falsely confident DSR readings. What DOES degrade at
        small T is PRECISION: a single DSR draw's std deviation under the
        null starts at 0.204 (T=5) and only flattens to within ~1-2% of
        its T=1000 asymptotic floor (0.157) by around T=75; T=30 sits at
        0.162 (~3.4% above the floor) -- past the steepest part of the
        curve, comfortably closer to the precision floor than to the old
        threshold's implicit assumption. See calibrate_min_reliable_T.py
        for the full curve and methodology.
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
    S = eval_result.get('S', 12)  # 2026-08-18: read dynamically so this
                                    # text can't drift from the S actually
                                    # used -- same lesson as oversight.py's
                                    # min_reliable_T bug found today
    skew = eval_result.get('skew')
    kurtosis = eval_result.get('kurtosis')
    small_sample = T < min_reliable_T or n_trials < min_reliable_trials

    lines = []
    header = f"AFML Pipeline Assessment -- {asset_label}"
    lines.append(header)
    lines.append("=" * len(header))
    lines.append("")
    lines.append(
        f"Across {n_trials} model configuration(s) tested on {T:.2f} "
        f"out-of-sample (purged, embargoed, uniqueness-weighted) "
        f"observations, the best-performing configuration was "
        f"'{best_trial}', with an unannualized Sharpe ratio of "
        f"{sr_hat:.4f}."
    )
    lines.append("")
    lines.append(
        f"Probability of Backtest Overfitting (PBO): {prob_overfit:.2%}. "
        "This estimates the chance that the best in-sample configuration "
        "underperforms the median configuration out-of-sample -- i.e. that "
        "picking the 'winning' model was a matter of noise, not genuine "
        "skill."
    )
    lines.append(
        f"(PBO carries substantial sampling noise at this pipeline's scale "
        f"REGARDLESS of sample size T -- unlike DSR, whose precision "
        f"improves with T, PBO's precision is governed by S (currently "
        f"{S}) and the number of trials. calibrate_pbo_precision.py's "
        f"null-hypothesis Monte Carlo (2026-08-18, this pipeline's real "
        f"T=237/N=20 scale) found a genuinely zero-edge strategy at S={S} "
        f"produces a single PBO draw with a 5th-95th percentile width of "
        f"~0.6-0.7. Treat PBO differences between runs smaller than that "
        f"as inconclusive, not evidence the underlying strategy or "
        f"pipeline constants changed.)"
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
            f"** SAMPLE SIZE WARNING: only {T:.2f} observations and {n_trials} "
            f"trial configuration(s) went into this estimate. DSR's own "
            "precision degrades at small T (calibrate_min_reliable_T.py, "
            "2026-08-17) -- below this pipeline's min_reliable_T threshold, "
            "DSR should NOT be read as reliable evidence of edge (or lack "
            "of it) until re-checked with a substantially larger out-of-"
            "sample count. (PBO's own precision caveat above applies "
            "separately and REGARDLESS of this warning -- PBO's noise "
            "does not shrink with T the way DSR's does.) **"
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
