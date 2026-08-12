"""
pipeline/orchestration/report.py

Turns the orchestration layer's raw statistics (trial Sharpes, PBO
probability, deflated Sharpe ratio, latest bet signal) into a plain-English
"edge / risk / confidence" writeup for a non-technical trading-club reader.

No new AFML formula lives here -- this is presentation logic over
stages.py's real, already-computed numbers. Deliberately does NOT output a
buy/sell directive: per this project's scope, the report states the
evidence (edge, risk, confidence) and lets the reader decide, consistent
with this codebase's own honestly-reported null results on this same
dataset (Ch11-15: PBO ~0.83, CPCV paths negative, DSR 0/5 survive).
"""


def _confidence_band(dsr):
    if dsr is None or dsr != dsr:  # NaN check without a numpy import here
        return "undetermined (insufficient data for a deflated Sharpe estimate)"
    if dsr >= 0.95:
        return "high"
    if dsr >= 0.5:
        return "moderate"
    return "low"


def build_report(eval_result, signal, asset_label='this asset',
                  min_reliable_T=250, min_reliable_trials=5):
    """
    Parameters
    ----------
    eval_result : dict, output of stages.evaluate_overfitting
    signal : float or None, output of stages.latest_bet_signal
    asset_label : str, human-readable asset name for the writeup
    min_reliable_T : int, below this sample size, PBO/DSR are flagged as
        statistically unreliable (a single draw can range ~0.04-0.99 for a
        genuinely zero-edge strategy on small samples -- see this project's
        own ch11/backtest_dangers/pbo.py TDD notes on estimator imprecision)
    min_reliable_trials : int, below this trial count, DSR's multiple-
        testing correction has too little information to be meaningful

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
