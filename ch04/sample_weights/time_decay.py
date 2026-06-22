import numpy as np
import pandas as pd

# Implementation of Time-Decay Factors — AFML Chapter 4, Section 4.7,
# Snippet 4.11, page 70
#
# A THIRD weighting concept, complementary to both average uniqueness
# (Section 4.5) and return attribution (Section 4.6). This one applies a
# piecewise-linear decay across TIME — even a perfectly unique, high-
# conviction observation becomes less relevant to train on as market
# dynamics evolve away from when it occurred.
#
# --- The core idea ---
# Markets change. A labeled observation from five years ago may reflect a
# market regime that no longer exists. We want the model to pay LESS
# attention to old observations and MORE attention to recent ones — but we
# don't want to just chop off old data entirely (that wastes information);
# instead we apply a smooth, linear decay.
#
# --- Why cumulative uniqueness as the x-axis, not calendar time? ---
# If we decayed by raw calendar date, the result would be distorted by gaps
# and clusters in your CUSUM event dates — a quiet month with no events vs. a
# busy week with twenty would decay very differently for reasons that have
# nothing to do with genuine information content. By using the CUMULATIVE
# SUM of average uniqueness (tW) as the x-axis instead, the decay is anchored
# to how much truly new, non-redundant information has accumulated over time
# — consistent with the rest of this chapter's emphasis on uniqueness over
# raw observation count.
#
# --- The clf_last_w parameter controls the SHAPE of the decay ---
# clf_last_w = 1    : no decay at all — every observation weighted equally
# clf_last_w = 0.5  : oldest observation retains half weight, newest gets full
# clf_last_w = 0     : oldest observation gets weight 0 (decays linearly to
#                       zero, but every observation still contributes SOME
#                       non-negative weight)
# clf_last_w < 0    : decay line crosses below zero before reaching the
#                       oldest observation — anything past that crossing
#                       point gets HARD EXCLUDED (weight clipped to exactly 0,
#                       not just down-weighted). More negative values exclude
#                       more of the old data outright.


def get_time_decay(tw, clf_last_w=1.0):
    # Apply piecewise-linear decay to observed uniqueness — AFML Snippet 4.11
    #
    # Newest observation always gets weight 1. Oldest observation gets
    # weight clf_last_w (which may be negative, in which case some of the
    # oldest observations get clipped to weight 0 entirely).
    #
    # --- Inputs ---
    # tw          : pd.Series — average uniqueness per observation, indexed
    #               by event date (this is the 'tW' output of
    #               get_average_uniqueness() from uniqueness.py)
    # clf_last_w  : float — desired weight for the OLDEST observation
    #               (default 1.0 = no decay at all)
    #
    # --- Output ---
    # pd.Series — same index as tw (sorted chronologically), values = the
    # final time-decayed weight for each observation, clipped at 0 (never
    # negative)

    # Sort chronologically (oldest first) and take the cumulative sum of
    # uniqueness. This is our x-axis: "how much unique information has
    # accumulated by this point in time."
    clf_w = tw.sort_index().cumsum()

    # Build the piecewise-linear decay line: y = const + slope * x
    # Anchored so the newest observation (max cumulative uniqueness) maps
    # to y=1, and the oldest observation (x=0) maps to y=clf_last_w.
    if clf_last_w >= 0:
        # Standard case: decay ranges from clf_last_w (oldest) up to 1 (newest),
        # always non-negative.
        slope = (1. - clf_last_w) / clf_w.iloc[-1]
    else:
        # clf_last_w < 0: the line is allowed to cross zero before reaching
        # the very oldest observation. Solving for the slope such that the
        # line hits exactly 0 at x = (clf_last_w+1)*clf_w.iloc[-1], and still
        # reaches y=1 at the newest observation.
        slope = 1. / ((clf_last_w + 1) * clf_w.iloc[-1])

    # The intercept is derived by requiring y=1 at the newest observation
    # (x = clf_w.iloc[-1], the final cumulative uniqueness value):
    #   1 = const + slope * clf_w.iloc[-1]  →  const = 1 - slope * clf_w.iloc[-1]
    const = 1. - slope * clf_w.iloc[-1]

    # Apply the line to every observation's cumulative-uniqueness x-value
    clf_w = const + slope * clf_w

    # Clip any negative weights to exactly 0 — only triggers when
    # clf_last_w < 0, where the line is allowed to dip below zero for
    # sufficiently old observations (hard-excluding them from training).
    clf_w[clf_w < 0] = 0

    return clf_w
