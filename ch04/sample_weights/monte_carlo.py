import numpy as np
import pandas as pd
import sys
import os

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '..')))
from ch04.sample_weights.indicator_matrix      import get_ind_matrix
from ch04.sample_weights.avg_uniqueness_matrix import get_avg_uniqueness
from ch04.sample_weights.sequential_bootstrap  import seq_bootstrap

# Generating a Random t1 Series — AFML Chapter 4, Snippet 4.7, page 66-67
#
# Generates a synthetic, randomly-overlapping set of observations for use in
# the Monte Carlo experiment (Section 4.5.4) that measures how much better
# sequential bootstrap is compared to standard bootstrap, on average.
#
# --- Why do we need randomly generated data here? ---
# The hand-worked 3-observation example (Section 4.5.3) proves the MECHANICS
# of sequential bootstrap work correctly, but it's just one tiny case. To know
# whether sequential bootstrap is ACTUALLY better on realistic data, we need
# to test it across many different randomly-generated overlap patterns and
# see if the improvement holds up statistically. This function creates one
# such random scenario; the Monte Carlo loop (Snippet 4.9) calls it
# repeatedly (e.g. a million times) and aggregates the results.
#
# --- What does each parameter control? ---
# num_obs  : how many observations (events) to generate
# num_bars : how many total bars exist in the synthetic price series
#            (observations are placed somewhere within this range)
# max_h    : the maximum possible duration (in bars) of any one observation
#            Larger max_h → observations span more bars → more overlap likely


def get_rnd_t1(num_obs, num_bars, max_h):
    # Generate a random t1 Series — AFML Snippet 4.7
    #
    # --- Inputs ---
    # num_obs  : int — number of observations to generate
    # num_bars : int — total number of bars in the synthetic price series
    # max_h    : int — maximum observation duration in bars
    #
    # --- Output ---
    # pd.Series — index = observation start bar (t0), values = observation
    #             end bar (t1), sorted by start bar. Note: if two observations
    #             happen to start on the SAME bar, the later one silently
    #             overwrites the earlier one in the Series (this mirrors the
    #             book's exact behavior via .loc[ix]=val).

    t1 = pd.Series(dtype=float)

    for i in range(num_obs):
        # Pick a random starting bar anywhere in [0, num_bars)
        ix = np.random.randint(0, num_bars)

        # Pick a random duration in [1, max_h), and set the end bar
        # This guarantees every observation lasts AT LEAST 1 bar
        val = ix + np.random.randint(1, max_h)

        t1.loc[ix] = val

    return t1.sort_index()


# Uniqueness from Standard and Sequential Bootstraps — AFML Chapter 4,
# Snippet 4.8, page 67
#
# A single Monte Carlo trial: generate one random overlapping observation
# set, run BOTH standard bootstrap and sequential bootstrap on it, and
# return the average uniqueness each method achieved. This is the function
# that gets called thousands (or millions) of times by the outer Monte Carlo
# loop (Snippet 4.9) to build up a statistically meaningful comparison.
#
# --- Why package this as one function? ---
# Each call to aux_mc is completely independent of every other call — it
# generates its own random data, runs both methods, and returns a result.
# This independence is exactly what makes it trivial to parallelize: you can
# run thousands of these trials simultaneously across CPU cores with no
# coordination needed between them (see Snippet 4.9, which uses
# mp_pandas_obj-style multiprocessing to do exactly this).


def aux_mc(num_obs, num_bars, max_h):
    # Run one Monte Carlo trial comparing standard vs sequential bootstrap
    # — AFML Snippet 4.8
    #
    # --- Inputs ---
    # num_obs  : int — number of observations to generate for this trial
    # num_bars : int — total number of bars in the synthetic price series
    # max_h    : int — maximum observation duration in bars
    #
    # --- Output ---
    # dict — {'stdU': standard bootstrap uniqueness,
    #         'seqU': sequential bootstrap uniqueness}
    #        Both are floats in (0, 1]. Higher is better (less redundancy).

    # Generate one random, overlapping observation set (Snippet 4.7)
    t1 = get_rnd_t1(num_obs, num_bars, max_h)

    # Build the indicator matrix for this random observation set (Snippet 4.3)
    bar_ix = range(int(t1.max()) + 1)
    ind_m = get_ind_matrix(bar_ix, t1)

    # --- Standard bootstrap ---
    # Draw with replacement, EQUAL probability every draw — the naive approach
    phi = np.random.choice(ind_m.columns, size=ind_m.shape[1])
    std_u = get_avg_uniqueness(ind_m[phi]).mean()

    # --- Sequential bootstrap ---
    # Draw with replacement, but weighted toward unique observations (Snippet 4.5)
    phi = seq_bootstrap(ind_m)
    seq_u = get_avg_uniqueness(ind_m[phi]).mean()

    return {'stdU': std_u, 'seqU': seq_u}


# Multi-Threaded Monte Carlo — AFML Chapter 4, Snippet 4.9, page 67-68
#
# The outer driver: build a list of identical job specs (each one calling
# aux_mc with the same parameters), run them all (optionally in parallel
# across multiple CPU cores), and report summary statistics comparing
# standard bootstrap uniqueness vs sequential bootstrap uniqueness across
# every trial.
#
# --- Why does running the SAME job spec many times produce different results? ---
# aux_mc internally calls get_rnd_t1 and np.random.choice, both of which draw
# fresh random numbers every call. So even though every job in the list looks
# identical on paper, each one generates its own random overlapping
# observation set and its own random bootstrap draws — exactly what we want
# for a Monte Carlo experiment.
#
# --- Performance note from the book ---
# The book states that numIters=1e6 with numObs=10, numBars=100, maxH=5 takes
# about 6 hours on a 24-core server (vs. 6 days single-threaded). For
# practical use on your machine, start with a much smaller numIters
# (e.g. 1000-10000) to get a feel for the result before scaling up.
#
# --- Reusing your existing multiprocessing engine ---
# The book introduces a separate mpEngine module with processJobs/processJobs_
# for this snippet. Since you already have process_jobs (single-threaded) and
# process_jobs_mp (multi-threaded) in utils/multiprocess.py — built for the
# mp_pandas_obj pattern in Chapters 3 and 4 — we reuse those directly here
# rather than introducing a second, separate multiprocessing system.

from utils.multiprocess import process_jobs, process_jobs_mp


def main_mc(num_obs=10, num_bars=100, max_h=5, num_iters=1e6, num_threads=1):
    # Run the full Monte Carlo experiment — AFML Snippet 4.9
    #
    # --- Inputs ---
    # num_obs    : int   — observations per trial (passed to aux_mc)
    # num_bars   : int   — bars per trial (passed to aux_mc)
    # max_h      : int   — max observation duration per trial (passed to aux_mc)
    # num_iters  : int   — total number of Monte Carlo trials to run
    #              Default matches the book (1e6), but start smaller for testing.
    # num_threads: int   — number of parallel workers (default 1, single-threaded)
    #              NOTE: on Windows, num_threads > 1 requires this function to
    #              be called from inside an `if __name__ == '__main__':` guard,
    #              same constraint as utils/multiprocess.py's process_jobs_mp.
    #
    # --- Output ---
    # pd.DataFrame — one row per trial, columns 'stdU' and 'seqU'
    #                Also prints summary statistics (.describe()) to match
    #                the book's reporting style.

    # Build one identical job spec per iteration. Each job, when run, calls
    # aux_mc(num_obs=num_obs, num_bars=num_bars, max_h=max_h) — the randomness
    # comes from inside aux_mc itself, not from varying the job spec.
    jobs = []
    for i in range(int(num_iters)):
        job = {'num_obs': num_obs, 'num_bars': num_bars, 'max_h': max_h}
        jobs.append(job)

    if num_threads == 1:
        out = process_jobs(aux_mc, jobs)
    else:
        out = process_jobs_mp(aux_mc, jobs, num_threads)

    result = pd.DataFrame(out)
    print(result.describe())
    return result
