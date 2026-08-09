import copy
import sys
import time
import numpy as np
import pandas as pd
import multiprocessing as mp
import datetime

# multiprocess.py — AFML shared utility
# Multiprocessing engine used throughout AFML chapters 3, 4, and beyond.
# Referenced in Snippet 3.3, Snippet 4.2, and Chapter 20.
#
# --- What does mpPandasObj do? ---
# Many AFML functions need to loop over a large pandas index (e.g. thousands
# of CUSUM events) and apply a function to each one. This is slow when done
# single-threaded. mpPandasObj splits the index into chunks, runs the function
# on each chunk in a separate CPU core in parallel, then reassembles the results.
#
# --- How the book calls it ---
# mpPandasObj(func, pdObj, numThreads, **kwargs)
#
#   func       : the function to call on each chunk
#   pdObj      : tuple ('argument_name', pandas_index)
#                'argument_name' tells mpPandasObj which argument of func
#                receives the chunk. e.g. ('molecule', events.index)
#   numThreads : how many parallel processes to use
#                1 = single-threaded (safest for debugging)
#                mp.cpu_count() = use all available cores
#   **kwargs   : any other arguments to pass to func unchanged
#
# --- What func must look like ---
# func(molecule, **kwargs) → pd.Series or pd.DataFrame
#   molecule = the chunk of the pandas index assigned to this worker
#   The function processes only the rows in molecule and returns results
#   for those rows. mpPandasObj concatenates all chunks at the end.
#
# --- Example (from Snippet 3.3) ---
# df0 = mpPandasObj(
#     func=apply_pt_sl_on_t1,
#     pdObj=('molecule', events.index),
#     numThreads=numThreads,
#     close=close,
#     events=events,
#     pt_sl=[pt_sl, pt_sl]
# )
# Here apply_pt_sl_on_t1 receives a 'molecule' argument (a chunk of events.index)
# and uses it to filter which events to process.


def lin_parts(num_atoms, num_threads):
    # Partition a range of num_atoms into num_threads roughly equal parts.
    # Returns an array of partition boundaries.
    # Example: lin_parts(10, 3) → [0, 4, 7, 10]  (3 chunks of 4, 3, 3)
    parts = np.linspace(0, num_atoms, min(num_threads, num_atoms) + 1)
    parts = np.ceil(parts).astype(int)
    return parts


def mp_pandas_obj(func, pd_obj, num_threads=1, mp_batches=1, lin_mols=True, **kwargs):
    # mpPandasObj — AFML Chapter 20 multiprocessing engine
    # Used throughout the book: Snippets 3.3, 4.2, and many more.
    #
    # --- Inputs ---
    # func        : callable — function to apply to each chunk
    #               Must accept (molecule, **kwargs) as its first argument
    # pd_obj      : tuple ('arg_name', pandas_index)
    #               'arg_name' is the keyword argument name func expects
    #               pandas_index is the full index to split into chunks
    # num_threads : int — number of parallel processes (default 1)
    #               Set to 1 for debugging; mp.cpu_count() for full speed
    # mp_batches  : int — number of batches per thread (default 1)
    #               Higher values reduce memory per batch but add overhead
    # lin_mols    : bool — use linear partitioning (True) vs nested (False)
    #               Linear is correct for most AFML use cases
    # **kwargs    : passed unchanged to func
    #
    # --- Output ---
    # pd.DataFrame or pd.Series — concatenated results from all chunks

    # Unpack pdObj tuple
    arg_name, pd_index = pd_obj

    # Split the index into chunks — one per job
    parts = lin_parts(len(pd_index), num_threads * mp_batches)
    jobs  = []
    for i in range(1, len(parts)):
        # Each job gets a slice of the index (called 'molecule' in the book)
        molecule = pd_index[parts[i - 1]:parts[i]]
        # Build the keyword arguments for this job
        job_kwargs = {arg_name: molecule}
        job_kwargs.update(kwargs)
        jobs.append(job_kwargs)

    # Guard: if the index was empty (e.g. min_ret filtered out all events),
    # there are no jobs to run — return an empty DataFrame immediately.
    if len(jobs) == 0:
        return pd.DataFrame()

    if num_threads == 1:
        # Single-threaded: run jobs sequentially
        # This is the safe default for debugging and small datasets
        out = process_jobs(func, jobs)
    else:
        # Multi-threaded: run jobs in parallel across CPU cores
        out = process_jobs_mp(func, jobs, num_threads)

    # Concatenate all chunk results into one DataFrame/Series
    if isinstance(out[0], pd.DataFrame):
        return pd.concat(out)
    elif isinstance(out[0], pd.Series):
        return pd.concat(out)
    else:
        return out


def process_jobs(func, jobs):
    # Single-threaded execution — run each job in sequence.
    # Used when num_threads=1 (debugging, small datasets).
    out = []
    for job in jobs:
        out.append(func(**job))
    return out


def process_jobs_mp(func, jobs, num_threads):
    # Multi-threaded execution using Python's multiprocessing pool.
    # Each job runs in a separate process on its own CPU core.
    #
    # Note: on Windows, multiprocessing requires the entry point to be
    # protected by if __name__ == '__main__'. If you hit issues, set
    # num_threads=1 as a fallback.
    pool    = mp.Pool(processes=num_threads)
    outputs = pool.map(_job_wrapper, [(func, job) for job in jobs])
    pool.close()
    pool.join()
    return outputs


def _job_wrapper(args):
    # Helper needed because mp.Pool.map only accepts one argument.
    # Unpacks (func, kwargs) and calls func(**kwargs).
    func, kwargs = args
    return func(**kwargs)


# =============================================================================
# AFML Chapter 20 additions (2026-08-08) -- nested_parts, report_progress,
# and the output-reduction path (process_jobs_redux / mp_job_list).
#
# ADDITIVE ONLY: everything above this line (lin_parts, mp_pandas_obj,
# process_jobs, process_jobs_mp, _job_wrapper) is untouched. Ch03, Ch04,
# Ch08, and Ch10 all import those functions and are already real-machine
# confirmed against them -- changing their signatures or behavior here
# would risk silently breaking four closed chapters. Everything below is
# new surface area only.
#
# Book-fidelity note: Snippets 20.5-20.13 are printed in Python 2 (xrange,
# print-statement, im_func/im_self/im_class for bound-method pickling).
# Translated to Python 3 below: xrange->range throughout. Section 20.5.4's
# copy_reg-based bound-method pickle registration is NOT ported -- it's a
# Python-2-specific workaround (Python 2's bound methods aren't pickleable
# at all without it; Python 3's bound methods pickle natively via
# __reduce__ in the normal case). Porting im_func/im_self/im_class verbatim
# would simply raise AttributeError in Python 3, since those attributes
# were renamed to __func__/__self__ back in Python 3.0 and the whole
# workaround is unneeded here. This is documented rather than silently
# dropped, per the book-fidelity rule (identify + fix/adapt real snippet
# issues, don't reconstruct from memory or skip silently).
# =============================================================================


def nested_parts(num_atoms, num_threads, upper_triang=False):
    # AFML Snippet 20.6 (nestedParts) -- partition of atoms with an inner
    # loop, i.e. molecules sized for a lower- (or upper-) triangular
    # workload (e.g. a two-nested-loop SADF-style computation, Chapter 17)
    # rather than lin_parts' equal-atoms-per-molecule split, which would
    # badly imbalance a triangular task (early/late rows do very
    # different amounts of work).
    #
    # LOAD-BEARING: closed-form solution to r_m = (-1 + sqrt(1 + 4*(r_{m-1}^2
    # + r_{m-1} + N(N+1)/M))) / 2, solved for the row boundary that makes
    # each of M molecules carry ~1/M of the N(N+1)/2 total triangular-task
    # count. See book Section 20.4.2 for the derivation.
    parts, num_threads_ = [0], min(num_threads, num_atoms)
    for _ in range(num_threads_):
        part = 1 + 4 * (parts[-1] ** 2 + parts[-1]
                         + num_atoms * (num_atoms + 1.) / num_threads_)
        part = (-1 + part ** .5) / 2.
        parts.append(part)
    parts = np.round(parts).astype(int)
    if upper_triang:
        # For an upper-triangular workload (row i has N-i+1 tasks, so row 1
        # is heaviest and row N lightest), reverse the molecule sizes so the
        # heaviest molecule still comes first in the partition.
        parts = np.cumsum(np.diff(parts)[::-1])
        parts = np.append(np.array([0]), parts)
    return parts


def report_progress(job_num, num_jobs, time0, task):
    # AFML Snippet 20.9 (reportProgress) -- progress message for long-
    # running multiprocessing jobs. Returns the formatted message (so it's
    # testable) and also writes it to stderr, matching the book's behavior.
    msg = [float(job_num) / num_jobs, (time.time() - time0) / 60.]
    msg.append(msg[1] * (1 / msg[0] - 1))
    time_stamp = str(datetime.datetime.fromtimestamp(time.time()))
    msg = (time_stamp + ' ' + str(round(msg[0] * 100, 2)) + '% ' + task
           + ' done after ' + str(round(msg[1], 2)) + ' minutes. Remaining '
           + str(round(msg[2], 2)) + ' minutes.')
    if job_num < num_jobs:
        sys.stderr.write(msg + '\r')
    else:
        sys.stderr.write(msg + '\n')
    return msg


def process_jobs_redux(func, jobs, num_threads=1, redux=None, redux_args=None,
                        redux_in_place=False, report=False):
    # AFML Snippet 20.12 (processJobsRedux) -- like process_jobs/
    # process_jobs_mp above, but reduces molecular outputs ON THE FLY as
    # they arrive, instead of collecting all of them into a list first.
    # This matters when outputs are large: waiting for every molecule to
    # finish before combining them risks a memory error, whereas reducing
    # as results stream back keeps peak memory bounded.
    #
    # Kept as a separate function from process_jobs/process_jobs_mp (rather
    # than adding a redux= kwarg to those) so their existing, already-relied-
    # -on behavior (return a plain list, used by Ch03/04/08/10) is never at
    # risk of being changed.
    #
    # func/jobs convention matches this file's existing process_jobs (func
    # passed separately, jobs are plain kwarg dicts) rather than the book's
    # job['func'] + expandCall pattern -- that divergence was already
    # established by the pre-existing code in this file, not introduced here.
    if redux_args is None:
        redux_args = {}

    time0 = time.time()
    out = None

    if num_threads == 1:
        # Sequential path -- for debugging, matches book's intent that
        # numThreads==1 be usable to isolate bugs from multiprocessing itself.
        results = (func(**job) for job in jobs)
    else:
        pool = mp.Pool(processes=num_threads)
        results = pool.imap_unordered(_job_wrapper, [(func, job) for job in jobs])

    for i, out_ in enumerate(results, 1):
        if out is None:
            if redux is None:
                # No reducer given: fall back to plain list-accumulation,
                # matching the book's documented fallback behavior.
                out, redux, redux_in_place = [out_], list.append, True
            else:
                out = copy.deepcopy(out_)
        else:
            if redux_in_place:
                redux(out, out_, **redux_args)
            else:
                out = redux(out, out_, **redux_args)
        if report:
            report_progress(i, len(jobs), time0, func.__name__)

    if num_threads != 1:
        pool.close()
        pool.join()

    if isinstance(out, (pd.Series, pd.DataFrame)):
        out = out.sort_index()
    return out


def mp_job_list(func, arg_list, num_threads=1, mp_batches=1, lin_mols=True,
                 redux=None, redux_args=None, redux_in_place=False,
                 report=False, **kwargs):
    # AFML Snippet 20.13 (mpJobList) -- like mp_pandas_obj above, but routes
    # through process_jobs_redux so molecular outputs get combined on the
    # fly rather than being collected into a list and concatenated at the
    # end. Use this (instead of mp_pandas_obj) when outputs are large enough
    # that holding all of them in memory simultaneously is a real concern
    # (book's example: Section 20.6's principal-components-by-file-chunk).
    #
    # arg_name/pd_index convention matches mp_pandas_obj's pd_obj tuple.
    arg_name, atoms = arg_list
    if lin_mols:
        parts = lin_parts(len(atoms), num_threads * mp_batches)
    else:
        parts = nested_parts(len(atoms), num_threads * mp_batches)

    jobs = []
    for i in range(1, len(parts)):
        molecule = atoms[parts[i - 1]:parts[i]]
        job_kwargs = {arg_name: molecule}
        job_kwargs.update(kwargs)
        jobs.append(job_kwargs)

    if len(jobs) == 0:
        return None

    return process_jobs_redux(func, jobs, num_threads=num_threads, redux=redux,
                               redux_args=redux_args, redux_in_place=redux_in_place,
                               report=report)
