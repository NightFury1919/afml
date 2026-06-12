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
