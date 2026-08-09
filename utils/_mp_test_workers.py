# _mp_test_workers.py
# Module-level worker functions used by test_multiprocess.py.
#
# WHY THIS FILE EXISTS: on Windows, multiprocessing uses 'spawn' (not
# 'fork'), which means worker processes re-import the module that defines
# the target function from scratch. A function defined inline inside a test
# (or a lambda) is not picklable/importable that way and will hang or raise
# PicklingError on Windows even though it may work fine on Linux (fork).
# Keeping these at true module level, in their own file, avoids that
# footgun for real on the target machine, not just in the Linux sandbox.

import pandas as pd


def square(molecule):
    # Simple deterministic worker: squares each atom in molecule.
    # Used as the 'func' passed to mp_pandas_obj / process_jobs /
    # process_jobs_mp, with pdObj arg name 'molecule' matching book usage.
    return pd.Series({i: i ** 2 for i in molecule})


def double_list(molecule):
    # Returns a plain list (not a pandas object) -- exercises mp_pandas_obj's
    # "not a DataFrame/Series" fallback branch (returns raw `out` list).
    return [i * 2 for i in molecule]


def sum_molecule(molecule):
    # Returns a plain Python int -- used for process_jobs_redux with a
    # numeric (operator.add) reducer.
    return sum(molecule)


def series_from_molecule(molecule):
    # Returns a one-entry-per-atom Series -- used for process_jobs_redux
    # with pd.Series.add as the reducer (tests the non-in-place path).
    return pd.Series({i: 1 for i in molecule})


def list_of_molecule(molecule):
    # Returns a plain list -- used for process_jobs_redux with
    # list.append as an explicit in-place reducer.
    return list(molecule)


def dict_from_molecule(molecule):
    # Returns a {atom: atom**2} dict -- used for process_jobs_redux with
    # dict.update (the book's own worked example of an in-place reducer).
    return {i: i ** 2 for i in molecule}
