# vectorization.py -- AFML Section 20.2, Snippets 20.1 (un-vectorized) and
# 20.2 (vectorized) Cartesian product of a dictionary of lists.
#
# Book-fidelity translation notes (both snippets are printed in Python 2):
#   - `print {'a':a, ...}` (a print STATEMENT) -> `print({'a':a, ...})`
#     (a function call). Trivial syntax-only change, no semantics differ.
#   - Snippet 20.2's `from itertools import izip,product` -- `itertools.izip`
#     does not exist in Python 3 (removed; `zip` itself became lazy/iterator-
#     based in Python 3, which is exactly what izip provided in Python 2).
#     So the Python 3 equivalent is simply the builtin `zip`, no import
#     needed for that half. `itertools.product` is unchanged.
#   - `dict0.values()` -- in Python 2 this returned a list; in Python 3.7+
#     dict iteration order is guaranteed insertion order, so `dict0.values()`
#     and `dict0.keys()` are already guaranteed to correspond position-for-
#     position (no separate `dict0.keys()` list was even needed by the
#     original snippet, though we thread keys through explicitly below
#     for clarity rather than relying on itertools.product's own zip-with-
#     keys-afterward trick, which needs the keys captured once, not
#     re-read from a live dict).


import itertools


def cartesian_product_unvectorized(dict0):
    # AFML Snippet 20.1 -- un-vectorized Cartesian product via nested
    # For loops. Only works cleanly for a small, hardcoded number of
    # dimensions (one explicit loop per key) -- doesn't generalize to an
    # arbitrary/runtime-determined number of dict0 keys, which is exactly
    # the limitation Snippet 20.2 exists to fix. Hardcoded to 3 keys here
    # (a, b, c) to mirror the book's own 3-loop example faithfully; this
    # function is NOT meant to generalize -- that's the whole point of
    # showing it next to the vectorized version.
    jobs = []
    for a in dict0['a']:
        for b in dict0['b']:
            for c in dict0['c']:
                jobs.append({'a': a, 'b': b, 'c': c})
    return jobs


def cartesian_product_vectorized(dict0):
    # AFML Snippet 20.2 -- vectorized Cartesian product via
    # itertools.product, generalizing to ANY number of dict0 keys with no
    # code changes (unlike the nested-loop version above, which needs one
    # explicit `for` per key).
    keys = list(dict0.keys())
    return [dict(zip(keys, combo)) for combo in itertools.product(*(dict0[k] for k in keys))]


def cartesian_product_vectorized_generator(dict0):
    # Same as cartesian_product_vectorized, but as a generator (lazy),
    # matching the book's own `jobs=(... for i in ...)` generator
    # expression in Snippet 20.2 rather than eagerly building a list --
    # matters when the Cartesian product is too large to hold in memory
    # at once (the book's own motivating question: "how would this look
    # with 100 dimensions?").
    keys = list(dict0.keys())
    return (dict(zip(keys, combo)) for combo in itertools.product(*(dict0[k] for k in keys)))
