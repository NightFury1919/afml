"""
TDD suite for vectorization.py (AFML Snippets 20.1/20.2, Cartesian product
of a dict of lists). Hand-computed expected job lists -- 2x2x2=8 combos for
the book's own example dict.
"""
import os
import sys

import pytest

sys.path.insert(0, os.path.dirname(__file__))
from vectorization import (
    cartesian_product_unvectorized,
    cartesian_product_vectorized,
    cartesian_product_vectorized_generator,
)


BOOK_DICT0 = {'a': ['1', '2'], 'b': ['+', '*'], 'c': ['!', '@']}

# LOAD-BEARING: hand-enumerated 2x2x2=8 combinations, in the same order the
# book's own triple-nested for-loop (a outer, b middle, c inner) would
# produce them.
EXPECTED_JOBS = [
    {'a': '1', 'b': '+', 'c': '!'}, {'a': '1', 'b': '+', 'c': '@'},
    {'a': '1', 'b': '*', 'c': '!'}, {'a': '1', 'b': '*', 'c': '@'},
    {'a': '2', 'b': '+', 'c': '!'}, {'a': '2', 'b': '+', 'c': '@'},
    {'a': '2', 'b': '*', 'c': '!'}, {'a': '2', 'b': '*', 'c': '@'},
]


class TestCartesianProductUnvectorized:
    def test_matches_hand_enumerated_jobs(self):
        assert cartesian_product_unvectorized(BOOK_DICT0) == EXPECTED_JOBS

    def test_job_count_is_product_of_list_lengths(self):
        dict0 = {'a': ['1', '2'], 'b': ['+', '*'], 'c': ['!', '@']}
        assert len(cartesian_product_unvectorized(dict0)) == 2 * 2 * 2


class TestCartesianProductVectorized:
    def test_matches_unvectorized_version(self):
        # The whole point of Snippet 20.2 is that it's equivalent output to
        # Snippet 20.1, just generalized -- verify that directly.
        assert cartesian_product_vectorized(BOOK_DICT0) == \
            cartesian_product_unvectorized(BOOK_DICT0)

    def test_matches_hand_enumerated_jobs(self):
        assert cartesian_product_vectorized(BOOK_DICT0) == EXPECTED_JOBS

    def test_generalizes_beyond_three_keys(self):
        # The book's own motivating question: "how would this code look for
        # 100 dimensions?" -- confirm it works unmodified for 4+ keys, which
        # the hardcoded-nested-loop version structurally cannot do.
        dict0 = {'w': ['x', 'y'], 'a': [1], 'b': [2, 3], 'c': [4, 5, 6]}
        out = cartesian_product_vectorized(dict0)
        assert len(out) == 2 * 1 * 2 * 3
        assert all(set(job.keys()) == {'w', 'a', 'b', 'c'} for job in out)

    def test_single_valued_lists_give_single_combo(self):
        dict0 = {'a': ['only']}
        assert cartesian_product_vectorized(dict0) == [{'a': 'only'}]

    def test_empty_list_gives_no_combos(self):
        # An empty dimension makes the whole Cartesian product empty --
        # itertools.product's own documented behavior, worth pinning here
        # since it's easy to assume a dimension with 0 options gets skipped
        # rather than collapsing the entire result to nothing.
        dict0 = {'a': ['1', '2'], 'b': []}
        assert cartesian_product_vectorized(dict0) == []


class TestCartesianProductVectorizedGenerator:
    def test_is_lazy_not_a_list(self):
        gen = cartesian_product_vectorized_generator(BOOK_DICT0)
        assert not isinstance(gen, list)

    def test_materializes_to_same_result_as_list_version(self):
        gen = cartesian_product_vectorized_generator(BOOK_DICT0)
        assert list(gen) == cartesian_product_vectorized(BOOK_DICT0)


if __name__ == '__main__':
    pytest.main([__file__, '-v'])
