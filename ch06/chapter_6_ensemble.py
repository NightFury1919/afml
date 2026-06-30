"""
Snippet 6.1 -- Accuracy of the bagging classifier.

PLAIN-ENGLISH IDEA:
If each of N independent classifiers is individually correct with
probability p, this function computes P[X > N/k] -- the probability
that the number of correct votes X exceeds the per-class "fair share"
threshold N/k. This is a NECESSARY (but not sufficient) condition for
the bagging ensemble to make the correct prediction by majority vote.

WHY THIS MATTERS:
- If p > 1/k (individual classifier beats random guessing), then as N
  grows, P[X > N/k] grows toward 1 -- the ensemble gets more reliable.
- If p <= 1/k (individual classifier is at or below random guessing),
  then as N grows, P[X > N/k] shrinks toward 0 -- more classifiers
  makes things WORSE, not better.
- This is the formal proof that bagging amplifies whatever tendency
  (good or bad) the individual classifiers already have. It cannot
  rescue a fundamentally poor classifier (p <= 1/k).

NECESSARY vs SUFFICIENT:
- SUFFICIENT condition for correct ensemble prediction: X > N/2
  (strict majority). If true, the correct class is guaranteed to win.
- NECESSARY condition: X > N/k. If false, the correct class cannot
  possibly win. But clearing it doesn't guarantee a win either --
  a rival class could also clear it with more votes.

FORMULA:
P[X > N/k] = 1 - sum_{i=0}^{floor(N/k)} C(N,i) * p^i * (1-p)^(N-i)

where C(N,i) is the binomial coefficient. X ~ Binomial(N, p) since
classifiers are assumed independent, each correct with probability p.

BOOK ERRATA (Snippet 6.1 as printed):
1. `from scipy.misc import comb` -- scipy.misc.comb was removed; use
   scipy.special.comb instead.
2. `xrange` -- Python 2 only; use range() in Python 3.
3. `print p, 1-p_` -- Python 2 print statement; use print() in Python 3.
All three are corrected below.
"""

from scipy.special import comb


def bagging_classifier_accuracy(N: int, p: float, k: int) -> float:
    """
    Compute P[X > N/k] -- the probability that the number of correct
    votes among N independent classifiers exceeds the per-class
    threshold N/k.

    Parameters
    ----------
    N : int
        Number of independent base classifiers in the ensemble.
    p : float
        Accuracy of each individual classifier (probability of a
        correct prediction). Must be in (0, 1).
    k : int
        Number of classes. For binary classification k=2, so the
        threshold is N/2 (simple majority). For k=3 classes the
        threshold is N/3 -- a lower bar to clear.

    Returns
    -------
    float
        P[X > N/k] -- the probability the necessary condition for a
        correct ensemble prediction is met. This is a lower bound on
        the actual probability of a correct ensemble prediction.

    Examples
    --------
    >>> # Book's own example: N=100, p=1/3, k=3
    >>> # At exactly the random-guessing baseline, P[X > N/k] ~ 0.48,
    >>> # already slightly above p itself.
    >>> round(bagging_classifier_accuracy(100, 1/3, 3), 4)
    0.4812

    >>> # p slightly above 1/k: ensemble improves markedly with N
    >>> round(bagging_classifier_accuracy(100, 0.4, 3), 4)
    0.9087

    >>> # p below 1/k: ensemble performs WORSE than individual
    >>> round(bagging_classifier_accuracy(100, 0.3, 3), 4)
    0.2207
    """
    if not 0 < p < 1:
        raise ValueError(f"p must be in (0, 1), got {p}")
    if N < 1:
        raise ValueError(f"N must be >= 1, got {N}")
    if k < 2:
        raise ValueError(f"k must be >= 2, got {k}")

    # Cumulative probability of X <= floor(N/k) -- i.e. failing to
    # clear the necessary-condition threshold.
    p_fail = sum(
        comb(N, i, exact=False) * p**i * (1 - p)**(N - i)
        for i in range(0, int(N / k) + 1)
    )
    return 1 - p_fail


# ---------------------------------------------------------------------
# TDD TEST RESULTS (tests/test_ch06.py)
# Run 2026-06-30. All verified against the book's argument and
# independently cross-checked against scipy.stats.binom.cdf.
# ---------------------------------------------------------------------
# test_book_example_n100_p_third_k3                          PASSED
# test_above_random_guessing_gives_high_probability_at_large_N PASSED
# test_below_random_guessing_declines_with_more_estimators    PASSED
# test_at_random_guessing_threshold_stays_flat                PASSED
# test_matches_scipy_binom_cdf_independently                  PASSED
# test_binary_classification_k2                               PASSED
# test_monotonically_increases_with_N_when_above_threshold    PASSED
# test_invalid_p_raises                                       PASSED
# test_invalid_N_raises                                       PASSED
# test_invalid_k_raises                                       PASSED
# 10 passed in 1.18s
# ---------------------------------------------------------------------
