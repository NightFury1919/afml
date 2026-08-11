"""
Chapter 16: Machine Learning Asset Allocation -- Critical Line Algorithm (CLA)
================================================================================

This is NOT an AFML book snippet. AFML Section 16.5/16.6 explicitly defers
CLA's implementation to a separate reference: Bailey, D.H. and Lopez de
Prado, M. (2013), "An Open-Source Implementation of the Critical-Line
Algorithm for Portfolio Optimization", Algorithms 2013, 6, 169-196.
This module is a faithful, from-scratch translation of that paper's
Appendix A.1 (Snippet 18) -- the complete CLA class -- with only the
minimal Python-2-to-3 fixes needed to run under mlfinlab (Python 3.10.20
/ numpy 1.23.5), per this project's book-fidelity convention extended to
external referenced material.

CLA solves the standard portfolio problem (minimize variance subject to
a target return, weights in [lB, uB], full investment) via Markowitz's
Critical Line Algorithm: rather than one QP solve per risk target, it
walks the sequence of "turning points" -- solutions where the set of
free (non-boundary) assets changes -- starting from the highest-return
corner and working down to the global minimum-variance portfolio. Every
turning point is an exact solution (no numerical QP tolerance), and the
entire efficient frontier is recovered as convex combinations between
neighboring turning points.

Python-2-to-3 fixes (documented per project convention, NOT book bugs --
these are language-version issues, the paper's logic is unchanged):
  1. `if wB==None:` / `if self.l[-1]==None:` -- comparing a numpy array
     (or None) with `==` is ambiguous in Py3/numpy (raises "truth value
     of an array is ambiguous", or for real None just silently wrong in
     Py2 vs. Py3 comparison semantics). Fixed to `is None` throughout.
  2. `if l>l_in:` / `if l>l_out:` where `l_in`/`l_out` are initialized to
     `None` -- Python 2 allowed comparing `float > None` (arbitrary but
     consistent ordering); Python 3 raises `TypeError: '>' not supported
     between instances of 'float' and 'NoneType'`. Fixed to explicit
     `if l_in is None or l > l_in:`.
  3. `a[:]=zip(range(...), b)` -- Python 2's `zip` returns a list;
     Python 3's `zip` returns a lazy iterator, which does not broadcast
     into a structured-array slice assignment. Fixed to `list(zip(...))`.
  4. `np.linspace(0, 1, points/len(self.w))` -- `points/len(self.w)` is
     true (float) division in Python 3; `linspace`'s `num` argument must
     be an int. Fixed to floor division `points//len(self.w)`.
  5. `range(len(self.w)-1)` then indexing `b[-1]` -- Python 3 `range`
     objects DO support negative indexing via `__getitem__` (unlike
     Python 2, where `range` returned a plain list) -- this one needed
     NO fix, kept as-is; noted here only so a future reader doesn't
     "fix" something that already works.

Hand-traced TDD: test_cla.py validates against the paper's own Section 5
numerical example (Table 1 inputs, Table 2 turning points, and the
getMaxSR/getMinVar figures quoted in the paper's prose) -- real,
published numbers, not just internal self-consistency checks.
"""
import numpy as np


def _scalar(x):
    """Extract a Python float from a (1,1)-shaped ndarray (or anything
    array-like with exactly one element).

    NOT a Py2/3 fix -- a numpy-version-compatibility fix. The paper's
    printed code uses bare `float(...)` on (1,1) arrays throughout,
    which numpy allowed (with at most a deprecation warning) through
    1.23.5 (this project's pinned mlfinlab version) but numpy >=2.0
    raises `TypeError: only 0-dimensional arrays can be converted to
    Python scalars` for anything with ndim>0, even shape (1,1). Using
    `.reshape(-1)[0]` here makes the class correct under both, rather
    than relying on this project's specific numpy pin never changing.
    """
    return float(np.asarray(x).reshape(-1)[0])


class CLA:
    """Markowitz's Critical Line Algorithm.

    Parameters
    ----------
    mean : ndarray, shape (n, 1)
        Vector of expected returns.
    covar : ndarray, shape (n, n)
        Covariance matrix.
    lB, uB : ndarray, shape (n, 1)
        Lower / upper bounds for each weight. Full investment
        (weights sum to 1) is implied, not passed separately.

    Attributes (populated by solve())
    ----------------------------------
    w : list of ndarray
        Weight vector at each turning point, in order from highest
        expected return (w[0]) to the global minimum-variance portfolio
        (w[-1]).
    l : list of float or None
        Lambda (the return-target Lagrange multiplier) at each turning
        point. l[0] is None (the first turning point is constructed
        directly, not via a lambda-driven transition).
    g : list of float or None
        Gamma at each turning point.
    f : list of list of int
        The free-asset set F used to compute each turning point.
    """

    def __init__(self, mean, covar, lB, uB):
        self.mean = mean
        self.covar = covar
        self.lB = lB
        self.uB = uB
        self.w = []   # solution
        self.l = []   # lambdas
        self.g = []   # gammas
        self.f = []   # free weights

    # ------------------------------------------------------------------
    def solve(self):
        """Compute the turning points, free sets, and weights."""
        f, w = self.initAlgo()
        self.w.append(np.copy(w))   # store solution
        self.l.append(None)
        self.g.append(None)
        self.f.append(f[:])
        while True:
            # 1) case a): bound one free weight
            l_in = None
            if len(f) > 1:
                covarF, covarFB, meanF, wB = self.getMatrices(f)
                covarF_inv = np.linalg.inv(covarF)
                j = 0
                for i in f:
                    l, bi = self.computeLambda(
                        covarF_inv, covarFB, meanF, wB, j,
                        [self.lB[i], self.uB[i]])
                    # LOAD-BEARING: `l is not None` guard is NOT a Py2/3
                    # translation fix -- it fixes a genuine bug in the
                    # reference paper's own code (see computeLambda's
                    # docstring / module header). Ethan sign-off: pending.
                    if l is not None and (l_in is None or l > l_in):
                        l_in, i_in, bi_in = l, i, bi
                    j += 1
            # 2) case b): free one bounded weight
            l_out = None
            if len(f) < self.mean.shape[0]:
                b = self.getB(f)
                for i in b:
                    covarF, covarFB, meanF, wB = self.getMatrices(f + [i])
                    covarF_inv = np.linalg.inv(covarF)
                    l, bi = self.computeLambda(
                        covarF_inv, covarFB, meanF, wB,
                        meanF.shape[0] - 1, self.w[-1][i])
                    # Py2/3 fix #1 + #2 (`is None` checks, explicit
                    # None-guard before comparison) PLUS the same
                    # LOAD-BEARING `l is not None` guard as above.
                    prev_l = self.l[-1]
                    if (l is not None
                            and (prev_l is None or l < prev_l)
                            and (l_out is None or l > l_out)):
                        l_out, i_out = l, i
            if ((l_in is None or l_in < 0) and (l_out is None or l_out < 0)):
                # 3) compute minimum variance solution
                self.l.append(0)
                covarF, covarFB, meanF, wB = self.getMatrices(f)
                covarF_inv = np.linalg.inv(covarF)
                meanF = np.zeros(meanF.shape)
            else:
                # 4) decide lambda
                if l_in is not None and (l_out is None or l_in > l_out):
                    self.l.append(l_in)
                    f.remove(i_in)
                    w[i_in] = bi_in   # set value at the correct boundary
                else:
                    self.l.append(l_out)
                    f.append(i_out)
                covarF, covarFB, meanF, wB = self.getMatrices(f)
                covarF_inv = np.linalg.inv(covarF)
            # 5) compute solution vector
            wF, g = self.computeW(covarF_inv, covarFB, meanF, wB)
            for i in range(len(f)):
                w[f[i]] = wF[i]
            self.w.append(np.copy(w))   # store solution
            self.g.append(g)
            self.f.append(f[:])
            if self.l[-1] == 0:
                break
        # 6) purge turning points
        self.purgeNumErr(10e-10)
        self.purgeExcess()

    # ------------------------------------------------------------------
    def initAlgo(self):
        """Find the first turning point: the smallest subset of assets
        with the highest returns such that the sum of their upper
        boundaries equals or exceeds one."""
        # 1) form structured array
        a = np.zeros((self.mean.shape[0]), dtype=[('id', int), ('mu', float)])
        b = [self.mean[i][0] for i in range(self.mean.shape[0])]
        # Py2/3 fix #3: zip() is a lazy iterator in Py3; must materialize
        # to a list before broadcasting into the structured-array slice.
        a[:] = list(zip(range(self.mean.shape[0]), b))
        # 2) sort structured array
        b = np.sort(a, order='mu')
        # 3) first free weight
        i, w = b.shape[0], np.copy(self.lB)
        while sum(w) < 1:
            i -= 1
            w[b[i][0]] = self.uB[b[i][0]]
            w[b[i][0]] += 1 - sum(w)
        return [b[i][0]], w

    # ------------------------------------------------------------------
    def computeBi(self, c, bi):
        if c > 0:
            bi = bi[1][0]
        if c < 0:
            bi = bi[0][0]
        return bi

    # ------------------------------------------------------------------
    def computeW(self, covarF_inv, covarFB, meanF, wB):
        # 1) compute gamma
        onesF = np.ones(meanF.shape)
        g1 = np.dot(np.dot(onesF.T, covarF_inv), meanF)
        g2 = np.dot(np.dot(onesF.T, covarF_inv), onesF)
        if wB is None:   # Py2/3 fix #1
            g, w1 = _scalar(-self.l[-1] * g1 / g2 + 1 / g2), 0
        else:
            onesB = np.ones(wB.shape)
            g3 = np.dot(onesB.T, wB)
            g4 = np.dot(covarF_inv, covarFB)
            w1 = np.dot(g4, wB)
            g4 = np.dot(onesF.T, w1)
            g = _scalar(-self.l[-1] * g1 / g2 + (1 - g3 + g4) / g2)
        # 2) compute weights
        w2 = np.dot(covarF_inv, onesF)
        w3 = np.dot(covarF_inv, meanF)
        return -w1 + g * w2 + self.l[-1] * w3, g

    # ------------------------------------------------------------------
    def computeLambda(self, covarF_inv, covarFB, meanF, wB, i, bi):
        # 1) C
        onesF = np.ones(meanF.shape)
        c1 = np.dot(np.dot(onesF.T, covarF_inv), onesF)
        c2 = np.dot(covarF_inv, meanF)
        c3 = np.dot(np.dot(onesF.T, covarF_inv), meanF)
        c4 = np.dot(covarF_inv, onesF)
        c = -c1 * c2[i] + c3 * c4[i]
        if c == 0:
            # BUG IN THE REFERENCE PAPER (not a Py2/3 issue): the printed
            # code does a bare `return` here (i.e. returns None). Every
            # caller does `l, bi = self.computeLambda(...)`, which
            # crashes with TypeError trying to unpack None -- in Python
            # 2 as much as Python 3. Returning (None, None) and having
            # every caller explicitly skip None candidates (see solve())
            # is the fix. Flagged for Ethan sign-off; c==0 is a genuine
            # degenerate case (the two ratio terms in the lambda formula
            # exactly cancel) rather than a normal-path condition, so it
            # may be rare in practice, but the crash is real if hit.
            return None, None
        # 2) bi
        if type(bi) == list:
            bi = self.computeBi(c, bi)
        # 3) lambda
        if wB is None:   # Py2/3 fix #1
            # all free assets
            return _scalar((c4[i] - c1 * bi) / c), bi
        else:
            onesB = np.ones(wB.shape)
            l1 = np.dot(onesB.T, wB)
            l2 = np.dot(covarF_inv, covarFB)
            l3 = np.dot(l2, wB)
            l2 = np.dot(onesF.T, l3)
            # Paper's prose formula names this term "l3[i]" in the final
            # expression (the printed Snippet 6 has a transcription typo
            # "i3[i]"; the working Snippet 18 code correctly uses l3[i] --
            # confirmed against the paper's own Appendix code, not fixed
            # from memory).
            return _scalar(((1 - l1 + l2) * c4[i] - c1 * (bi + l3[i])) / c), bi

    # ------------------------------------------------------------------
    def getMatrices(self, f):
        """Slice covarF, covarFB, meanF, wB for the given free set f."""
        covarF = self.reduceMatrix(self.covar, f, f)
        meanF = self.reduceMatrix(self.mean, f, [0])
        b = self.getB(f)
        covarFB = self.reduceMatrix(self.covar, f, b)
        wB = self.reduceMatrix(self.w[-1], b, [0])
        return covarF, covarFB, meanF, wB

    # ------------------------------------------------------------------
    def getB(self, f):
        return self.diffLists(list(range(self.mean.shape[0])), f)

    # ------------------------------------------------------------------
    def diffLists(self, list1, list2):
        return list(set(list1) - set(list2))

    # ------------------------------------------------------------------
    def reduceMatrix(self, matrix, listX, listY):
        """Reduce a matrix to the provided list of rows and columns."""
        if len(listX) == 0 or len(listY) == 0:
            return None
        matrix_ = matrix[:, listY[0]:listY[0] + 1]
        for i in listY[1:]:
            a = matrix[:, i:i + 1]
            matrix_ = np.append(matrix_, a, 1)
        matrix__ = matrix_[listX[0]:listX[0] + 1, :]
        for i in listX[1:]:
            a = matrix_[i:i + 1, :]
            matrix__ = np.append(matrix__, a, 0)
        return matrix__

    # ------------------------------------------------------------------
    def purgeNumErr(self, tol):
        """Purge turning points that violate the inequality constraints
        as a result of a near-singular covariance matrix."""
        i = 0
        while True:
            if i == len(self.w):
                break
            w = self.w[i]
            flagged = False
            for j in range(w.shape[0]):
                if w[j] - self.lB[j] < -tol or w[j] - self.uB[j] > tol:
                    del self.w[i]
                    del self.l[i]
                    del self.g[i]
                    del self.f[i]
                    flagged = True
                    break
            if not flagged:
                i += 1

    # ------------------------------------------------------------------
    def purgeExcess(self):
        """Remove turning points that violate the convex hull (result of
        an unnecessary drop in lambda)."""
        i, repeat = 0, False
        while True:
            if repeat is False:
                i += 1
            if i == len(self.w) - 1:
                break
            w = self.w[i]
            mu = np.dot(w.T, self.mean)[0, 0]
            j, repeat = i + 1, False
            while True:
                if j == len(self.w):
                    break
                w = self.w[j]
                mu_ = np.dot(w.T, self.mean)[0, 0]
                if mu < mu_:
                    del self.w[i]
                    del self.l[i]
                    del self.g[i]
                    del self.f[i]
                    repeat = True
                    break
                else:
                    j += 1

    # ------------------------------------------------------------------
    def getMinVar(self):
        """Return (std, weights) of the minimum-variance solution among
        all stored turning points (Section 4.1)."""
        var = []
        for w in self.w:
            a = np.dot(np.dot(w.T, self.covar), w)
            var.append(a)
        return min(var) ** .5, self.w[var.index(min(var))]

    # ------------------------------------------------------------------
    def getMaxSR(self):
        """Return (Sharpe ratio, weights) of the maximum-Sharpe-ratio
        portfolio, found via Golden Section search on each segment
        between neighboring turning points (Section 4.2)."""
        w_sr, sr = [], []
        for i in range(len(self.w) - 1):
            w0 = np.copy(self.w[i])
            w1 = np.copy(self.w[i + 1])
            kargs = {'minimum': False, 'args': (w0, w1)}
            a, b = self.goldenSection(self.evalSR, 0, 1, **kargs)
            w_sr.append(a * w0 + (1 - a) * w1)
            sr.append(b)
        return max(sr), w_sr[sr.index(max(sr))]

    # ------------------------------------------------------------------
    def evalSR(self, a, w0, w1):
        """Evaluate the Sharpe ratio of the portfolio at convex-combination
        parameter `a` between turning points w0 and w1."""
        w = a * w0 + (1 - a) * w1
        b = np.dot(w.T, self.mean)[0, 0]
        c = np.dot(np.dot(w.T, self.covar), w)[0, 0] ** .5
        return b / c

    # ------------------------------------------------------------------
    def goldenSection(self, obj, a, b, **kargs):
        """Golden Section search. Finds a minimum by default; pass
        kargs['minimum']=False to search for a maximum."""
        from math import log, ceil
        tol, sign, args = 1.0e-9, 1, None
        if 'minimum' in kargs and kargs['minimum'] == False:
            sign = -1
        if 'args' in kargs:
            args = kargs['args']
        numIter = int(ceil(-2.078087 * log(tol / abs(b - a))))
        r = 0.618033989
        c = 1.0 - r
        # initialize
        x1 = r * a + c * b
        x2 = c * a + r * b
        f1 = sign * obj(x1, *args)
        f2 = sign * obj(x2, *args)
        # loop
        for i in range(numIter):
            if f1 > f2:
                a = x1
                x1 = x2
                f1 = f2
                x2 = c * a + r * b
                f2 = sign * obj(x2, *args)
            else:
                b = x2
                x2 = x1
                f2 = f1
                x1 = r * a + c * b
                f1 = sign * obj(x1, *args)
        if f1 < f2:
            return x1, sign * f1
        else:
            return x2, sign * f2

    # ------------------------------------------------------------------
    def efFrontier(self, points):
        """Compute `points` points of the efficient frontier as convex
        combinations between neighboring turning points."""
        mu, sigma, weights = [], [], []
        # Py2/3 fix #4: points/len(self.w) is float division in Py3;
        # linspace's num arg must be an int.
        a = np.linspace(0, 1, points // len(self.w))[:-1]
        b = list(range(len(self.w) - 1))
        for i in b:
            w0, w1 = self.w[i], self.w[i + 1]
            if i == b[-1]:
                a = np.linspace(0, 1, points // len(self.w))
            for j in a:
                w = w1 * j + (1 - j) * w0
                weights.append(np.copy(w))
                mu.append(np.dot(w.T, self.mean)[0, 0])
                sigma.append(np.dot(np.dot(w.T, self.covar), w)[0, 0] ** .5)
        return mu, sigma, weights
