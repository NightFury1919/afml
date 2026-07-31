"""
Chapter 16: Machine Learning Asset Allocation -- Hierarchical Risk Parity (HRP)
================================================================================

Implements AFML Snippets 16.1-16.4 (tree clustering, quasi-diagonalization,
recursive bisection, and the full numerical-example driver).

Why HRP instead of Markowitz's CLA or naive risk parity (IVP)?
----------------------------------------------------------------
A covariance matrix is a complete graph: every asset is a potential
substitute for every other asset. Inverting it (which CLA and mean-variance
optimization require) means solving for all those pairwise relationships
at once -- a tiny estimation error in any one correlation can swing the
whole solution ("Markowitz's curse", Section 16.3). HRP sidesteps matrix
inversion entirely by replacing the complete graph with a TREE: cluster
similar assets together (stage 1), reorder the covariance matrix so
similar assets sit next to each other (stage 2), then allocate weight
top-down by recursively splitting each cluster in proportion to inverse
variance (stage 3). No inversion, no positive-definiteness requirement --
HRP works even on a singular covariance matrix.

Book bugs / Python-2-isms fixed here (see project handoff for detail):
  - getRecBipart's bisection used `len(i)/2`, which is integer division
    in Python 2 but float division in Python 3 -- would raise a TypeError
    on list slicing. Fixed to `len(i)//2`.
  - `pd.Series.append()` (used in getQuasiDiag) is deprecated in pandas
    1.5.3 and removed in pandas 2.x. Replaced with `pd.concat`.
  - `xrange` -> `range`, `print x` -> `print(x)`.

Fidelity note: function/variable names (getIVP, getClusterVar, getQuasiDiag,
getRecBipart, correlDist) are kept exactly as printed in the book, per this
project's book-fidelity convention -- not rewritten to snake_case.
"""
import warnings

import numpy as np
import pandas as pd
import scipy.cluster.hierarchy as sch


# =============================================================================
# Snippet 16.4 (helper): getIVP
# =============================================================================
def getIVP(cov, **kargs):
    """Compute the inverse-variance portfolio.

    Optimal allocation for a DIAGONAL covariance matrix (Appendix 16.A.2):
    weight on each asset is inversely proportional to its own variance,
    normalized to sum to 1. Used inside getClusterVar (stage 3's
    within-cluster weighting) and as one of the three allocations compared
    in the chapter (HRP vs CLA vs IVP).
    """
    ivp = 1. / np.diag(cov)
    ivp /= ivp.sum()
    return ivp


# =============================================================================
# Snippet 16.4 (helper): getClusterVar
# =============================================================================
def getClusterVar(cov, cItems):
    """Compute the variance of a cluster, assuming inverse-variance weights
    within that cluster (Section 16.4.3, step 3b).

    This is the quadratic form w'Vw where V is the covariance sub-matrix
    for just this cluster's constituents and w is their IVP weighting.
    getRecBipart calls this on each half of a bisection to decide how much
    weight that half should receive relative to the other half.
    """
    cov_ = cov.loc[cItems, cItems]          # matrix slice: just this cluster
    w_ = getIVP(cov_).reshape(-1, 1)
    cVar = np.dot(np.dot(w_.T, cov_), w_)[0, 0]
    return cVar


# =============================================================================
# Snippet 16.2 / 16.4: getQuasiDiag
# =============================================================================
def getQuasiDiag(link):
    """Stage 2: quasi-diagonalization (Section 16.4.2).

    Reorders items so that similar investments sit next to each other in
    the covariance matrix, without changing basis (no PCA-style rotation --
    the original assets are preserved, just reordered). Works by walking
    the linkage matrix from its final (top) row downward, recursively
    replacing each cluster reference with its constituent items until only
    original (unclustered) item indices remain.

    Parameters
    ----------
    link : ndarray, shape (N-1, 4)
        A scipy linkage matrix, e.g. sch.linkage(dist, 'single'). Each row
        is (item1, item2, distance, num_original_items_in_new_cluster).

    Returns
    -------
    list of int
        Original item indices (0-indexed), sorted so that items merged
        early in the clustering process sit close together.
    """
    link = link.astype(int)
    sortIx = pd.Series([link[-1, 0], link[-1, 1]])
    numItems = link[-1, 3]                  # number of original items
    while sortIx.max() >= numItems:
        sortIx.index = range(0, sortIx.shape[0] * 2, 2)   # make space
        df0 = sortIx[sortIx >= numItems]                  # find clusters
        i = df0.index
        j = df0.values - numItems
        sortIx[i] = link[j, 0]                            # item 1
        df0 = pd.Series(link[j, 1], index=i + 1)
        # Book uses sortIx.append(df0); pandas 1.5.3 deprecates Series.append
        # (removed entirely in pandas 2.x) -- pd.concat is the direct
        # replacement with identical semantics here.
        sortIx = pd.concat([sortIx, df0])                 # item 2
        sortIx = sortIx.sort_index()                      # re-sort
        sortIx.index = range(sortIx.shape[0])              # re-index
    return sortIx.tolist()


# =============================================================================
# Snippet 16.3 / 16.4: getRecBipart
# =============================================================================
def getRecBipart(cov, sortIx):
    """Stage 3: recursive bisection (Section 16.4.3).

    Starting from ALL items in one cluster with unit weight, repeatedly
    bisect each cluster into two halves (preserving the quasi-diagonal
    order from stage 2), and split the weight between the two halves in
    INVERSE proportion to their variance (the half with more risk gets
    less weight). Recurses until every cluster is a single item.

    Parameters
    ----------
    cov : pd.DataFrame
        Covariance matrix, indexed/columned by asset name.
    sortIx : list
        Asset ordering from getQuasiDiag (translated back to asset names,
        not raw integer positions -- see chapter driver code).

    Returns
    -------
    pd.Series
        Final HRP weights, indexed by asset name, summing to 1.
    """
    # Book: w=pd.Series(1,index=sortIx) -- creates an INTEGER-dtype Series
    # (since 1 is an int). The loop below then does `w[...] *= alpha` with
    # a float alpha, which requires int64->float64 upcasting on assignment.
    # Older pandas did this silently; current pandas raises TypeError
    # outright ("Invalid value ... for dtype 'int64'"). Initializing as
    # 1.0 (float) from the start sidesteps the whole issue, independent of
    # which pandas version's upcasting rules apply.
    w = pd.Series(1.0, index=sortIx)
    cItems = [sortIx]                        # initialize: all items, one cluster
    while len(cItems) > 0:
        # Bi-section: split each cluster with >1 item into two halves.
        # Book: len(i)/2 -- true-divides to a float in Python 3, which
        # breaks list slicing. Fixed to floor division (len(i)//2), matching
        # Python 2's implicit int-division behaviour the book relied on.
        cItems = [i[j:k] for i in cItems for j, k in
                  ((0, len(i) // 2), (len(i) // 2, len(i))) if len(i) > 1]
        for i in range(0, len(cItems), 2):    # parse in pairs
            cItems0 = cItems[i]               # cluster 1
            cItems1 = cItems[i + 1]           # cluster 2
            cVar0 = getClusterVar(cov, cItems0)
            cVar1 = getClusterVar(cov, cItems1)
            alpha = 1 - cVar0 / (cVar0 + cVar1)
            w[cItems0] *= alpha               # weight 1
            w[cItems1] *= 1 - alpha           # weight 2
    return w


# =============================================================================
# Snippet 16.4: correlDist
# =============================================================================
def correlDist(corr):
    """A distance matrix based on correlation, 0 <= d[i,j] <= 1.

    d[i,j] = sqrt((1 - rho[i,j]) / 2). This is a proper metric (see
    Appendix 16.A.1 for the proof): non-negative, zero iff identical,
    symmetric, and sub-additive.
    """
    dist = ((1 - corr) / 2.) ** .5
    return dist


# =============================================================================
# Snippet 16.4: getHRP driver -- ties stages 1-3 together
# =============================================================================
def getHRP(cov, corr):
    """Run the full 3-stage HRP algorithm on a covariance/correlation pair.

    Note on Snippet 16.1's sch.linkage(dist, 'single') call: passing the
    raw 2D distance matrix (not a condensed/squareform vector) is
    deliberate -- scipy then computes the EUCLIDEAN distance BETWEEN ROWS
    of that matrix as its clustering distance, which is exactly the
    book's "distance of distances" d-tilde (Section 16.4.1, step 2). This
    is unusual enough that scipy raises a ClusterWarning ("looks
    suspiciously like an uncondensed distance matrix") -- suppressed here
    deliberately, since it IS an uncondensed distance matrix, on purpose.
    """
    corr = pd.DataFrame(corr)
    cov = pd.DataFrame(cov)
    dist = correlDist(corr)
    with warnings.catch_warnings():
        warnings.filterwarnings(
            'ignore', category=sch.ClusterWarning,
            message='.*uncondensed distance matrix.*'
        )
        link = sch.linkage(dist, 'single')
    sortIx = getQuasiDiag(link)
    sortIx = corr.index[sortIx].tolist()      # recover asset labels
    hrp = getRecBipart(cov, sortIx)
    return hrp.sort_index()


# =============================================================================
# Snippet 16.4: plotCorrMatrix
# =============================================================================
def plotCorrMatrix(path, corr, labels=None):
    """Heatmap of the correlation matrix, saved to `path`.

    Book uses matplotlib.pyplot as mpl; kept as a thin wrapper so callers
    can pass a live pyplot Axes-less save-to-file workflow, matching the
    book's demo script rather than returning a Figure for inline display
    (the chapter notebook wraps this separately for inline rendering).
    """
    import matplotlib.pyplot as mpl
    if labels is None:
        labels = []
    mpl.pcolor(corr)
    mpl.colorbar()
    mpl.yticks(np.arange(.5, corr.shape[0] + .5), labels)
    mpl.xticks(np.arange(.5, corr.shape[0] + .5), labels)
    mpl.savefig(path)
    mpl.clf()
    mpl.close()
    return


# =============================================================================
# Snippet 16.4: generateData (book's own synthetic-data generator)
# =============================================================================
def generateData(nObs, size0, size1, sigma1, random_state=None):
    """Simulate correlated time series: size0 uncorrelated base series, plus
    size1 further series each built as a perturbation of a randomly-chosen
    base series (so those size1 series are correlated with -- but not
    identical to -- one of the first size0).

    Book uses global np.random.seed/random.seed with a hardcoded 12345.
    Per project convention, this is threaded through as a seeded
    numpy.random.Generator instead of relying on legacy global state.
    """
    rng = random_state if random_state is not None else np.random.default_rng(12345)
    # 1) generate some uncorrelated data
    x = rng.normal(0, 1, size=(nObs, size0))     # each column is a variable
    # 2) create correlation between the variables
    cols = rng.integers(0, size0, size=size1)
    y = x[:, cols] + rng.normal(0, sigma1, size=(nObs, len(cols)))
    x = np.append(x, y, axis=1)
    x = pd.DataFrame(x, columns=range(1, x.shape[1] + 1))
    return x, cols
