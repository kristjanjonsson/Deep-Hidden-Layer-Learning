import numpy as np
from scipy.linalg import eigh


def gram(X):
    '''
    Calculates the Gram matrix XG of X.

    Parameters
    ----------
    X : (numSamples, numFeatures)
        Data matrix.

    Returns
    XG : (numSamples, numSamples)
         Gram matrix of X.
    '''
    return X.dot(X.T)


def effectiveDimension(eigvals, p=0.95):
    '''
    Calculates the the number of eigvals we need to take to explain > p of
    the variance of it's corresponding matrix.

    Parameters
    ----------
    eigvals : vector of eigenvalues from a P.S.D matrix.
              Must be sorted from largest to smallest.
    p : float between 0 and 1
        The percent of variance threshold.

    Returns
    d : int
        The smallest d such that eigvals[:d].sum() / eigvals.sum() > 0.95.
    '''
    total = eigvals.sum()
    percents = eigvals.cumsum() / total
    for i, pct in enumerate(percents):
        if pct > p:
            return i + 1
    raise ValueError('Impossible')


def factorize(G):
    '''
    Factorize G into H such that G = H.dot(H.T) using eigendecomposition.

    Parameters
    ----------
    G : (N, N)
         A positive semidefinite matrix.

    Returns
    ----------
    H : (N, M)
        The factorized matrix of G corresponding to nonzero eigenvalues.
    eigvals : (M, )
        The nonzero eigenvalues of G in descending order.
    '''
    eigvals, U = eigh(G)  # This is faster than SVD.

    # Change from ascending to descending order.
    eigvals, U = eigvals[::-1], U[:, ::-1]

    # Remove non-positive eigenvalues.
    # All should be >= 0 but some are < 0 so probably numerical issue.
    pos_idx = eigvals > 0
    eigvals = eigvals[pos_idx]
    U = U[:, pos_idx]

    sqrt_Lambda = np.diag(np.sqrt(eigvals))
    H = U.dot(sqrt_Lambda)
    return H, eigvals


def hiddenTargets(XG, YG, alpha):
    '''
    Returns the hidden targets for HG = (1 - alpha) * XG + alpha * YG.

    Parameters
    ----------
    XG : (N, N)
         Gram matrix for the inputs.
    YG : (N, N)
         Gram matrix for the outputs.
    alpha : float in [0, 1]

    Returns
    H : (N, N)
        Row i of H is the hidden target for sample i of the data.
    eigvals : (N, )
        The eigenvalues of HG in descending order.
    '''
    HG = (1 - alpha) * XG + alpha * YG
    return factorize(HG)
