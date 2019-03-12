"""
This module contains some extensions of numpy/pandas functionality (hence the name 'NumPy EXTensions').
"""
from scipy.special import gamma
import numpy as np


class Dirichlet:
    """
    An alternative Dirichlet Class which
        a) Precomputes the normalisation parameters for efficiency
        b) can vectorise over matrices, where the last-dimension is the probability distribution.
    """
    def __init__(self, alpha):
        """
        Initialise the Dirichlet with a particular alpha

        :param alpha: Alpha (Concentration) parameter. This is a K-D Matrix, where the last dimension is the probability
        """
        alpha = np.asarray(alpha)                                                   # K   Dimensional
        self.norm = np.prod(gamma(alpha), axis=-1) / gamma(np.sum(alpha, axis=-1))  # K-1 Dimensional
        self.lognorm = np.log(self.norm)                                            # K-1 Dimensional
        self.alpha = np.asarray(alpha) - 1.0                                        # K   Dimensional

    def pdf(self, x):
        """
        Compute the PDF of the passed Matrix (must be of the same dimensionality and size as Alpha)

        :param x: Probability Matrix
        :return: Dirichlet Probability on x
        """
        return np.prod(np.power(x, self.alpha), axis=-1) / self.norm

    def logpdf(self, x):
        """
        Compute the Log-PDF of the passed Matrix (must be of the same dimensionality and size as Alpha)

        :param x: Probability Matrix
        :return:  Log-Probability on X
        """
        return np.sum(np.multiply(np.log(x), self.alpha), axis=-1) - self.lognorm

    def logsumpdf(self, x):
        """
        Compute the Log-PDF of the passed Matrix (must be of the same dimensionality and size as Alpha) and sum it.

        :param x: Probability Matrix
        :return:  Log-Sum-Probability on X
        """
        return self.logpdf(x).sum()


def value_map(array, _to, _from=None, shuffle=False):
    """
    Map Values in a Numpy Array from a domain to another.

    Currently supports only 1D arrays

    :param array:  Array to map
    :param _to:    Values to map to
    :param _from:  Values to map from: if not present, it is assumed that the entries in to are ordered in their domain
                    [0, len(to)-1]
    :param shuffle: Optimisation: if True, does not need to use searchsorted (uses just array indexing). Only set this
                     to True iff _from contains all contiguous values from 0 up to len(_to) - 1 (also no duplicates).
    :return:       Mapped Array
    """
    # If just shuffling, then use just array indexing...
    if shuffle:
        if _from is not None:
            sort_idx = np.argsort(_from)
            return np.asarray(_to)[sort_idx][array]
        else:
            return np.asarray(_to)[array]

    # Otherwise, have to use search sorted: in this case, from cannot be None!
    else:
        sort_idx = np.argsort(_from)
        idx = np.searchsorted(_from, array, sorter=sort_idx)
        return np.asarray(_to)[sort_idx][idx]


def masked_logsumexp(a, axis=None, b=None, keepdims=False, return_sign=False):
    """
    Compute the log of the sum of exponentials of input elements. This is identical to the scipy.special function,
       but ignores Masked (NaN) values.

    Parameters
    ----------
    a : array_like
        Input array.
    axis : None or int or tuple of ints, optional
        Axis or axes over which the sum is taken. By default `axis` is None,
        and all elements are summed.
    b : array-like, optional
        Scaling factor for exp(`a`) must be of the same shape as `a` or
        broadcastable to `a`. These values may be negative in order to
        implement subtraction.
    keepdims : bool, optional
        If this is set to True, the axes which are reduced are left in the
        result as dimensions with size one. With this option, the result
        will broadcast correctly against the original array.
    return_sign : bool, optional
        If this is set to True, the result will be a pair containing sign
        information; if False, results that are negative will be returned
        as NaN. Default is False (no sign information).

    Returns
    -------
    res : ndarray
        The result, ``np.log(np.sum(np.exp(a)))`` calculated in a numerically
        more stable way. If `b` is given then ``np.log(np.sum(b*np.exp(a)))``
        is returned.
    sgn : ndarray
        If return_sign is True, this will be an array of floating-point
        numbers matching res and +1, 0, or -1 depending on the sign
        of the result. If False, only one result is returned.
    """
    if b is not None:
        a, b = np.broadcast_arrays(a, b)
        if np.any(b == 0):
            a = a + 0.  # promote to at least float
            a[b == 0] = -np.inf

    a_max = np.nanmax(a, axis=axis, keepdims=True)

    if a_max.ndim > 0:
        a_max[~np.isfinite(a_max)] = 0
    elif not np.isfinite(a_max):
        a_max = 0

    if b is not None:
        b = np.asarray(b)
        tmp = b * np.exp(a - a_max)
    else:
        tmp = np.exp(a - a_max)

    # suppress warnings about log of zero
    with np.errstate(divide='ignore'):
        s = np.nansum(tmp, axis=axis, keepdims=keepdims)
        if return_sign:
            sgn = np.sign(s)
            s *= sgn  # /= makes more sense but we need zero -> zero
        out = np.log(s)

    if not keepdims:
        a_max = np.squeeze(a_max, axis=axis)
    out += a_max

    if return_sign:
        return out, sgn
    else:
        return out


def sum_to_one(x, axis=None, norm=False):
    """
    Ensure that the elements of x sum to 1 (normally for probabilities), by dividing by their sum.

    The function avoids division by-zero errors.

    :param x:       Array to normalise
    :param axis:    If not None (default) specifies axis to normalise across: otherwise, normalisation happens across
                    flattened array.
    :param norm:    If Norm, return also the normaliser
    :return: Normalised Array
    """
    _sum = np.sum(x, axis=axis, keepdims=True)  # Find Sum
    _sum[_sum == 0] = 1.0                       # Avoid Division by Zero
    return (np.divide(x, _sum), 1.0/_sum.squeeze()) if norm else np.divide(x, _sum)


def maximise_trace(x):
    """
    Maximise the Trace of a SQUARE Matrix X using the Hungarian Algorithm

    :param x: Numpy 2D SQUARE Array
    :return: Tuple containing (in order):
                * optimal permutation of columns to achieve a maximal trace
                * size of this trace
    """
    _rows, _cols = hungarian(np.full(len(x), np.max(x)) - x)
    return _cols, x[_rows, _cols].sum()
