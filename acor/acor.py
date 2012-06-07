__all__ = ["acor", "function"]

import numpy as np

import _acor


def acor(data, maxlag=10):
    """
    Estimate the autocorrelation time of a time series

    Parameters
    ----------
    data : numpy.ndarray (N,) or (M, N)
        The time series.

    maxlag : int, optional
        N must be greater than maxlag times the estimated autocorrelation
        time.

    Returns
    -------
    tau : float
        An estimate of the autocorrelation time.

    mean : float
        The sample mean of data.

    sigma : float
        An estimate of the standard deviation of the sample mean.

    """
    return _acor.acor(np.array(data), maxlag)


def function(data):
    """
    Calculate the autocorrelation function for a 1D time series.

    Parameters
    ----------
    data : numpy.ndarray (N,)
        The time series.

    Returns
    -------
    rho : numpy.ndarray (N,)
        An autocorrelation function.

    """
    N = len(data)
    mu = np.mean(data)
    result = np.zeros_like(data)

    for t in xrange(N):
        result[t] = np.mean((data[:N - t] - mu) * (data[t:] - mu))

    return result / result[0]
