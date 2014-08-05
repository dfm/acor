__all__ = ["acor", "function"]

import numpy as np

from . import _acor


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


def function(data, maxt=None):
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
    data = np.atleast_1d(data)
    assert len(np.shape(data)) == 1, \
        "The autocorrelation function can only by computed " \
        + "on a 1D time series."
    if maxt is None:
        maxt = len(data)
    result = np.zeros(maxt, dtype=float)
    _acor.function(np.array(data, dtype=float), result)
    return result / result[0]
